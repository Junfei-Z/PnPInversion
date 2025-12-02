"""
Adaptive Inversion: Intelligently adjust num_inner_steps based on semantic distance

This module analyzes the semantic gap between source and target prompts,
and automatically determines the optimal number of inversion steps.

Key idea:
- Larger semantic gap → More inversion steps needed
- Smaller semantic gap → Fewer steps sufficient
"""

import torch
import numpy as np
from transformers import CLIPTokenizer, CLIPTextModel


class AdaptiveInversionController:
    """
    Intelligent controller for adaptive null-text inversion

    Features:
    1. Semantic distance analysis (CLIP similarity)
    2. Edit type detection (object/attribute/style)
    3. Token-level change analysis
    4. Adaptive step calculation with multiple strategies
    """

    def __init__(self, device='cuda'):
        """
        Initialize the adaptive controller

        Args:
            device: Device to run CLIP model
        """
        self.device = device

        # Load CLIP for semantic analysis
        print("Loading CLIP for adaptive inversion analysis...")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.text_encoder.eval()

        # Configuration for different strategies
        self.strategies = {
            'conservative': {
                'min_steps': 0,
                'max_steps': 10,
                'threshold_low': 0.9,    # Very similar prompts
                'threshold_high': 0.7    # Very different prompts
            },
            'moderate': {
                'min_steps': 0,
                'max_steps': 20,
                'threshold_low': 0.85,
                'threshold_high': 0.6
            },
            'aggressive': {
                'min_steps': 5,
                'max_steps': 30,
                'threshold_low': 0.8,
                'threshold_high': 0.5
            }
        }

        print("✓ Adaptive inversion controller initialized")

    def compute_clip_similarity(self, prompt_src, prompt_tar):
        """
        Compute CLIP text embedding similarity

        Args:
            prompt_src: Source prompt
            prompt_tar: Target prompt

        Returns:
            float: Cosine similarity (0-1, higher = more similar)
        """
        # Tokenize
        tokens_src = self.tokenizer(
            prompt_src,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        tokens_tar = self.tokenizer(
            prompt_tar,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Get embeddings
        with torch.no_grad():
            emb_src = self.text_encoder(**tokens_src).pooler_output  # [1, 768]
            emb_tar = self.text_encoder(**tokens_tar).pooler_output  # [1, 768]

        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb_src, emb_tar, dim=1)

        return similarity.item()

    def analyze_token_changes(self, prompt_src, prompt_tar):
        """
        Analyze token-level changes between prompts

        Returns:
            dict: Statistics about token changes
        """
        # Tokenize (without padding for analysis)
        tokens_src = self.tokenizer.encode(prompt_src, add_special_tokens=False)
        tokens_tar = self.tokenizer.encode(prompt_tar, add_special_tokens=False)

        # Convert to sets for comparison
        set_src = set(tokens_src)
        set_tar = set(tokens_tar)

        # Compute statistics
        added_tokens = set_tar - set_src
        removed_tokens = set_src - set_tar
        common_tokens = set_src & set_tar

        total_tokens = len(set_src | set_tar)
        change_ratio = len(added_tokens | removed_tokens) / max(total_tokens, 1)

        return {
            'num_added': len(added_tokens),
            'num_removed': len(removed_tokens),
            'num_common': len(common_tokens),
            'change_ratio': change_ratio,
            'total_src': len(tokens_src),
            'total_tar': len(tokens_tar)
        }

    def detect_edit_type(self, prompt_src, prompt_tar, token_stats):
        """
        Detect the type of edit based on prompt analysis

        Returns:
            str: Edit type (object_replacement, attribute_change, style_transfer, minor_edit)
        """
        change_ratio = token_stats['change_ratio']
        num_added = token_stats['num_added']
        num_removed = token_stats['num_removed']

        # Object replacement: many tokens changed
        if change_ratio > 0.4 and num_added >= 2 and num_removed >= 2:
            return 'object_replacement'

        # Attribute change: some tokens changed
        elif change_ratio > 0.2:
            return 'attribute_change'

        # Style transfer: moderate changes
        elif change_ratio > 0.1:
            return 'style_transfer'

        # Minor edit
        else:
            return 'minor_edit'

    def calculate_num_inner_steps(
        self,
        prompt_src,
        prompt_tar,
        strategy='moderate',
        verbose=True
    ):
        """
        Main function: Calculate optimal num_inner_steps

        Args:
            prompt_src: Source prompt
            prompt_tar: Target prompt
            strategy: 'conservative', 'moderate', or 'aggressive'
            verbose: Print analysis details

        Returns:
            int: Optimal number of inner steps
        """
        if verbose:
            print("\n" + "="*80)
            print("ADAPTIVE INVERSION ANALYSIS")
            print("="*80)
            print(f"Source: {prompt_src}")
            print(f"Target: {prompt_tar}")
            print(f"Strategy: {strategy}")

        # 1. Compute CLIP similarity
        clip_sim = self.compute_clip_similarity(prompt_src, prompt_tar)

        # 2. Analyze token changes
        token_stats = self.analyze_token_changes(prompt_src, prompt_tar)

        # 3. Detect edit type
        edit_type = self.detect_edit_type(prompt_src, prompt_tar, token_stats)

        if verbose:
            print("-"*80)
            print("Analysis:")
            print(f"  CLIP Similarity: {clip_sim:.4f}")
            print(f"  Token Change Ratio: {token_stats['change_ratio']:.2%}")
            print(f"  Tokens Added: {token_stats['num_added']}")
            print(f"  Tokens Removed: {token_stats['num_removed']}")
            print(f"  Edit Type: {edit_type}")

        # 4. Calculate num_inner_steps based on strategy
        config = self.strategies[strategy]

        # Base calculation from CLIP similarity
        # Lower similarity → More steps needed
        if clip_sim >= config['threshold_low']:
            # Very similar: minimum steps
            base_steps = config['min_steps']
        elif clip_sim <= config['threshold_high']:
            # Very different: maximum steps
            base_steps = config['max_steps']
        else:
            # Linear interpolation
            ratio = (config['threshold_low'] - clip_sim) / (config['threshold_low'] - config['threshold_high'])
            base_steps = int(config['min_steps'] + ratio * (config['max_steps'] - config['min_steps']))

        # 5. Adjust based on edit type
        type_multipliers = {
            'object_replacement': 1.5,   # Hardest: need more steps
            'attribute_change': 1.2,
            'style_transfer': 1.0,
            'minor_edit': 0.8            # Easiest: fewer steps
        }

        adjusted_steps = int(base_steps * type_multipliers[edit_type])

        # 6. Adjust based on token change ratio
        if token_stats['change_ratio'] > 0.5:
            adjusted_steps = int(adjusted_steps * 1.3)

        # 7. Clamp to valid range
        final_steps = np.clip(adjusted_steps, config['min_steps'], config['max_steps'])

        if verbose:
            print("-"*80)
            print("Decision:")
            print(f"  Base Steps (from similarity): {base_steps}")
            print(f"  Type Adjustment (×{type_multipliers[edit_type]}): {adjusted_steps}")
            print(f"  Final num_inner_steps: {final_steps}")
            print("="*80 + "\n")

        return final_steps

    def get_recommendations(self, prompt_src, prompt_tar):
        """
        Get recommendations for all strategies

        Returns:
            dict: Recommendations for each strategy
        """
        recommendations = {}

        for strategy_name in self.strategies.keys():
            steps = self.calculate_num_inner_steps(
                prompt_src,
                prompt_tar,
                strategy=strategy_name,
                verbose=False
            )
            recommendations[strategy_name] = steps

        return recommendations


# Convenience function for quick use
def get_adaptive_num_inner_steps(
    prompt_src,
    prompt_tar,
    tokenizer,
    strategy='moderate',
    verbose=True,
    device='cuda'
):
    """
    Quick function to get adaptive num_inner_steps

    Args:
        prompt_src: Source prompt
        prompt_tar: Target prompt
        tokenizer: CLIP tokenizer (can be None, will load if needed)
        strategy: 'conservative', 'moderate', or 'aggressive'
        verbose: Print analysis
        device: Device

    Returns:
        int: Optimal num_inner_steps
    """
    # Create controller (will cache in practice)
    controller = AdaptiveInversionController(device=device)

    # Calculate steps
    num_steps = controller.calculate_num_inner_steps(
        prompt_src,
        prompt_tar,
        strategy=strategy,
        verbose=verbose
    )

    return num_steps


if __name__ == "__main__":
    # Demo
    controller = AdaptiveInversionController(device='cpu')

    # Test cases
    test_cases = [
        ("a photo of a cat", "a photo of a dog"),           # Object replacement
        ("a smiling person", "an angry person"),             # Attribute change
        ("a photo of a house", "a painting of a house"),     # Style transfer
        ("a red car", "a blue car"),                         # Minor edit
    ]

    print("\n" + "🎯"*40)
    print("ADAPTIVE INVERSION DEMO")
    print("🎯"*40 + "\n")

    for src, tar in test_cases:
        steps = controller.calculate_num_inner_steps(src, tar, strategy='moderate')
        print(f"\n{'='*80}\n")

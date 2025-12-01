"""
Semantic-Aware Adaptive Attention Control

This module provides automatic adjustment of P2P attention control parameters
based on semantic analysis of source and target prompts.

Theory:
- Cross-attention controls editing strength
- Self-attention controls structural consistency
- Different edits require different parameter configurations
"""

import torch
from typing import Tuple, Optional


def compute_prompt_difference(
    prompt_src: str,
    prompt_tar: str,
    tokenizer
) -> float:
    """
    Calculate semantic difference between two prompts using token-based Jaccard distance.

    Args:
        prompt_src: Source prompt text
        prompt_tar: Target prompt text
        tokenizer: CLIP tokenizer from Stable Diffusion pipeline

    Returns:
        difference: Semantic difference in [0, 1]
                   0 = identical prompts
                   1 = completely different prompts

    Example:
        >>> diff = compute_prompt_difference("a cat", "a white cat", tokenizer)
        >>> # diff ≈ 0.2 (small change)
        >>> diff = compute_prompt_difference("a cat", "a house", tokenizer)
        >>> # diff ≈ 0.8 (large change)
    """
    # Tokenize both prompts
    tokens_src = set(tokenizer.encode(prompt_src))
    tokens_tar = set(tokenizer.encode(prompt_tar))

    # Calculate Jaccard distance
    intersection = len(tokens_src & tokens_tar)
    union = len(tokens_src | tokens_tar)

    if union == 0:
        return 0.0

    # Jaccard similarity
    similarity = intersection / union

    # Convert to difference (distance)
    difference = 1.0 - similarity

    return difference


def adaptive_cross_replace_steps(
    prompt_src: str,
    prompt_tar: str,
    tokenizer,
    edit_type_id: Optional[str] = None
) -> float:
    """
    Adaptively determine cross_replace_steps based on semantic difference.

    Theory (from Prompt-to-Prompt paper):
    - Cross-attention maps control which regions are edited
    - Higher cross_replace_steps = preserve more from source image
    - Lower cross_replace_steps = allow more edits

    Strategy:
    - Small semantic change (e.g., "cat" → "white cat"): high preservation (0.7-0.8)
    - Medium change (e.g., "cat" → "dog"): moderate (0.4-0.5)
    - Large change (e.g., "cat" → "house"): low preservation (0.2-0.3)

    Args:
        prompt_src: Source prompt
        prompt_tar: Target prompt
        tokenizer: CLIP tokenizer
        edit_type_id: PIE-Bench editing type ID (optional, for fine-tuning)

    Returns:
        cross_replace_steps: Recommended value in [0.2, 0.8]
    """
    diff = compute_prompt_difference(prompt_src, prompt_tar, tokenizer)

    # Base adjustment based on semantic difference
    if diff < 0.2:
        # Very small change: preserve most structure
        # Example: "a cat" → "a fluffy cat"
        base_steps = 0.8
    elif diff < 0.4:
        # Medium change: moderate preservation
        # Example: "a cat" → "a dog"
        base_steps = 0.5
    elif diff < 0.6:
        # Large change: allow more editing
        # Example: "a cat" → "a car"
        base_steps = 0.3
    else:
        # Very large change: minimal preservation
        # Example: "a cat on grass" → "a building at night"
        base_steps = 0.2

    # Fine-tune based on PIE-Bench edit type
    if edit_type_id is not None:
        if edit_type_id in ['4', '6', '7']:
            # Attribute edits (content/color/material)
            # These should preserve structure more
            base_steps = min(0.8, base_steps + 0.1)
        elif edit_type_id in ['1', '2', '3']:
            # Object edits (change/add/delete object)
            # These need more freedom to change
            base_steps = max(0.2, base_steps - 0.1)
        elif edit_type_id == '5':
            # Pose changes: moderate adjustment
            pass  # Keep base_steps
        elif edit_type_id in ['8', '9']:
            # Background/style changes: need more freedom
            base_steps = max(0.2, base_steps - 0.05)

    # Ensure within valid range
    return max(0.2, min(0.8, base_steps))


def adaptive_self_replace_steps(
    prompt_src: str,
    prompt_tar: str,
    tokenizer
) -> float:
    """
    Adaptively determine self_replace_steps based on semantic difference.

    Theory:
    - Self-attention controls spatial structure and layout
    - Should generally be higher than cross_replace_steps
    - Less sensitive to prompt changes than cross-attention

    Args:
        prompt_src: Source prompt
        prompt_tar: Target prompt
        tokenizer: CLIP tokenizer

    Returns:
        self_replace_steps: Recommended value in [0.4, 0.9]
    """
    diff = compute_prompt_difference(prompt_src, prompt_tar, tokenizer)

    if diff < 0.3:
        # Small change: preserve structure heavily
        return 0.9
    elif diff < 0.6:
        # Medium change: moderate preservation
        return 0.7
    else:
        # Large change: allow structural changes
        return 0.5


def get_adaptive_parameters(
    prompt_src: str,
    prompt_tar: str,
    tokenizer,
    edit_type_id: Optional[str] = None,
    verbose: bool = True
) -> Tuple[float, float]:
    """
    Get both cross and self replace steps in one call.

    Args:
        prompt_src: Source prompt
        prompt_tar: Target prompt
        tokenizer: CLIP tokenizer
        edit_type_id: PIE-Bench editing type ID
        verbose: Whether to print adjustment info

    Returns:
        (cross_replace_steps, self_replace_steps)
    """
    cross = adaptive_cross_replace_steps(prompt_src, prompt_tar, tokenizer, edit_type_id)
    self_repl = adaptive_self_replace_steps(prompt_src, prompt_tar, tokenizer)

    if verbose:
        diff = compute_prompt_difference(prompt_src, prompt_tar, tokenizer)
        print(f"[Adaptive Control]")
        print(f"  Semantic difference: {diff:.3f}")
        print(f"  Cross-replace steps: {cross:.2f}")
        print(f"  Self-replace steps:  {self_repl:.2f}")
        if edit_type_id:
            print(f"  Edit type: {edit_type_id}")

    return cross, self_repl

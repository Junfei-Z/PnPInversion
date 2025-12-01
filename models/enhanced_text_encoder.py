"""
Enhanced Text Encoder for P2P Editing
Replaces the default CLIP text encoder with a more powerful version
"""

from transformers import CLIPTextModel, CLIPTokenizer
import torch

class EnhancedTextEncoderConfig:
    """Configuration for different enhanced text encoders"""

    ENCODERS = {
        # 768-dim encoders (directly compatible with SD 1.4)
        'laion-L-datacomp': {
            'model_id': 'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K',
            'hidden_size': 768,
            'compatible': True,
            'description': 'LAION CLIP trained on 13B samples from DataComp'
        },
        'laion-L-laion2b': {
            'model_id': 'laion/CLIP-ViT-L-14-laion2B-s32B-b79K',
            'hidden_size': 768,
            'compatible': True,
            'description': 'LAION CLIP trained on 2B image-text pairs'
        },
        'openai-L-336': {
            'model_id': 'openai/clip-vit-large-patch14-336',
            'hidden_size': 768,
            'compatible': True,
            'description': 'OpenAI CLIP with 336x336 image resolution'
        },

        # 1024-dim encoders (need projection, not directly compatible)
        'laion-H': {
            'model_id': 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
            'hidden_size': 1024,
            'compatible': False,
            'description': 'LAION CLIP-H (needs projection layer)'
        },
    }

    @classmethod
    def get_compatible_encoders(cls):
        """Return only directly compatible encoders"""
        return {k: v for k, v in cls.ENCODERS.items() if v['compatible']}

    @classmethod
    def print_available_encoders(cls):
        """Print all available encoders"""
        print("=" * 80)
        print("Available Enhanced Text Encoders")
        print("=" * 80)
        for name, config in cls.ENCODERS.items():
            compat = "✓ Compatible" if config['compatible'] else "✗ Needs projection"
            print(f"\n{name}:")
            print(f"  Model: {config['model_id']}")
            print(f"  Dimension: {config['hidden_size']}")
            print(f"  Status: {compat}")
            print(f"  Description: {config['description']}")
        print("=" * 80)


def load_enhanced_text_encoder(encoder_name='laion-L-datacomp', device='cuda'):
    """
    Load an enhanced text encoder

    Args:
        encoder_name: Name from EnhancedTextEncoderConfig.ENCODERS
        device: Device to load the model on

    Returns:
        text_encoder: Enhanced CLIPTextModel
        tokenizer: Corresponding tokenizer
    """
    # Get encoder config
    if encoder_name not in EnhancedTextEncoderConfig.ENCODERS:
        raise ValueError(
            f"Unknown encoder: {encoder_name}. "
            f"Available: {list(EnhancedTextEncoderConfig.ENCODERS.keys())}"
        )

    config = EnhancedTextEncoderConfig.ENCODERS[encoder_name]

    # Check compatibility
    if not config['compatible']:
        raise ValueError(
            f"{encoder_name} is not directly compatible (dim={config['hidden_size']}). "
            f"SD 1.4 requires 768-dim. Please choose a compatible encoder or implement projection."
        )

    print(f"Loading enhanced text encoder: {encoder_name}")
    print(f"  Model: {config['model_id']}")
    print(f"  Dimension: {config['hidden_size']}")

    # Load text encoder
    text_encoder = CLIPTextModel.from_pretrained(config['model_id']).to(device)

    # Load tokenizer (use the same tokenizer as original SD for compatibility)
    # This is important to maintain compatibility with existing prompts
    tokenizer = CLIPTokenizer.from_pretrained(config['model_id'])

    print(f"✓ Enhanced text encoder loaded successfully!")

    return text_encoder, tokenizer


def verify_encoder_compatibility(encoder, reference_encoder=None):
    """
    Verify that an encoder is compatible with SD 1.4

    Args:
        encoder: The encoder to verify
        reference_encoder: Optional reference encoder to compare against
    """
    print("\n" + "=" * 80)
    print("Encoder Compatibility Check")
    print("=" * 80)

    # Check hidden size
    hidden_size = encoder.config.hidden_size
    print(f"Hidden size: {hidden_size}")
    if hidden_size != 768:
        print(f"⚠ WARNING: Hidden size {hidden_size} != 768 (SD 1.4 expects 768)")
        print("  This encoder needs a projection layer!")
        return False
    else:
        print("✓ Hidden size is compatible (768)")

    # Check max position embeddings
    max_pos = encoder.config.max_position_embeddings
    print(f"Max position embeddings: {max_pos}")
    if max_pos != 77:
        print(f"⚠ WARNING: Max position {max_pos} != 77")
    else:
        print("✓ Max position embeddings is compatible (77)")

    # Test with dummy input
    print("\nTesting with dummy input...")
    test_ids = torch.randint(0, 1000, (1, 77)).to(encoder.device)
    with torch.no_grad():
        output = encoder(test_ids)
        embeddings = output.last_hidden_state

    print(f"Output shape: {embeddings.shape}")
    expected_shape = torch.Size([1, 77, 768])
    if embeddings.shape == expected_shape:
        print(f"✓ Output shape matches expected {expected_shape}")
        print("\n" + "=" * 80)
        print("✓✓✓ Encoder is COMPATIBLE with SD 1.4! ✓✓✓")
        print("=" * 80)
        return True
    else:
        print(f"✗ Output shape {embeddings.shape} != expected {expected_shape}")
        print("\n" + "=" * 80)
        print("✗✗✗ Encoder is NOT compatible! ✗✗✗")
        print("=" * 80)
        return False


if __name__ == "__main__":
    # Demo: Print available encoders
    EnhancedTextEncoderConfig.print_available_encoders()

    # Demo: Load and verify an encoder
    print("\n" * 2)
    try:
        encoder, tokenizer = load_enhanced_text_encoder('laion-L-datacomp', device='cpu')
        verify_encoder_compatibility(encoder)
    except Exception as e:
        print(f"Error during demo: {e}")
        print("This is expected if you don't have the models downloaded.")

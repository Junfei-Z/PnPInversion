"""
Demo: Using Enhanced Text Encoder with P2P Editor

This script demonstrates how to use the enhanced text encoder feature.
"""

from models.p2p_editor import P2PEditor
from models.enhanced_text_encoder import EnhancedTextEncoderConfig
import torch

def main():
    # Print available enhanced encoders
    print("=" * 80)
    print("STEP 1: Check Available Enhanced Text Encoders")
    print("=" * 80)
    EnhancedTextEncoderConfig.print_available_encoders()

    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Choose an enhanced encoder
    # Options: 'laion-L-datacomp', 'laion-L-laion2b', 'openai-L-336', or None for baseline
    enhanced_encoder_name = 'laion-L-datacomp'  # ⭐ Change this to test different encoders

    print("\n" + "=" * 80)
    print("STEP 2: Initialize P2P Editor")
    print("=" * 80)

    # Baseline: without enhanced encoder
    if enhanced_encoder_name is None:
        print("Creating P2P Editor with ORIGINAL text encoder...")
        editor = P2PEditor(
            method_list=["ddim+p2p"],
            device=device,
            num_ddim_steps=50,
            enhanced_encoder=None  # Use original encoder
        )
        print("✓ Using original OpenAI CLIP encoder")

    # Enhanced: with enhanced encoder
    else:
        print(f"Creating P2P Editor with ENHANCED text encoder: {enhanced_encoder_name}")
        editor = P2PEditor(
            method_list=["ddim+p2p"],
            device=device,
            num_ddim_steps=50,
            enhanced_encoder=enhanced_encoder_name  # ⭐ Use enhanced encoder
        )

    print("\n" + "=" * 80)
    print("STEP 3: Ready to Edit Images!")
    print("=" * 80)

    # Example usage (commented out to avoid actual execution)
    """
    result = editor(
        edit_method="ddim+p2p",
        image_path="path/to/image.jpg",
        prompt_src="a photo of a cat",
        prompt_tar="a photo of a dog",
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6
    )
    """

    print("\n" + "=" * 80)
    print("Usage Example:")
    print("=" * 80)
    print("""
    # Initialize with enhanced encoder
    editor = P2PEditor(
        method_list=["ddim+p2p"],
        device="cuda",
        num_ddim_steps=50,
        enhanced_encoder='laion-L-datacomp'  # ⭐ Key parameter
    )

    # Use it the same way as before
    panel_image, edited_image = editor(
        edit_method="ddim+p2p",
        image_path="example.jpg",
        prompt_src="a photo of a cat",
        prompt_tar="a photo of a tiger",
        guidance_scale=7.5
    )
    """)

    print("\n" + "=" * 80)
    print("Comparison Tips:")
    print("=" * 80)
    print("""
    To compare baseline vs enhanced:

    1. Run with enhanced_encoder=None (baseline)
       - Evaluate on PIE-Bench
       - Record SSIM, LPIPS, CLIP scores

    2. Run with enhanced_encoder='laion-L-datacomp'
       - Evaluate on PIE-Bench
       - Record SSIM, LPIPS, CLIP scores

    3. Compare the results!
       - Better CLIP score = better text-image alignment
       - Higher SSIM = better structure preservation
       - Lower LPIPS = better perceptual quality
    """)

    return editor


if __name__ == "__main__":
    # Run the demo
    try:
        editor = main()
        print("\n✓✓✓ Demo completed successfully! ✓✓✓\n")
    except Exception as e:
        print(f"\n✗✗✗ Error: {e} ✗✗✗")
        print("\nThis is expected if you don't have the models downloaded yet.")
        print("The models will be automatically downloaded when you first use them.")

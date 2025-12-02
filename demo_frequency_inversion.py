"""
Quick Demo: Frequency-Preserving P2P Editing

This script demonstrates the frequency-preserving inversion on a single image.
Use this to quickly test the method before running on the full dataset.
"""

import torch
from models.p2p_editor_frequency import P2PEditorFrequency
import argparse


def main():
    parser = argparse.ArgumentParser(description="Quick demo of frequency-preserving editing")
    parser.add_argument('--image_path', type=str, required=True,
                        help="Path to input image")
    parser.add_argument('--prompt_src', type=str, required=True,
                        help="Source prompt describing the image")
    parser.add_argument('--prompt_tar', type=str, required=True,
                        help="Target prompt for editing")
    parser.add_argument('--output_path', type=str, default="demo_output.png",
                        help="Path to save output panel")
    parser.add_argument('--output_edited', type=str, default="demo_edited.png",
                        help="Path to save edited image only")
    parser.add_argument('--freq_weight', type=float, default=0.3,
                        help="Frequency preservation weight (0-1)")
    parser.add_argument('--num_inner_steps', type=int, default=10,
                        help="Number of optimization steps")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device to use (cuda or cpu)")

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("FREQUENCY-PRESERVING P2P EDITING - DEMO")
    print(f"{'='*80}")
    print(f"Image: {args.image_path}")
    print(f"Source prompt: {args.prompt_src}")
    print(f"Target prompt: {args.prompt_tar}")
    print(f"Frequency weight: {args.freq_weight}")
    print(f"Num inner steps: {args.num_inner_steps}")
    print(f"{'='*80}\n")

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = 'cpu'
    else:
        device = args.device

    # Initialize editor
    editor = P2PEditorFrequency(
        device=device,
        num_ddim_steps=50,
        freq_weight=args.freq_weight
    )

    # Run editing
    panel_image, edited_image = editor.edit_image(
        image_path=args.image_path,
        prompt_src=args.prompt_src,
        prompt_tar=args.prompt_tar,
        num_inner_steps=args.num_inner_steps,
        guidance_scale=7.5,
        cross_replace_steps=0.4,
        self_replace_steps=0.6,
    )

    # Save results
    panel_image.save(args.output_path)
    edited_image.save(args.output_edited)

    print(f"\n{'='*80}")
    print("RESULTS SAVED")
    print(f"{'='*80}")
    print(f"Panel (4 images): {args.output_path}")
    print(f"Edited only: {args.output_edited}")
    print(f"{'='*80}\n")

    print("✓ Demo completed successfully!")


if __name__ == "__main__":
    main()


"""
Example usage:

# Basic usage
python demo_frequency_inversion.py \
    --image_path data/annotation_images/0_random_140/000000000000.jpg \
    --prompt_src "a photo of a cat" \
    --prompt_tar "a photo of a dog" \
    --output_path results/demo_panel.png \
    --output_edited results/demo_edited.png

# Adjust frequency weight
python demo_frequency_inversion.py \
    --image_path your_image.jpg \
    --prompt_src "a red car" \
    --prompt_tar "a blue car" \
    --freq_weight 0.5 \
    --output_path demo_w05.png

# More optimization steps
python demo_frequency_inversion.py \
    --image_path your_image.jpg \
    --prompt_src "a smiling person" \
    --prompt_tar "an angry person" \
    --num_inner_steps 15 \
    --output_path demo_s15.png
"""

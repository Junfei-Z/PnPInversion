"""
Simple single image evaluation script
Calculate SSIM, LPIPS, CLIP similarity for a single edited image
"""

import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path


def load_image(image_path):
    """Load image and convert to tensor [1, 3, H, W]"""
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img).astype(np.float32) / 255.0  # [H, W, 3]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    return img, img_tensor


def calculate_ssim(img1_tensor, img2_tensor, device='cuda'):
    """Calculate SSIM between two images"""
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        img1 = img1_tensor.to(device)
        img2 = img2_tensor.to(device)

        with torch.no_grad():
            score = ssim(img1, img2).item()

        return score
    except Exception as e:
        print(f"SSIM calculation failed: {e}")
        return None


def calculate_lpips(img1_tensor, img2_tensor, device='cuda'):
    """Calculate LPIPS between two images"""
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex').to(device)

        # LPIPS expects [-1, 1] range
        img1 = img1_tensor.to(device) * 2 - 1
        img2 = img2_tensor.to(device) * 2 - 1

        with torch.no_grad():
            score = lpips_fn(img1, img2).item()

        return score
    except Exception as e:
        print(f"LPIPS calculation failed: {e}")
        return None


def calculate_clip_similarity(image_pil, text_prompt, device='cuda'):
    """Calculate CLIP similarity between image and text"""
    try:
        from transformers import CLIPProcessor, CLIPModel

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        inputs = processor(text=[text_prompt], images=image_pil, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            score = probs[0][0].item()

        return score
    except Exception as e:
        print(f"CLIP calculation failed: {e}")
        return None


def evaluate_single_image(source_path, edited_path, target_prompt, device='cuda'):
    """
    Evaluate a single edited image

    Args:
        source_path: Path to source image
        edited_path: Path to edited image
        target_prompt: Target text prompt
        device: Device to use

    Returns:
        dict: Dictionary containing metrics
    """
    print(f"Evaluating: {edited_path}")

    # Load images
    source_pil, source_tensor = load_image(source_path)
    edited_pil, edited_tensor = load_image(edited_path)

    # Ensure same size
    if source_tensor.shape != edited_tensor.shape:
        print(f"Warning: Images have different sizes, resizing edited to match source")
        edited_pil = edited_pil.resize(source_pil.size, Image.BILINEAR)
        _, edited_tensor = load_image(edited_path)

    results = {
        'source_path': source_path,
        'edited_path': edited_path,
        'target_prompt': target_prompt,
    }

    # Calculate SSIM
    print("  Computing SSIM...")
    ssim_score = calculate_ssim(source_tensor, edited_tensor, device)
    results['ssim'] = ssim_score
    if ssim_score is not None:
        print(f"    SSIM: {ssim_score:.4f}")

    # Calculate LPIPS
    print("  Computing LPIPS...")
    lpips_score = calculate_lpips(source_tensor, edited_tensor, device)
    results['lpips'] = lpips_score
    if lpips_score is not None:
        print(f"    LPIPS: {lpips_score:.4f}")

    # Calculate CLIP similarity
    print("  Computing CLIP similarity...")
    clip_score = calculate_clip_similarity(edited_pil, target_prompt, device)
    results['clip_similarity'] = clip_score
    if clip_score is not None:
        print(f"    CLIP: {clip_score:.4f}")

    return results


def evaluate_directory(source_dir, edited_dir, prompts_dict, device='cuda'):
    """
    Evaluate all images in a directory

    Args:
        source_dir: Directory containing source images
        edited_dir: Directory containing edited images
        prompts_dict: Dictionary mapping image names to target prompts
        device: Device to use

    Returns:
        list: List of result dictionaries
    """
    source_dir = Path(source_dir)
    edited_dir = Path(edited_dir)

    results = []

    for edited_path in edited_dir.glob('**/*.jpg'):
        # Find corresponding source image
        rel_path = edited_path.relative_to(edited_dir)
        source_path = source_dir / rel_path

        if not source_path.exists():
            print(f"Warning: Source image not found: {source_path}")
            continue

        # Get target prompt
        image_name = edited_path.name
        target_prompt = prompts_dict.get(image_name, "")

        if not target_prompt:
            print(f"Warning: No prompt for {image_name}")
            continue

        # Evaluate
        result = evaluate_single_image(str(source_path), str(edited_path), target_prompt, device)
        results.append(result)

    return results


def print_summary(results):
    """Print summary statistics"""
    if not results:
        print("No results to summarize")
        return

    ssim_vals = [r['ssim'] for r in results if r['ssim'] is not None]
    lpips_vals = [r['lpips'] for r in results if r['lpips'] is not None]
    clip_vals = [r['clip_similarity'] for r in results if r['clip_similarity'] is not None]

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Number of images: {len(results)}")

    if ssim_vals:
        print(f"SSIM:  Mean={np.mean(ssim_vals):.4f}, Std={np.std(ssim_vals):.4f}")
    if lpips_vals:
        print(f"LPIPS: Mean={np.mean(lpips_vals):.4f}, Std={np.std(lpips_vals):.4f}")
    if clip_vals:
        print(f"CLIP:  Mean={np.mean(clip_vals):.4f}, Std={np.std(clip_vals):.4f}")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate single image or directory")
    parser.add_argument('--source', type=str, required=True,
                       help='Source image or directory')
    parser.add_argument('--edited', type=str, required=True,
                       help='Edited image or directory')
    parser.add_argument('--prompt', type=str, default="",
                       help='Target text prompt (for single image)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    source_path = Path(args.source)
    edited_path = Path(args.edited)

    # Single image evaluation
    if source_path.is_file() and edited_path.is_file():
        result = evaluate_single_image(
            str(source_path),
            str(edited_path),
            args.prompt,
            device
        )

        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"Source: {result['source_path']}")
        print(f"Edited: {result['edited_path']}")
        print(f"Prompt: {result['target_prompt']}")
        print(f"SSIM:  {result['ssim']:.4f}" if result['ssim'] else "SSIM:  N/A")
        print(f"LPIPS: {result['lpips']:.4f}" if result['lpips'] else "LPIPS: N/A")
        print(f"CLIP:  {result['clip_similarity']:.4f}" if result['clip_similarity'] else "CLIP:  N/A")
        print("="*80)

    # Directory evaluation (requires prompts file)
    elif source_path.is_dir() and edited_path.is_dir():
        print("Directory evaluation not yet implemented")
        print("Please use single image mode for now")

    else:
        print("Error: Invalid source or edited path")


if __name__ == "__main__":
    main()

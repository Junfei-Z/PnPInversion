"""
Complete Comparison Experiment: Baseline vs Enhanced Text Encoder

This script:
1. Runs baseline (original CLIP) on first 10 images from 0_random_140
2. Runs enhanced (LAION CLIP) on the same 10 images
3. Evaluates both using eval_p2p_metrics_official.py
4. Generates comparison report

Author: Enhanced Text Encoder Improvement
"""

import os
import json
import argparse
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

from models.p2p_editor import P2PEditor
from utils.utils import load_512


def load_pie_bench_samples(data_path, category="0", num_samples=10):
    """
    Load samples from PIE-Bench mapping file

    Args:
        data_path: Path to PIE-Bench data directory
        category: Category ID (e.g., "0" for 0_random_140)
        num_samples: Number of samples to load

    Returns:
        List of sample dictionaries
    """
    mapping_file = os.path.join(data_path, "mapping_file.json")

    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

    with open(mapping_file, "r") as f:
        all_data = json.load(f)

    # Filter by category and take first num_samples
    samples = []
    for key, item in all_data.items():
        if item["editing_type_id"] == category:
            item["id"] = key
            samples.append(item)

            if len(samples) >= num_samples:
                break

    print(f"Loaded {len(samples)} samples from category {category}")
    return samples


def run_editing_batch(editor, samples, data_path, output_dir, edit_method="ddim+p2p", exp_name="baseline"):
    """
    Run editing on a batch of samples

    Args:
        editor: P2PEditor instance
        samples: List of sample dictionaries
        data_path: PIE-Bench data directory
        output_dir: Output directory for edited images
        edit_method: Editing method to use
        exp_name: Experiment name (baseline or enhanced)
    """
    print(f"\n{'='*80}")
    print(f"Running {exp_name.upper()} experiment")
    print(f"{'='*80}")

    # Create output directory structure matching eval script expectations
    # Structure: output_dir/ddim+p2p_only/annotation_images/...
    save_dir = os.path.join(output_dir, exp_name, "ddim+p2p_only")

    results = []

    for sample in tqdm(samples, desc=f"{exp_name} editing"):
        # Get paths
        image_path = os.path.join(data_path, "annotation_images", sample["image_path"])

        if not os.path.exists(image_path):
            print(f"[Warning] Image not found: {image_path}")
            continue

        # Prepare prompts
        original_prompt = sample["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = sample["editing_prompt"].replace("[", "").replace("]", "")

        # Get blend word if exists
        blend_word = None
        if "blended_word" in sample and sample["blended_word"]:
            blend_word = (
                (sample["blended_word"].split(" ")[0], ),
                (sample["blended_word"].split(" ")[1], )
            )

        try:
            # Run editing
            panel_image, edited_image = editor(
                edit_method=edit_method,
                image_path=image_path,
                prompt_src=original_prompt,
                prompt_tar=editing_prompt,
                guidance_scale=7.5,
                cross_replace_steps=0.4,
                self_replace_steps=0.6,
                blend_word=blend_word,
                edit_type_id=sample["editing_type_id"]
            )

            # Save edited image (single, not panel) to match eval script expectations
            # Path: save_dir/annotation_images/{original_path}
            rel_path = sample["image_path"]
            output_path = os.path.join(save_dir, "annotation_images", rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            edited_image.save(output_path)

            # Also save panel for visual inspection
            panel_path = output_path.replace(".jpg", "_panel.jpg").replace(".png", "_panel.png")
            panel_image.save(panel_path)

            results.append({
                "id": sample["id"],
                "image_path": image_path,
                "edited_path": output_path,
                "panel_path": panel_path,
                "original_prompt": original_prompt,
                "editing_prompt": editing_prompt
            })

        except Exception as e:
            print(f"[Error] Failed to edit {image_path}: {e}")
            continue

    print(f"✓ {exp_name}: Successfully edited {len(results)}/{len(samples)} images")
    print(f"✓ Results saved to: {save_dir}")

    return results


def run_evaluation(data_path, output_path, exp_name, metrics_csv):
    """
    Run evaluation using eval_p2p_metrics_official.py

    Args:
        data_path: PIE-Bench data directory
        output_path: Output directory containing edited images
        exp_name: Experiment name (baseline or enhanced)
        metrics_csv: Output CSV file path
    """
    print(f"\n{'='*80}")
    print(f"Evaluating {exp_name.upper()}")
    print(f"{'='*80}")

    cmd = f"""python eval_p2p_metrics_official.py \
        --data_path {data_path} \
        --output_path {os.path.join(output_path, exp_name)} \
        --edit_category_list 0 \
        --edit_method_list ddim+p2p \
        --metrics_csv {metrics_csv}"""

    print(f"Running command:\n{cmd}\n")

    import subprocess
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print(result.stdout)
        print(f"✓ Evaluation completed: {metrics_csv}")
    else:
        print(f"✗ Evaluation failed:")
        print(result.stderr)
        raise RuntimeError(f"Evaluation failed for {exp_name}")

    return metrics_csv


def parse_metrics_csv(csv_path):
    """Parse metrics CSV and compute statistics"""
    import csv

    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if len(rows) == 0:
        return None

    ssim_vals = [float(r["SSIM_bg"]) for r in rows]
    lpips_vals = [float(r["LPIPS_bg"]) for r in rows]
    clip_vals = [float(r["CLIP_whole"]) for r in rows]

    return {
        "num_samples": len(rows),
        "SSIM_bg": {
            "mean": np.mean(ssim_vals),
            "std": np.std(ssim_vals),
            "values": ssim_vals
        },
        "LPIPS_bg": {
            "mean": np.mean(lpips_vals),
            "std": np.std(lpips_vals),
            "values": lpips_vals
        },
        "CLIP_whole": {
            "mean": np.mean(clip_vals),
            "std": np.std(clip_vals),
            "values": clip_vals
        }
    }


def generate_comparison_report(baseline_stats, enhanced_stats, output_file):
    """Generate comparison report"""

    report = []
    report.append("=" * 80)
    report.append("COMPARISON REPORT: Baseline vs Enhanced Text Encoder")
    report.append("=" * 80)
    report.append("")

    # Configuration
    report.append("Configuration:")
    report.append(f"  Baseline: OpenAI CLIP (openai/clip-vit-large-patch14)")
    report.append(f"  Enhanced: LAION CLIP (laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K)")
    report.append(f"  Number of samples: {baseline_stats['num_samples']}")
    report.append("")

    # Results table
    report.append("Results:")
    report.append("-" * 80)
    report.append(f"{'Metric':<20} {'Baseline':<20} {'Enhanced':<20} {'Δ Change':<20}")
    report.append("-" * 80)

    metrics = ["SSIM_bg", "LPIPS_bg", "CLIP_whole"]
    metric_names = {
        "SSIM_bg": "SSIM (bg) ↑",
        "LPIPS_bg": "LPIPS (bg) ↓",
        "CLIP_whole": "CLIP Similarity ↑"
    }

    for metric in metrics:
        baseline_mean = baseline_stats[metric]["mean"]
        enhanced_mean = enhanced_stats[metric]["mean"]

        # Calculate change
        if metric == "LPIPS_bg":
            # Lower is better for LPIPS
            change = baseline_mean - enhanced_mean  # Positive = improvement
            change_pct = (change / baseline_mean) * 100 if baseline_mean != 0 else 0
            symbol = "↓" if change > 0 else "↑"
        else:
            # Higher is better for SSIM and CLIP
            change = enhanced_mean - baseline_mean  # Positive = improvement
            change_pct = (change / baseline_mean) * 100 if baseline_mean != 0 else 0
            symbol = "↑" if change > 0 else "↓"

        baseline_str = f"{baseline_mean:.4f}"
        enhanced_str = f"{enhanced_mean:.4f}"
        change_str = f"{symbol}{abs(change):.4f} ({abs(change_pct):.2f}%)"

        report.append(f"{metric_names[metric]:<20} {baseline_str:<20} {enhanced_str:<20} {change_str:<20}")

    report.append("-" * 80)
    report.append("")

    # Summary
    report.append("Summary:")

    # SSIM
    ssim_change = enhanced_stats["SSIM_bg"]["mean"] - baseline_stats["SSIM_bg"]["mean"]
    ssim_pct = (ssim_change / baseline_stats["SSIM_bg"]["mean"]) * 100
    report.append(f"  SSIM (Background Preservation):")
    if ssim_change > 0:
        report.append(f"    ✓ Enhanced is BETTER by {ssim_change:.4f} (+{ssim_pct:.2f}%)")
    else:
        report.append(f"    ✗ Enhanced is WORSE by {abs(ssim_change):.4f} ({ssim_pct:.2f}%)")

    # LPIPS
    lpips_change = baseline_stats["LPIPS_bg"]["mean"] - enhanced_stats["LPIPS_bg"]["mean"]
    lpips_pct = (lpips_change / baseline_stats["LPIPS_bg"]["mean"]) * 100
    report.append(f"  LPIPS (Lower is Better):")
    if lpips_change > 0:
        report.append(f"    ✓ Enhanced is BETTER by {lpips_change:.4f} (-{lpips_pct:.2f}%)")
    else:
        report.append(f"    ✗ Enhanced is WORSE by {abs(lpips_change):.4f} (+{abs(lpips_pct):.2f}%)")

    # CLIP
    clip_change = enhanced_stats["CLIP_whole"]["mean"] - baseline_stats["CLIP_whole"]["mean"]
    clip_pct = (clip_change / baseline_stats["CLIP_whole"]["mean"]) * 100
    report.append(f"  CLIP Similarity (Text-Image Alignment):")
    if clip_change > 0:
        report.append(f"    ✓ Enhanced is BETTER by {clip_change:.4f} (+{clip_pct:.2f}%)")
    else:
        report.append(f"    ✗ Enhanced is WORSE by {abs(clip_change):.4f} ({clip_pct:.2f}%)")

    report.append("")
    report.append("=" * 80)

    # Write to file and print
    report_text = "\n".join(report)

    with open(output_file, 'w') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n✓ Report saved to: {output_file}")

    return report_text


def main():
    parser = argparse.ArgumentParser(description="Run comparison experiment")
    parser.add_argument('--data_path', type=str, default='data',
                       help='Path to PIE-Bench data directory')
    parser.add_argument('--output_dir', type=str, default='comparison_output',
                       help='Output directory for edited images')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to test (default: 10)')
    parser.add_argument('--category', type=str, default='0',
                       help='PIE-Bench category (default: 0 for random_140)')
    parser.add_argument('--enhanced_encoder', type=str, default='laion-L-datacomp',
                       choices=['laion-L-datacomp', 'laion-L-laion2b', 'openai-L-336'],
                       help='Enhanced encoder to use')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--skip_baseline', action='store_true',
                       help='Skip baseline experiment (only run enhanced)')
    parser.add_argument('--skip_enhanced', action='store_true',
                       help='Skip enhanced experiment (only run baseline)')
    parser.add_argument('--skip_editing', action='store_true',
                       help='Skip editing (only run evaluation on existing results)')

    args = parser.parse_args()

    # Check device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("[Warning] CUDA not available, using CPU")
        device = 'cpu'

    print("=" * 80)
    print("COMPARISON EXPERIMENT: Baseline vs Enhanced Text Encoder")
    print("=" * 80)
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Category: {args.category}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Enhanced encoder: {args.enhanced_encoder}")
    print(f"Device: {device}")
    print("=" * 80)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load samples
    if not args.skip_editing:
        print("\n📁 Loading PIE-Bench samples...")
        samples = load_pie_bench_samples(
            data_path=args.data_path,
            category=args.category,
            num_samples=args.num_samples
        )

        if len(samples) == 0:
            print("[Error] No samples loaded!")
            return

    # Run baseline experiment
    baseline_results = None
    if not args.skip_baseline:
        if not args.skip_editing:
            print("\n🔵 STEP 1: Running BASELINE experiment...")
            editor_baseline = P2PEditor(
                method_list=["ddim+p2p"],
                device=device,
                num_ddim_steps=50,
                enhanced_encoder=None  # Original encoder
            )

            baseline_results = run_editing_batch(
                editor=editor_baseline,
                samples=samples,
                data_path=args.data_path,
                output_dir=args.output_dir,
                edit_method="ddim+p2p",
                exp_name="baseline"
            )

        # Evaluate baseline
        print("\n📊 Evaluating BASELINE...")
        baseline_csv = os.path.join(args.output_dir, "metrics_baseline.csv")
        run_evaluation(
            data_path=args.data_path,
            output_path=args.output_dir,
            exp_name="baseline",
            metrics_csv=baseline_csv
        )

    # Run enhanced experiment
    enhanced_results = None
    if not args.skip_enhanced:
        if not args.skip_editing:
            print("\n🟢 STEP 2: Running ENHANCED experiment...")
            editor_enhanced = P2PEditor(
                method_list=["ddim+p2p"],
                device=device,
                num_ddim_steps=50,
                enhanced_encoder=args.enhanced_encoder  # Enhanced encoder
            )

            enhanced_results = run_editing_batch(
                editor=editor_enhanced,
                samples=samples,
                data_path=args.data_path,
                output_dir=args.output_dir,
                edit_method="ddim+p2p",
                exp_name="enhanced"
            )

        # Evaluate enhanced
        print("\n📊 Evaluating ENHANCED...")
        enhanced_csv = os.path.join(args.output_dir, "metrics_enhanced.csv")
        run_evaluation(
            data_path=args.data_path,
            output_path=args.output_dir,
            exp_name="enhanced",
            metrics_csv=enhanced_csv
        )

    # Generate comparison report
    if not args.skip_baseline and not args.skip_enhanced:
        print("\n📈 STEP 3: Generating comparison report...")

        baseline_csv = os.path.join(args.output_dir, "metrics_baseline.csv")
        enhanced_csv = os.path.join(args.output_dir, "metrics_enhanced.csv")

        baseline_stats = parse_metrics_csv(baseline_csv)
        enhanced_stats = parse_metrics_csv(enhanced_csv)

        if baseline_stats and enhanced_stats:
            report_file = os.path.join(args.output_dir, "comparison_report.txt")
            generate_comparison_report(baseline_stats, enhanced_stats, report_file)
        else:
            print("[Warning] Could not parse metrics CSV files")

    print("\n" + "=" * 80)
    print("✓✓✓ EXPERIMENT COMPLETED! ✓✓✓")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - baseline/: Baseline edited images")
    print(f"  - enhanced/: Enhanced edited images")
    print(f"  - metrics_baseline.csv: Baseline metrics")
    print(f"  - metrics_enhanced.csv: Enhanced metrics")
    print(f"  - comparison_report.txt: Comparison report")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""
Run P2P Editing with Frequency-Preserving Inversion

This script is similar to run_editing_p2p_metrics.py but uses the frequency-preserving
editor to demonstrate the effect of frequency domain regularization.

Key differences:
- Uses P2PEditorFrequency instead of P2PEditor
- Does NOT use adaptive num_inner_steps (fixed value)
- Focused on demonstrating frequency preservation effect
"""

import os
import numpy as np
import argparse
import json
from PIL import Image
import torch
import random

from models.p2p_editor_frequency import P2PEditorFrequency

# Metrics
from skimage.metrics import structural_similarity as ssim
import lpips
import clip
from torchvision import transforms
import csv


def mask_decode(encoded_mask, image_shape=[512, 512]):
    """Decode PIE-Bench mask format"""
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,))

    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i + 1], length - encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i] + j] = 1

    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    # Avoid annotation errors in boundary
    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1

    return mask_array


def setup_seed(seed=1234):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ⭐ 方法名 → 子目录名的映射（与 run_editing_p2p_metrics.py 对齐）
image_save_paths = {
    # Frequency-preserving methods with different configurations
    "frequency+p2p": "frequency+p2p",
    "frequency_w01+p2p": "frequency_w01+p2p",  # freq_weight=0.1
    "frequency_w03+p2p": "frequency_w03+p2p",  # freq_weight=0.3
    "frequency_w05+p2p": "frequency_w05+p2p",  # freq_weight=0.5
    "frequency_s5+p2p": "frequency_s5+p2p",    # num_inner_steps=5
    "frequency_s10+p2p": "frequency_s10+p2p",  # num_inner_steps=10
    "frequency_s15+p2p": "frequency_s15+p2p",  # num_inner_steps=15
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run P2P editing with frequency-preserving inversion"
    )
    parser.add_argument('--rerun_exist_images', action="store_true",
                        help="If set, re-generate images even if they already exist.")
    parser.add_argument('--data_path', type=str, default="data",
                        help="Root path of PIE-Bench")
    parser.add_argument('--output_path', type=str, default="output",
                        help="Root path to save outputs.")
    parser.add_argument('--edit_category_list', nargs='+', type=str,
                        default=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])

    # ⭐ 与原版对齐：使用 edit_method_list
    parser.add_argument('--edit_method_list', nargs='+', type=str,
                        default=["frequency+p2p"],
                        help="List of editing methods to run (e.g., frequency+p2p, frequency_w03+p2p)")

    # Frequency-specific parameters
    parser.add_argument('--num_inner_steps', type=int, default=10,
                        help="Number of optimization steps (FIXED - not adaptive)")
    parser.add_argument('--freq_weight', type=float, default=0.3,
                        help="Frequency preservation weight (0-1). "
                             "Higher = more structure preservation")

    # Metrics
    parser.add_argument('--compute_metrics', action="store_true",
                        help="If set, compute SSIM/LPIPS/CLIP and save to csv.")
    parser.add_argument('--metrics_csv', type=str,
                        default="metrics_frequency.csv",
                        help="Path to save metrics csv.")

    args = parser.parse_args()

    rerun_exist_images = args.rerun_exist_images
    data_path = args.data_path
    output_path = args.output_path
    edit_category_list = args.edit_category_list
    edit_method_list = args.edit_method_list

    print(f"\n{'='*80}")
    print("FREQUENCY-PRESERVING P2P EDITING")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Data path: {data_path}")
    print(f"  Output path: {output_path}")
    print(f"  Edit methods: {edit_method_list}")
    print(f"  Num inner steps: {args.num_inner_steps} (FIXED)")
    print(f"  Frequency weight: {args.freq_weight}")
    print(f"  Edit categories: {edit_category_list}")
    print(f"  Compute metrics: {args.compute_metrics}")
    print(f"{'='*80}\n")

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # ⭐ Initialize frequency-preserving editor
    # 注意：这里只需要初始化一次，不需要像 P2PEditor 那样传入 method_list
    freq_editor = P2PEditorFrequency(
        device=device,
        num_ddim_steps=50,
        freq_weight=args.freq_weight
    )

    # Initialize metric models
    if args.compute_metrics:
        print("Initializing LPIPS & CLIP models for evaluation...")
        lpips_fn = lpips.LPIPS(net='vgg').to(device)
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        to_tensor = transforms.ToTensor()
        metrics_records = []

    # Read mapping file
    with open(os.path.join(data_path, "mapping_file.json"), "r") as f:
        editing_instruction = json.load(f)

    for key, item in editing_instruction.items():

        if item["editing_type_id"] not in edit_category_list:
            continue

        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")

        # Source image path
        image_path = os.path.join(data_path, "annotation_images", item["image_path"])

        blended_word = item["blended_word"].split(" ") if item["blended_word"] != "" else []
        mask = Image.fromarray(
            np.uint8(mask_decode(item["mask"])[:, :, np.newaxis].repeat(3, 2))
        ).convert("L")

        # ⭐ 遍历每个方法（与原版对齐）
        for edit_method in edit_method_list:
            # 相对路径
            rel_path = os.path.relpath(image_path, data_path)

            # ① panel：4图拼接 (text | src | recon | edited)
            panel_save_path = os.path.join(
                output_path,
                image_save_paths[edit_method],
                rel_path
            )

            # ② 单独 edited 图：*_only 目录，用于算指标
            single_save_root = image_save_paths[edit_method] + "_only"
            single_save_path = os.path.join(
                output_path,
                single_save_root,
                rel_path
            )

            # 是否需要重新生成
            need_run = (not os.path.exists(single_save_path)) or rerun_exist_images

            if need_run:
                print(f"editing image [{image_path}] with [{edit_method}]")
                setup_seed()
                torch.cuda.empty_cache()

                # ⭐ Run frequency-preserving editing
                panel_image, edited_image = freq_editor.edit_image(
                    image_path=image_path,
                    prompt_src=original_prompt,
                    prompt_tar=editing_prompt,
                    num_inner_steps=args.num_inner_steps,
                    guidance_scale=7.5,
                    cross_replace_steps=0.4,
                    self_replace_steps=0.6,
                    blend_word=(((blended_word[0],),
                                 (blended_word[1],))) if len(blended_word) else None,
                    eq_params={
                        "words": (blended_word[1],),
                        "values": (2,)
                    } if len(blended_word) else None,
                )

                # 保存 panel
                os.makedirs(os.path.dirname(panel_save_path), exist_ok=True)
                panel_image.save(panel_save_path)

                # 保存单独 edited 图
                os.makedirs(os.path.dirname(single_save_path), exist_ok=True)
                edited_image.save(single_save_path)

                print("finish")
            else:
                print(f"skip image [{image_path}] with [{edit_method}] (single edited image already exists)")
                if args.compute_metrics:
                    edited_image = Image.open(single_save_path).convert("RGB")

            # 2) 计算指标：SSIM/LPIPS/CLIP（全部基于 single_save_path 里的 edited_image）
            if args.compute_metrics:
                if (not os.path.exists(image_path)) or (not os.path.exists(single_save_path)):
                    print(f"[Warning] missing source or edited image for {image_path}, skip metrics.")
                    continue

                # 读原图 & edited 单图
                src_img = Image.open(image_path).convert("RGB")
                edited_img = edited_image.convert("RGB")
                edited_img = edited_img.resize(src_img.size, Image.BILINEAR)

                src_np = np.array(src_img).astype(np.float32)
                edited_np = np.array(edited_img).astype(np.float32)

                # ---- mask: 编辑区域=1, 背景=0 ----
                mask_array = mask_decode(item["mask"])
                mask_3ch = mask_array[:, :, np.newaxis].repeat(3, axis=2)
                background_mask = 1.0 - mask_3ch  # 背景=1, 前景=0

                if background_mask.sum() > 0:
                    # === 背景 SSIM ===
                    src_bg = src_np * background_mask
                    edited_bg = edited_np * background_mask
                    ssim_val = ssim(
                        src_bg,
                        edited_bg,
                        channel_axis=-1,
                        data_range=255.0,
                    )

                    # === 背景 LPIPS ===
                    src_t = to_tensor(src_img).unsqueeze(0).to(device) * 2 - 1
                    edit_t = to_tensor(edited_img).unsqueeze(0).to(device) * 2 - 1

                    bg_mask_t = torch.from_numpy(background_mask).permute(2, 0, 1) \
                                    .unsqueeze(0).float().to(device)

                    src_t_bg = src_t * bg_mask_t
                    edit_t_bg = edit_t * bg_mask_t

                    lpips_val = lpips_fn(src_t_bg, edit_t_bg).item()
                else:
                    ssim_val = float('nan')
                    lpips_val = float('nan')

                # --- CLIP: target prompt (editing_prompt) vs edited image (single) ---
                with torch.no_grad():
                    clip_img = clip_preprocess(edited_image).unsqueeze(0).to(device)
                    img_feat = clip_model.encode_image(clip_img)
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

                    text_tokens = clip.tokenize([editing_prompt]).to(device)
                    text_feat = clip_model.encode_text(text_tokens)
                    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

                    clip_sim = (img_feat @ text_feat.T).item()

                metrics_records.append({
                    "id": key,
                    "edit_method": edit_method,
                    "editing_type_id": item["editing_type_id"],
                    "image_path": image_path,
                    "edited_image_path": single_save_path,
                    "original_prompt": original_prompt,
                    "editing_prompt": editing_prompt,
                    "SSIM_src_edited": ssim_val,
                    "LPIPS_src_edited": lpips_val,
                    "CLIP_tgtPrompt_editedImage": clip_sim
                })

    # 3) 写出 csv + 打印整体均值
    if args.compute_metrics and len(metrics_records) > 0:
        csv_path = args.metrics_csv
        fieldnames = [
            "id",
            "edit_method",
            "editing_type_id",
            "image_path",
            "edited_image_path",
            "original_prompt",
            "editing_prompt",
            "SSIM_src_edited",
            "LPIPS_src_edited",
            "CLIP_tgtPrompt_editedImage"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            for row in metrics_records:
                writer.writerow(row)

        all_ssim = [r["SSIM_src_edited"] for r in metrics_records]
        all_lpips = [r["LPIPS_src_edited"] for r in metrics_records]
        all_clip = [r["CLIP_tgtPrompt_editedImage"] for r in metrics_records]

        print(f"\n[Done] metrics saved to {csv_path}")
        print(f"[Summary over {len(metrics_records)} samples]")
        print(f"  Mean SSIM(source, edited)      : {np.mean(all_ssim):.4f}")
        print(f"  Mean LPIPS(source, edited)     : {np.mean(all_lpips):.4f}")
        print(f"  Mean CLIP(prompt, edited_image): {np.mean(all_clip):.4f}")

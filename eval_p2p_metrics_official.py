import os
import json
import argparse
import numpy as np
from PIL import Image

import torch

# ====== 尝试导入官方 MetricsCalculator ======
try:
    from evaluation.matrics_calculator import MetricsCalculator  # 你 repo 里的名字大概率是这个
except ImportError:
    from evaluation.metrics_calculator import MetricsCalculator  # 以防万一是这个名字
# ==============================================


def mask_decode(encoded_mask, image_shape=(512, 512)):
    """
    从 mapping_file.json 里的 RLE mask 解码成 [H, W] 的 0/1 numpy 数组
    （和你之前 run_editing_p2p_metrics.py 里的版本一致）。
    """
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,), dtype=np.uint8)

    for i in range(0, len(encoded_mask), 2):
        start = encoded_mask[i]
        run_len = encoded_mask[i + 1]
        end = min(start + run_len, length)
        mask_array[start:end] = 1

    mask_array = mask_array.reshape(image_shape[0], image_shape[1])

    # 边界置 1（PIE-Bench 的做法，用来避免标注边缘误差）
    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1

    return mask_array


# 和之前一样的 “方法名 → 子目录名” 映射
IMAGE_SAVE_PATHS = {
    "ddim+p2p": "ddim+p2p_only",
    "null-text-inversion+p2p": "null-text-inversion+p2p",
    "null-text-inversion+p2p_a800": "null-text-inversion+p2p_a800",
    "null-text-inversion+p2p_3090": "null-text-inversion+p2p_3090",
    "negative-prompt-inversion+p2p": "negative-prompt-inversion+p2p",
    "directinversion+p2p": "directinversion+p2p",
    "directinversion+p2p_guidance_0_1": "directinversion+p2p_guidance_0_1",
    "directinversion+p2p_guidance_0_5": "directinversion+p2p_guidance_0_5",
    "directinversion+p2p_guidance_0_25": "directinversion+p2p_guidance_0_25",
    "directinversion+p2p_guidance_0_75": "directinversion+p2p_guidance_0_75",
    "directinversion+p2p_guidance_1_1": "directinversion+p2p_guidance_1_1",
    "directinversion+p2p_guidance_1_5": "directinversion+p2p_guidance_1_5",
    "directinversion+p2p_guidance_1_25": "directinversion+p2p_guidance_1_25",
    "directinversion+p2p_guidance_1_75": "directinversion+p2p_guidance_1_75",
    "directinversion+p2p_guidance_25_1": "directinversion+p2p_guidance_25_1",
    "directinversion+p2p_guidance_25_5": "directinversion+p2p_guidance_25_5",
    "directinversion+p2p_guidance_25_25": "directinversion+p2p_guidance_25_25",
    "directinversion+p2p_guidance_25_75": "directinversion+p2p_guidance_25_75",
    "directinversion+p2p_guidance_5_1": "directinversion+p2p_guidance_5_1",
    "directinversion+p2p_guidance_5_5": "directinversion+p2p_guidance_5_5",
    "directinversion+p2p_guidance_5_25": "directinversion+p2p_guidance_5_25",
    "directinversion+p2p_guidance_5_75": "directinversion+p2p_guidance_5_75",
    "directinversion+p2p_guidance_75_1": "directinversion+p2p_guidance_75_1",
    "directinversion+p2p_guidance_75_5": "directinversion+p2p_guidance_75_5",
    "directinversion+p2p_guidance_75_25": "directinversion+p2p_guidance_75_25",
    "directinversion+p2p_guidance_75_75": "directinversion+p2p_guidance_75_75",
    "null-text-inversion+proximal-guidance": "null-text-inversion+proximal-guidance",
    "negative-prompt-inversion+proximal-guidance": "negative-prompt-inversion+proximal-guidance",
    "ablation_null-latent-inversion+p2p": "ablation_null-latent-inversion+p2p",
    "ablation_directinversion_08+p2p": "ablation_directinversion_08+p2p",
    "ablation_directinversion_04+p2p": "ablation_directinversion_04+p2p",
    "ablation_directinversion_interval_2+p2p": "ablation_directinversion_interval_2+p2p",
    "ablation_directinversion_interval_5+p2p": "ablation_directinversion_interval_5+p2p",
    "ablation_directinversion_interval_10+p2p": "ablation_directinversion_interval_10+p2p",
    "ablation_directinversion_interval_24+p2p": "ablation_directinversion_interval_24+p2p",
    "ablation_directinversion_interval_49+p2p": "ablation_directinversion_interval_49+p2p",
    "ablation_null-text-inversion_single_branch+p2p": "ablation_null-text-inversion_single_branch+p2p",
    "ablation_directinversion_add-source+p2p": "ablation_directinversion_add-source+p2p",
    "ablation_directinversion_add-target+p2p": "ablation_directinversion_add-target+p2p",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="data",
                        help="PIE-Bench 根目录（包含 mapping_file.json 和 annotation_images/）")
    parser.add_argument('--output_path', type=str, default="output",
                        help="之前保存 edited_single 图像的根目录")
    parser.add_argument('--edit_category_list', nargs="+", type=str,
                        default=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    parser.add_argument('--edit_method_list', nargs="+", type=str,
                        default=["ddim+p2p"])
    parser.add_argument('--metrics_csv', type=str,
                        default="metrics_official_ddim_p2p.csv",
                        help="输出 csv 路径")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_calc = MetricsCalculator(device=device)

    # 读 mapping_file.json
    with open(os.path.join(args.data_path, "mapping_file.json"), "r") as f:
        editing_instruction = json.load(f)

    import csv
    fieldnames = [
        "id",
        "edit_method",
        "editing_type_id",
        "image_path",
        "edited_image_path",
        "original_prompt",
        "editing_prompt",
        # 背景区域指标（论文 Background Preservation）
        "SSIM_bg",
        "LPIPS_bg",
        # CLIP Whole Image
        "CLIP_whole",
    ]
    rows = []

    for key, item in editing_instruction.items():
        if item["editing_type_id"] not in args.edit_category_list:
            continue

        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")

        # 源图像路径
        src_path = os.path.join(args.data_path, "annotation_images", item["image_path"])
        if not os.path.exists(src_path):
            print(f"[Warning] source image not found: {src_path}")
            continue

        # mask（编辑区域 = 1）
        mask = mask_decode(item["mask"])  # [H, W], 0/1
        # 背景区域 = 1 - mask
        bg_mask_np = 1.0 - mask.astype(np.float32)  # [H, W]

        # 加载源图
        src_img = Image.open(src_path).convert("RGB")
        src_w, src_h = src_img.size

        for edit_method in args.edit_method_list:
            if edit_method not in IMAGE_SAVE_PATHS:
                print(f"[Warning] unknown edit_method {edit_method}, skip.")
                continue

            rel_path = os.path.relpath(src_path, args.data_path)  # annotation_images/...
            single_root = IMAGE_SAVE_PATHS[edit_method] + "_single"
            edited_path = os.path.join(args.output_path, single_root, rel_path)

            if not os.path.exists(edited_path):
                print(f"[Warning] edited image not found: {edited_path}")
                continue

            edited_img = Image.open(edited_path).convert("RGB")
            # 尺寸不一致的话，统一 resize 到源图大小（一般本来就是 512x512）
            if edited_img.size != src_img.size:
                edited_img = edited_img.resize(src_img.size, Image.BILINEAR)

            # ======= 1. 背景区域 SSIM / LPIPS（用官方 TorchMetrics 实现） =======
            # 转成 [1, 3, H, W]、范围 [0,1]
            src_np = np.array(src_img).astype(np.float32) / 255.0  # [H,W,3]
            edited_np = np.array(edited_img).astype(np.float32) / 255.0

            # 背景 mask [H,W] -> [1,1,H,W] -> broadcast 到 [1,3,H,W]
            if bg_mask_np.shape != src_np.shape[:2]:
                # 如果 mask 不是 512x512，简单 resize 一下
                bg_mask_img = Image.fromarray((bg_mask_np * 255).astype(np.uint8)).resize(
                    src_img.size, Image.NEAREST
                )
                bg_mask_np = np.array(bg_mask_img).astype(np.float32) / 255.0

            bg_mask_t = torch.from_numpy(bg_mask_np)[None, None, ...].to(device)  # [1,1,H,W]

            src_t = torch.from_numpy(src_np).permute(2, 0, 1)[None, ...].to(device)  # [1,3,H,W]
            edited_t = torch.from_numpy(edited_np).permute(2, 0, 1)[None, ...].to(device)

            bg_mask_t_3 = bg_mask_t.expand(-1, 3, -1, -1)  # [1,3,H,W]

            src_bg = src_t * bg_mask_t_3
            edited_bg = edited_t * bg_mask_t_3

            # 官方的 SSIM / LPIPS：data_range=1.0
            with torch.no_grad():
                ssim_bg = metrics_calc.ssim_metric_calculator(edited_bg, src_bg).item()
                lpips_bg = metrics_calc.lpips_metric_calculator(edited_bg, src_bg).item()

            # ======= 2. CLIP Whole Image Similarity（官方 CLIPScore） =======
            # 直接用他们的 calculate_clip_similarity（不加 mask）
            clip_whole = metrics_calc.calculate_clip_similarity(edited_img, editing_prompt)

            rows.append({
                "id": key,
                "edit_method": edit_method,
                "editing_type_id": item["editing_type_id"],
                "image_path": src_path,
                "edited_image_path": edited_path,
                "original_prompt": original_prompt,
                "editing_prompt": editing_prompt,
                "SSIM_bg": ssim_bg,
                "LPIPS_bg": lpips_bg,
                "CLIP_whole": clip_whole,
            })

    # ===== 写 CSV + 打印整体均值 =====
    if len(rows) == 0:
        print("[Warning] no metrics computed.")
        return

    os.makedirs(os.path.dirname(args.metrics_csv) or ".", exist_ok=True)
    with open(args.metrics_csv, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # 汇总
    ssim_vals = [r["SSIM_bg"] for r in rows]
    lpips_vals = [r["LPIPS_bg"] for r in rows]
    clip_vals = [r["CLIP_whole"] for r in rows]

    print(f"\n[Done] metrics saved to {args.metrics_csv}")
    print(f"[Summary over {len(rows)} samples]")
    print(f"  Mean SSIM (background)   : {np.mean(ssim_vals):.4f}")
    print(f"  Mean LPIPS (background)  : {np.mean(lpips_vals):.4f}")
    print(f"  Mean CLIP (whole image)  : {np.mean(clip_vals):.4f}")


if __name__ == "__main__":
    main()

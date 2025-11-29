import os
import numpy as np
import argparse
import json
from PIL import Image
import torch
import random

from models.p2p_editor import P2PEditor

# === 评估相关包 ===
from skimage.metrics import structural_similarity as ssim
import lpips
import clip
from torchvision import transforms
import csv
# =======================


def mask_decode(encoded_mask, image_shape=[512, 512]):
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,))

    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i + 1], length - encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i] + j] = 1

    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1

    return mask_array


def setup_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


image_save_paths = {
    "ddim+p2p": "ddim+p2p",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun_exist_images', action="store_true")  # rerun existing images
    parser.add_argument('--data_path', type=str, default="data")      # PIE-Bench 根目录
    parser.add_argument('--output_path', type=str, default="output")
    parser.add_argument('--edit_category_list', nargs='+', type=str,
                        default=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
    parser.add_argument('--edit_method_list', nargs='+', type=str,
                        default=["ddim+p2p"])  # 只跑 ddim+p2p

    # 是否计算指标 & 输出文件名
    parser.add_argument('--compute_metrics', action="store_true",
                        help="If set, compute SSIM/LPIPS/CLIP and save to csv.")
    parser.add_argument('--metrics_csv', type=str,
                        default="metrics_ddim_p2p.csv",
                        help="Path to save metrics csv.")

    args = parser.parse_args()

    rerun_exist_images = args.rerun_exist_images
    data_path = args.data_path
    output_path = args.output_path
    edit_category_list = args.edit_category_list
    edit_method_list = args.edit_method_list

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    p2p_editor = P2PEditor(edit_method_list, device, num_ddim_steps=50)

    # 初始化 metric 模型
    if args.compute_metrics:
        print("Initializing LPIPS & CLIP models for evaluation...")
        lpips_fn = lpips.LPIPS(net='vgg').to(device)
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        to_tensor = transforms.ToTensor()
        metrics_records = []

    with open(f"{data_path}/mapping_file.json", "r") as f:
        editing_instruction = json.load(f)

    for key, item in editing_instruction.items():

        if item["editing_type_id"] not in edit_category_list:
            continue

        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")

        # 源图像路径
        image_path = os.path.join(f"{data_path}/annotation_images", item["image_path"])

        editing_instruction_text = item["editing_instruction"]
        blended_word = item["blended_word"].split(" ") if item["blended_word"] != "" else []
        mask = Image.fromarray(
            np.uint8(mask_decode(item["mask"])[:, :, np.newaxis].repeat(3, 2))
        ).convert("L")

        for edit_method in edit_method_list:
            present_image_save_path = image_path.replace(
                data_path,
                os.path.join(output_path, image_save_paths[edit_method])
            )

            # 1) 生成 / 读取 edited image
            if ((not os.path.exists(present_image_save_path)) or rerun_exist_images):
                print(f"editing image [{image_path}] with [{edit_method}]")
                setup_seed()
                torch.cuda.empty_cache()

                edited_image = p2p_editor(
                    edit_method,
                    image_path=image_path,
                    prompt_src=original_prompt,
                    prompt_tar=editing_prompt,
                    guidance_scale=7.5,
                    cross_replace_steps=0.4,
                    self_replace_steps=0.6,
                    blend_word=(((blended_word[0],),
                                 (blended_word[1],))) if len(blended_word) else None,
                    eq_params={
                        "words": (blended_word[1],),
                        "values": (2,)
                    } if len(blended_word) else None,
                    proximal="l0",
                    quantile=0.75,
                    use_inversion_guidance=True,
                    recon_lr=1,
                    recon_t=400,
                )

                if not os.path.exists(os.path.dirname(present_image_save_path)):
                    os.makedirs(os.path.dirname(present_image_save_path))
                edited_image.save(present_image_save_path)

                print("finish")
            else:
                print(f"skip image [{image_path}] with [{edit_method}]")
                if args.compute_metrics:
                    edited_image = Image.open(present_image_save_path).convert("RGB")

            # 2) 计算指标：SSIM(src, edited), LPIPS(src, edited), CLIP(target prompt vs edited image)
            if args.compute_metrics:
                if not os.path.exists(image_path):
                    print(f"[Warning] source image not found: {image_path}, skip metrics.")
                    continue

                src_img = Image.open(image_path).convert("RGB").resize(
                    edited_image.size, Image.BILINEAR
                )

                # --- SSIM ---
                src_np = np.array(src_img)
                edited_np = np.array(edited_image)
                ssim_val = ssim(
                    src_np,
                    edited_np,
                    channel_axis=-1,
                    data_range=255
                )

                # --- LPIPS ---
                src_t = to_tensor(src_img).unsqueeze(0).to(device) * 2 - 1
                edit_t = to_tensor(edited_image).unsqueeze(0).to(device) * 2 - 1
                lpips_val = lpips_fn(src_t, edit_t).item()

                # --- CLIP: target prompt (editing_prompt) vs edited image ---
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
                    "edited_image_path": present_image_save_path,
                    "original_prompt": original_prompt,
                    "editing_prompt": editing_prompt,
                    "SSIM_src_edited": ssim_val,
                    "LPIPS_src_edited": lpips_val,
                    "CLIP_tgtPrompt_editedImage": clip_sim
                })

    # 3) 写出 csv
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
        print(f"\n[Done] metrics saved to {csv_path}")

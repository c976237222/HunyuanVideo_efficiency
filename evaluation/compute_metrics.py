#!/usr/bin/env python3
# compute_metrics.py

import argparse
import os
import numpy as np
import math
from glob import glob
from skimage.metrics import structural_similarity as compare_ssim
import imageio
import lpips
import torch
from tqdm import tqdm
import logging
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Compute metrics between two sets of videos.")
    parser.add_argument("--root1", type=str, required=True,
                        help="Directory of reference/original .mp4 videos.")
    parser.add_argument("--root2", type=str, required=True,
                        help="Directory of reconstructed/generated .mp4 videos.")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory to store the metric results.")
    return parser.parse_args()


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def compute_psnr(img1, img2):
    mse = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def compute_ssim(img1, img2):
    if np.all(img1 == img1[0, 0, 0]) or np.all(img2 == img2[0, 0, 0]):
        return 1.0
    return compare_ssim(img1, img2, data_range=img1.max() - img1.min(), channel_axis=-1)

def compute_lpips(img1, img2, loss_fn):
    img1_tensor = (
        torch.from_numpy(img1 / 255.0)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to("cuda:0")
    )
    img2_tensor = (
        torch.from_numpy(img2 / 255.0)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to("cuda:0")
    )

    img1_tensor = img1_tensor * 2 - 1
    img2_tensor = img2_tensor * 2 - 1

    return loss_fn(img1_tensor, img2_tensor).item()

def read_video(file_path):
    try:
        video = imageio.get_reader(file_path)
        frames = [frame for frame in video]
        video.close()
        return frames
    except Exception as e:
        logging.error(f"读取视频失败 {file_path}: {e}")
        return []

def save_results(results, root1, root2, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(results_dir, f"metrics_{timestamp}.txt")
    with open(output_file, "w") as f:
        f.write("\n")
        f.write(f"Root1: {root1}\n")
        f.write(f"Root2: {root2}\n")
        f.write(f"Timestamp: {timestamp}\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value}\n")
        f.write("\n")
    logging.info(f"结果已保存到 {output_file}")

def main():
    args = parse_args()

    root1 = args.root1
    root2 = args.root2
    results_dir = args.results_dir

    videos1 = sorted(glob(os.path.join(root1, "*.mp4")))
    videos2 = sorted(glob(os.path.join(root2, "*.mp4")))

    filenames1 = {os.path.basename(video) for video in videos1}
    filenames2 = {os.path.basename(video) for video in videos2}

    common_filenames = sorted(filenames1 & filenames2)

    if not common_filenames:
        logging.error("未找到匹配的视频文件。")
        return

    logging.info(f"找到 {len(common_filenames)} 对匹配视频。")

    matched_videos1 = [os.path.join(root1, filename) for filename in common_filenames]
    matched_videos2 = [os.path.join(root2, filename) for filename in common_filenames]

    metric_psnr = []
    metric_ssim = []
    metric_lpips_ = []

    lpips_model = lpips.LPIPS(net="alex").to("cuda:0")
    logging.info("已加载 LPIPS 模型 (AlexNet)。")

    for vid1_path, vid2_path in tqdm(zip(matched_videos1, matched_videos2),
                                     total=len(common_filenames),
                                     desc="处理视频中"):
        vid1_frames = read_video(vid1_path)
        vid2_frames = read_video(vid2_path)

        if not vid1_frames or not vid2_frames:
            logging.error(f"跳过无法读取的视频对: {vid1_path}, {vid2_path}")
            continue

        for f1, f2 in zip(vid1_frames, vid2_frames):
            try:
                psnr_value = compute_psnr(f1, f2)
                metric_psnr.append(psnr_value)
            except Exception as e:
                logging.error(f"计算 PSNR 失败: {e}")

            try:
                ssim_value = compute_ssim(f1, f2)
                metric_ssim.append(ssim_value)
            except Exception as e:
                logging.error(f"计算 SSIM 失败: {e}")

            try:
                lpips_value = compute_lpips(f1, f2, lpips_model)
                metric_lpips_.append(lpips_value)
            except Exception as e:
                logging.error(f"计算 LPIPS 失败: {e}")

    results = {}
    if metric_psnr:
        results["PSNR"] = sum(metric_psnr) / len(metric_psnr)
    if metric_ssim:
        results["SSIM"] = sum(metric_ssim) / len(metric_ssim)
    if metric_lpips_:
        results["LPIPS"] = sum(metric_lpips_) / len(metric_lpips_)

    logging.info(f"计算结果: {results}")
    save_results(results, root1, root2, results_dir)


if __name__ == "__main__":
    main()

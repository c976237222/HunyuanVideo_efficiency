#!/usr/bin/env python3
# compute_metrics_opencv_cpu.py

import argparse
import os
import numpy as np
import math
from glob import glob
import lpips
import torch
from tqdm import tqdm
import logging
from datetime import datetime
import concurrent.futures
import re
import cv2  # 导入OpenCV库
from skimage.metrics import structural_similarity as compare_ssim


def parse_args():
    parser = argparse.ArgumentParser(description="Compute metrics between two sets of videos (OpenCV CPU version).")
    parser.add_argument("--root1", type=str, required=True,
                        help="Directory of reference/original .mp4 videos.")
    parser.add_argument("--root2", type=str, required=True,
                        help="Directory that contains multiple exp_i directories, each with .mp4 videos.")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory to store the metric results.")
    parser.add_argument("--num-threads", type=int, default=4,
                        help="Number of parallel tasks.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for LPIPS calculation.")
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


def read_video_opencv(file_path):
    """
    使用OpenCV进行CPU视频解码
    返回帧列表: list of ndarray(H,W,3), dtype=uint8 (RGB格式)
    """
    frames = []
    try:
        # 使用OpenCV打开视频文件
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            logging.error(f"无法打开视频文件: {file_path}")
            return []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 将BGR转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()
    except Exception as e:
        logging.error(f"OpenCV解码失败 {file_path}: {e}")
    return frames


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


def compute_lpips_multi_video_batch(all_pairs, model, device, batch_size):
    lpips_values = []
    idx = 0
    n = len(all_pairs)

    while idx < n:
        end = min(idx + batch_size, n)
        batch_1 = []
        batch_2 = []
        for i in range(idx, end):
            f1, f2 = all_pairs[i]
            t1 = torch.from_numpy(f1).float().permute(2, 0, 1) / 255.0
            t2 = torch.from_numpy(f2).float().permute(2, 0, 1) / 255.0
            batch_1.append(t1)
            batch_2.append(t2)

        batch_1 = torch.stack(batch_1, dim=0).to(device)
        batch_2 = torch.stack(batch_2, dim=0).to(device)
        batch_1 = batch_1 * 2 - 1
        batch_2 = batch_2 * 2 - 1

        with torch.no_grad():
            dists = model(batch_1, batch_2)

        dists = dists.view(-1).cpu().numpy().tolist()
        lpips_values.extend(dists)
        idx = end

    return lpips_values


def compute_metrics_for_one_folder(root1, folder2, results_base_dir, lpips_model, device, batch_size):
    exp_name = os.path.basename(folder2.rstrip("/"))
    results_dir = os.path.join(results_base_dir, exp_name)
    os.makedirs(results_dir, exist_ok=True)

    videos1 = sorted(glob(os.path.join(root1, "*.mp4")))
    videos2 = sorted(glob(os.path.join(folder2, "*.mp4")))
    filenames1 = {os.path.basename(video) for video in videos1}
    filenames2 = {os.path.basename(video) for video in videos2}
    common_filenames = sorted(filenames1 & filenames2)

    if not common_filenames:
        logging.error(f"[{exp_name}] 未找到与 {root1} 匹配的视频文件，跳过。")
        return

    logging.info(f"[{exp_name}] (Device={device}) 找到 {len(common_filenames)} 对匹配视频。")

    metric_psnr = []
    metric_ssim = []
    metric_lpips_ = []
    all_pairs_for_lpips = []

    for filename in tqdm(common_filenames, desc=f"[{exp_name}] 处理视频 (Device={device})"):
        vid1_path = os.path.join(root1, filename)
        vid2_path = os.path.join(folder2, filename)

        # 使用OpenCV进行CPU解码
        vid1_frames = read_video_opencv(vid1_path)
        vid2_frames = read_video_opencv(vid2_path)

        len1 = len(vid1_frames)
        len2 = len(vid2_frames)
        if len1 == 0 or len2 == 0:
            logging.error(f"[{exp_name}] 视频帧为空或读取失败: {vid1_path}, {vid2_path}，跳过。")
            continue

        min_len = min(len1, len2)
        vid1_frames = vid1_frames[:min_len]
        vid2_frames = vid2_frames[:min_len]

        for f1, f2 in zip(vid1_frames, vid2_frames):
            try:
                metric_psnr.append(compute_psnr(f1, f2))
            except Exception as e:
                logging.error(f"[{exp_name}] 计算 PSNR 失败: {e}")

            try:
                metric_ssim.append(compute_ssim(f1, f2))
            except Exception as e:
                logging.error(f"[{exp_name}] 计算 SSIM 失败: {e}")

        for f1, f2 in zip(vid1_frames, vid2_frames):
            all_pairs_for_lpips.append((f1, f2))

    if all_pairs_for_lpips:
        try:
            lpips_vals = compute_lpips_multi_video_batch(
                all_pairs_for_lpips, lpips_model, device, batch_size
            )
            metric_lpips_.extend(lpips_vals)
        except Exception as e:
            logging.error(f"[{exp_name}] 计算 LPIPS 失败: {e}")

    results = {}
    if metric_psnr:
        results["PSNR"] = sum(metric_psnr) / len(metric_psnr)
    if metric_ssim:
        results["SSIM"] = sum(metric_ssim) / len(metric_ssim)
    if metric_lpips_:
        results["LPIPS"] = sum(metric_lpips_) / len(metric_lpips_)

    logging.info(f"[{exp_name}] (Device={device}) 计算结果: {results}")
    save_results(results, root1, folder2, results_dir)


def main():
    args = parse_args()
    root1 = args.root1
    root2 = args.root2
    results_dir = args.results_dir
    num_threads = args.num_threads
    batch_size = args.batch_size

    subdirs = []
    for entry in os.scandir(root2):
        if entry.is_dir() and re.match(r"^exp_\w+", entry.name):
            subdirs.append(entry.path)

    if not subdirs:
        logging.error(f"在 {root2} 下未找到任何 exp_{{i}} 子目录，退出。")
        return

    logging.info(f"在 {root2} 下找到 {len(subdirs)} 个 exp_* 子目录。")

    # 自动检测可用设备
    devices = ["cuda:0","cuda:1","cuda:2","cuda:3"] if torch.cuda.is_available() else ["cpu"]
    num_gpus = len(devices)

    logging.info(f"准备加载 {num_gpus} 个 LPIPS 模型 (AlexNet)，分别放到 {devices} ...")
    lpips_models = []
    for dev in devices:
        logging.info(f"加载 LPIPS 到 {dev}...")
        model = lpips.LPIPS(net="alex").to(dev)
        lpips_models.append(model)
    logging.info("全部 LPIPS 模型加载完毕。")

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_subdir = {}
        for i, folder2 in enumerate(subdirs):
            gpu_index = i % num_gpus
            device = devices[gpu_index]
            lpips_model = lpips_models[gpu_index]

            future = executor.submit(
                compute_metrics_for_one_folder,
                root1,
                folder2,
                results_dir,
                lpips_model,
                device,
                batch_size
            )
            future_to_subdir[future] = folder2

        for future in concurrent.futures.as_completed(future_to_subdir):
            subdir = future_to_subdir[future]
            try:
                future.result()
            except Exception as e:
                logging.error(f"目录 {subdir} 处理时出现异常: {e}")

    logging.info("所有子目录处理完成。")


if __name__ == "__main__":
    main()
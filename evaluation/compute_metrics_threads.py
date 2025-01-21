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
import concurrent.futures
import re


def parse_args():
    parser = argparse.ArgumentParser(description="Compute metrics between two sets of videos.")
    parser.add_argument("--root1", type=str, required=True,
                        help="Directory of reference/original .mp4 videos.")
    parser.add_argument("--root2", type=str, required=True,
                        help="Directory that contains multiple exp_i directories, each with .mp4 videos.")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory to store the metric results.")
    parser.add_argument("--num-threads", type=int, default=4,
                        help="Number of parallel tasks (usually <= #GPUs).")
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
    # 如果某帧是纯色（全0或全255），为了避免异常情况，给一个默认值
    if np.all(img1 == img1[0, 0, 0]) or np.all(img2 == img2[0, 0, 0]):
        return 1.0
    return compare_ssim(img1, img2, data_range=img1.max() - img1.min(), channel_axis=-1)


def read_video(file_path):
    """
    读视频并返回帧列表: list of ndarray(H,W,3), dtype=uint8
    """
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


def compute_lpips_multi_video_batch(all_pairs, model, device, batch_size):
    """
    将多个视频的帧（(f1, f2)）合并成一个列表 all_pairs，
    然后一次性分批 (batch_size) 送入 LPIPS 模型，返回对应的 LPIPS 值列表，与 all_pairs 长度一致。

    all_pairs: list of (frame1, frame2)，其中 frame1, frame2 是 ndarray(H,W,3) 
    model: 在 device 上的 LPIPS 模型
    device: "cuda:0" 等
    batch_size: 每次 forward 的最大帧数
    """
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
        # LPIPS需要 [-1,1]
        batch_1 = batch_1 * 2 - 1
        batch_2 = batch_2 * 2 - 1

        with torch.no_grad():
            dists = model(batch_1, batch_2)  # shape: [batch] or [batch,1,1,1]
        # 压平后转到CPU
        dists = dists.view(-1).cpu().numpy().tolist()
        lpips_values.extend(dists)

        idx = end

    return lpips_values


def compute_metrics_for_one_folder(root1, folder2, results_base_dir, lpips_model, device, batch_size):
    """
    以前：对每个视频单独批量LPIPS。 
    现在：把该子目录下所有匹配视频的所有帧 (f1, f2) 收集到 all_pairs 里，
         最后一次性进行LPIPS大批量推理。

    PSNR、SSIM 仍然逐帧计算，并分别保存结果到 metric_psnr, metric_ssim。
    LPIPS 计算则把所有帧合并后一次批量前向，以减少kernel启动开销。
    """
    exp_name = os.path.basename(folder2.rstrip("/"))
    results_dir = os.path.join(results_base_dir, exp_name)
    os.makedirs(results_dir, exist_ok=True)

    # 找到同名 mp4
    videos1 = sorted(glob(os.path.join(root1, "*.mp4")))
    videos2 = sorted(glob(os.path.join(folder2, "*.mp4")))
    filenames1 = {os.path.basename(video) for video in videos1}
    filenames2 = {os.path.basename(video) for video in videos2}
    common_filenames = sorted(filenames1 & filenames2)

    if not common_filenames:
        logging.error(f"[{exp_name}] 未找到与 {root1} 匹配的视频文件，跳过。")
        return

    logging.info(f"[{exp_name}] (GPU={device}) 找到 {len(common_filenames)} 对匹配视频。")

    metric_psnr = []
    metric_ssim = []
    metric_lpips_ = []

    # 新增：我们用一个大列表，收集所有帧对 (f1, f2) 用于后面统一LPIPS推理
    all_pairs_for_lpips = []

    # 1) 逐个视频读取帧，计算PSNR、SSIM，并把帧放到 all_pairs_for_lpips
    for filename in tqdm(common_filenames, desc=f"[{exp_name}] 处理视频 (GPU={device})"):
        vid1_path = os.path.join(root1, filename)
        vid2_path = os.path.join(folder2, filename)

        vid1_frames = read_video(vid1_path)
        vid2_frames = read_video(vid2_path)

        len1 = len(vid1_frames)
        len2 = len(vid2_frames)
        if len1 == 0 or len2 == 0:
            logging.error(f"[{exp_name}] 视频帧为空或读取失败: {vid1_path}, {vid2_path}，跳过。")
            continue

        # 取最小长度
        min_len = min(len1, len2)
        vid1_frames = vid1_frames[:min_len]
        vid2_frames = vid2_frames[:min_len]

        # PSNR / SSIM 逐帧计算
        for f1, f2 in zip(vid1_frames, vid2_frames):
            try:
                metric_psnr.append(compute_psnr(f1, f2))
            except Exception as e:
                logging.error(f"[{exp_name}] 计算 PSNR 失败: {e}")

            try:
                metric_ssim.append(compute_ssim(f1, f2))
            except Exception as e:
                logging.error(f"[{exp_name}] 计算 SSIM 失败: {e}")

        # 收集帧对 (f1, f2) 用于后面的 LPIPS 大批量
        for f1, f2 in zip(vid1_frames, vid2_frames):
            all_pairs_for_lpips.append((f1, f2))

    # 2) 现在统一对 all_pairs_for_lpips 做批量 LPIPS
    #    这样可以把子目录所有视频帧合并，批量越大，越能减少GPU空转。
    if all_pairs_for_lpips:
        try:
            lpips_vals = compute_lpips_multi_video_batch(
                all_pairs_for_lpips, lpips_model, device, batch_size
            )
            metric_lpips_.extend(lpips_vals)
        except Exception as e:
            logging.error(f"[{exp_name}] 计算 LPIPS 失败: {e}")

    # 3) 汇总指标
    results = {}
    if metric_psnr:
        results["PSNR"] = sum(metric_psnr) / len(metric_psnr)
    if metric_ssim:
        results["SSIM"] = sum(metric_ssim) / len(metric_ssim)
    if metric_lpips_:
        results["LPIPS"] = sum(metric_lpips_) / len(metric_lpips_)

    logging.info(f"[{exp_name}] (GPU={device}) 计算结果: {results}")
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

    # 假设有4张GPU，也可改成自己可用的数量
    devices = [f"cuda:{i}" for i in range(4)]
    num_gpus = len(devices)

    # 加载多份 LPIPS 模型
    logging.info(f"准备加载 {num_gpus} 个 LPIPS 模型 (AlexNet)，分别放到 {devices} ...")
    lpips_models = []
    for dev in devices:
        logging.info(f"加载 LPIPS 到 {dev}...")
        model = lpips.LPIPS(net="alex").to(dev)
        lpips_models.append(model)
    logging.info("全部 LPIPS 模型加载完毕。")

    # 多线程：给每个子目录分配一个线程，并按顺序映射到GPU
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

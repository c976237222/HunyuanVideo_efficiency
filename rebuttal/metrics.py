#!/usr/bin/env python3
# coding: utf-8

import argparse
import os
import csv
import math
import numpy as np
import imageio
import torch
import lpips
from tqdm import tqdm
from pytorch_msssim import ssim as calc_ssim_func

# =============== 关键：引入 FVMD 相关 ===============
# 假设通过 pip install . 安装后，可以直接 import fvmd
# 或者您可以以本地模块方式导入
try:
    import fvmd
except ImportError:
    print("[警告] 未找到 fvmd 模块，请确保已正确安装/引用 FVMD-frechet-video-motion-distance。")
    fvmd = None


def parse_args():
    parser = argparse.ArgumentParser(description="比较两个文件夹下同名视频的 SSIM, PSNR, LPIPS，并用FVMD计算整体分布级差异")
    parser.add_argument("--root1", type=str, required=True, help="真实（label）视频文件夹路径")
    parser.add_argument("--root2", type=str, required=True, help="重建（生成）视频文件夹路径")
    parser.add_argument("--csv_output", type=str, required=True, help="输出 CSV 文件路径")
    return parser.parse_args()


def read_video_frames(file_path):
    """
    读取视频的所有帧，返回帧列表（每个元素是 [H, W, C] 的 numpy 数组）。
    """
    try:
        reader = imageio.get_reader(file_path, 'ffmpeg')
        frames = [frame for frame in reader]
        reader.close()
        return frames
    except Exception as e:
        print(f"[警告] 无法读取视频: {file_path}, 错误: {e}")
        return []


def compute_psnr(img1, img2):
    """
    计算两张图像之间的 PSNR（峰值信噪比）。
    img1, img2 均为 numpy 数组，值范围 [0, 255]，格式 [H, W, C]。
    """
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * math.log10(255.0 / math.sqrt(mse))


def compute_lpips_metric(img1, img2, loss_fn):
    """
    计算两帧图像的 LPIPS。
    img1, img2: NumPy数组，形状 [H, W, C]，值范围 [0, 255]。
    LPIPS 模型期望输入在 [-1, 1]，因此需要做归一化转换。
    """
    # 转换为 [C, H, W] 并归一化到 [-1, 1]
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0

    # 放到 GPU（或 CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)
    loss_fn = loss_fn.to(device)

    with torch.no_grad():
        lpips_val = loss_fn(img1_tensor, img2_tensor).item()
    return lpips_val


def compute_metrics_for_videos(label_path, recon_path, lpips_model):
    """
    读取 label 视频和 recon 视频，计算 SSIM、PSNR、LPIPS 并返回其平均值。
    同时将帧列表 (label_frames, recon_frames) 返回，以便后续做 FVMD。
    """
    # 读取帧
    label_frames = read_video_frames(label_path)
    recon_frames = read_video_frames(recon_path)

    if len(label_frames) == 0 or len(recon_frames) == 0:
        return None, None, None, [], []  # 读取失败

    # 截取到帧数相同
    min_frames = min(len(label_frames), len(recon_frames))
    label_frames = label_frames[:min_frames]
    recon_frames = recon_frames[:min_frames]

    # 计算 PSNR、LPIPS（逐帧），最后取平均值
    psnr_vals, lpips_vals = [], []
    for lf, rf in zip(label_frames, recon_frames):
        try:
            psnr_vals.append(compute_psnr(lf, rf))
            lpips_vals.append(compute_lpips_metric(lf, rf, lpips_model))
        except Exception as e:
            print(f"[警告] 逐帧指标计算出错: {e}")
            continue

    if len(psnr_vals) == 0:
        return None, None, None, [], []

    avg_psnr = np.mean(psnr_vals)
    avg_lpips = np.mean(lpips_vals)

    # 计算整体 SSIM：先将所有帧堆叠到 Tensor，再一次性计算
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # [T, H, W, C] => [T, C, H, W] => float in [0,1]
    label_tensor = torch.stack([
        torch.from_numpy(f).permute(2, 0, 1).float() for f in label_frames
    ]).to(device) / 255.0

    recon_tensor = torch.stack([
        torch.from_numpy(f).permute(2, 0, 1).float() for f in recon_frames
    ]).to(device) / 255.0

    # 调整维度以匹配 pytorch_msssim 的输入 [N, C, T, H, W]
    label_tensor = label_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # => [1, C, T, H, W]
    recon_tensor = recon_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # => [1, C, T, H, W]

    with torch.no_grad():
        ssim_val = calc_ssim_func(label_tensor, recon_tensor, data_range=1.0, size_average=True).item()

    return ssim_val, avg_psnr, avg_lpips, label_frames, recon_frames


def compute_fvmd(label_videos, recon_videos):
    """
    演示：使用 FVMD-frechet-video-motion-distance 计算所有 label_videos 与 recon_videos 的分布级 Frechet Video Motion Distance。
    
    - label_videos, recon_videos: List[List[np.ndarray]]，每个元素是一个视频的帧序列；
      帧是 [H, W, C], 取值范围 [0, 255].
    - 返回 fvmd_value, rfvmd_value (若需要)

    下面的实现示例是伪代码，您需根据 FVMD 仓库的真实API（extract_features函数、compute_fvmd函数等）进行改写。
    """
    if fvmd is None:
        print("[错误] fvmd 模块未正确导入，无法计算 FVMD。")
        return None, None

    # 统一帧数：若要求统一长度，可截断或补零
    min_len_label = min(len(vid) for vid in label_videos) if label_videos else 0
    min_len_recon = min(len(vid) for vid in recon_videos) if recon_videos else 0
    min_len = min(min_len_label, min_len_recon)
    if min_len < 1:
        print("[警告] 视频帧数太少，无法计算 FVMD。")
        return None, None

    # 将帧序列堆叠成一个 numpy/tensor 以供 FVMD 特征提取
    # 具体接口可根据实际需要做 resize 或光流提取
    label_video_batch = []
    for frames in label_videos:
        frames = frames[:min_len]
        label_video_batch.append(np.stack(frames, axis=0))  # [T, H, W, C]

    recon_video_batch = []
    for frames in recon_videos:
        frames = frames[:min_len]
        recon_video_batch.append(np.stack(frames, axis=0))  # [T, H, W, C]

    label_video_batch = np.stack(label_video_batch, axis=0)   # => [N1, T, H, W, C]
    recon_video_batch = np.stack(recon_video_batch, axis=0)  # => [N2, T, H, W, C]

    # --------------------------
    # 以下为“示例”调用方式，需参考 FVMD 仓库中如何对视频进行光流+特征提取，再计算Frechet距离
    # --------------------------

    # 1) 提取 motion feature（可能需要 RAFT 等光流模型）
    #    假设 fvmd.extract_features_for_videos(...) 返回 shape=[N, feature_dim] 
    label_features = fvmd.extract_features_for_videos(label_video_batch)   # 伪示例
    recon_features = fvmd.extract_features_for_videos(recon_video_batch)   # 伪示例

    # 2) 计算 FVMD（Frechet Distance）
    fvmd_value = fvmd.compute_fvmd(label_features, recon_features)  # 伪示例

    # 3) 若需要 rFVMD，可做简单缩放或基于某论文/项目的定义
    #    这里仅演示做个除以 10.0
    rfvmd_value = fvmd_value / 10.0

    return fvmd_value, rfvmd_value


def main():
    args = parse_args()

    root_label = args.root1
    root_recon = args.root2
    csv_path = args.csv_output

    # 初始化 LPIPS 模型
    lpips_model = lpips.LPIPS(net='alex').eval()

    # 找到重建视频目录下所有 mp4
    recon_mp4_list = sorted([
        f for f in os.listdir(root_recon) if f.lower().endswith('.mp4')
    ])

    # 打开 CSV，写表头
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "ssim", "psnr", "lpips"])
    print(f"[信息] 将结果写入 CSV: {csv_path}")

    # 收集全部视频的帧序列，以便后面整体计算 FVMD
    all_label_videos = []
    all_recon_videos = []

    # 遍历重建视频，与 label 文件夹下同名视频对比，计算逐视频的三种指标
    for mp4_name in tqdm(recon_mp4_list, desc="Processing per-video metrics"):
        recon_path = os.path.join(root_recon, mp4_name)
        label_path = os.path.join(root_label, mp4_name)

        if not os.path.exists(label_path):
            print(f"[警告] Label 文件不存在: {label_path}，跳过计算。")
            continue

        ssim_val, psnr_val, lpips_val, label_frames, recon_frames = compute_metrics_for_videos(
            label_path, recon_path, lpips_model)

        # 如果为空说明读取或计算有问题，直接跳过
        if ssim_val is None or psnr_val is None or lpips_val is None:
            print(f"[警告] 指标计算失败，跳过: {mp4_name}")
            continue

        # 写入 CSV（逐视频）
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                recon_path,
                round(ssim_val, 4),
                round(psnr_val, 4),
                round(lpips_val, 4)
            ])

        # 收集帧序列，用于后续整体 FVMD
        all_label_videos.append(label_frames)
        all_recon_videos.append(recon_frames)

    # ============ 计算整批视频的 FVMD (Frechet Video Motion Distance) ===============
    fvmd_val, rfvmd_val = compute_fvmd(all_label_videos, all_recon_videos)

    if fvmd_val is not None and rfvmd_val is not None:
        print(f"[信息] FVMD计算完成：FVMD={fvmd_val:.4f}, rFVMD={rfvmd_val:.4f}")
        # 追加到CSV末行
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "FVMD/rFVMD (entire dataset)",
                "FVMD=%.4f" % fvmd_val,
                "rFVMD=%.4f" % rfvmd_val,
                "-"
            ])
    else:
        print("[警告] FVMD 计算失败，可能 fvmd 模块未正确安装或出错。")

    print(f"\n[完成] 已写入逐视频的 SSIM/PSNR/LPIPS，以及在CSV结尾追加 FVMD结果：{csv_path}")


if __name__ == "__main__":
    main()

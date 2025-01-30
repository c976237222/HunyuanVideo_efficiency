#!/usr/bin/env python3
# compute_metrics_per_tile.py

import argparse
import os
import numpy as np
import math
import csv
from glob import glob
import imageio
import lpips
import torch
from tqdm import tqdm
import logging
from datetime import datetime
from pytorch_msssim import ssim as calc_ssim_func  # 新增导入

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def parse_args():
    parser = argparse.ArgumentParser(description="计算视频质量指标")
    parser.add_argument("--root1", type=str, required=True, help="参考视频目录")
    parser.add_argument("--root2", type=str, required=True, help="生成视频目录")
    parser.add_argument("--csv-output", type=str, required=True, help="CSV输出路径")
    return parser.parse_args()

def parse_filename(filename):
    """解析文件名结构：filename|chunk_idx|tile_ci|.mp4"""
    try:
        parts = filename.split('|')
        if len(parts) < 4:
            raise ValueError("文件名格式不正确，缺少必要部分")
        return {
            'file_name': parts[0],
            'chunk_idx': int(parts[1]),
            'tile_ci': float(parts[2]),
        }
    except Exception as e:
        logging.error(f"文件名解析失败: {filename} - {e}")
        return None

def compute_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * math.log10(255.0 / math.sqrt(mse))

def compute_lpips_metric(img1, img2, loss_fn):
    """
    计算LPIPS指标
    img1, img2: NumPy数组，形状为 (H, W, C)，范围为 [0, 255]
    """
    img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to("cuda") / 127.5 - 1.0
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().to("cuda") / 127.5 - 1.0
    with torch.no_grad():
        return loss_fn(img1_tensor, img2_tensor).item()

def read_video_frames(file_path):
    try:
        reader = imageio.get_reader(file_path, 'ffmpeg')
        frames = [frame for frame in reader]
        reader.close()
        return frames
    except Exception as e:
        logging.error(f"视频读取失败: {file_path} - {e}")
        return None

def process_video_pair(ref_path, gen_path, lpips_model):
    """处理单个视频对并返回指标"""
    # 读取视频帧
    ref_frames = read_video_frames(ref_path)
    gen_frames = read_video_frames(gen_path)
    
    if not ref_frames or not gen_frames:
        return None
    
    # 确保帧数一致
    min_frames = min(len(ref_frames), len(gen_frames))
    ref_frames = ref_frames[:min_frames]
    gen_frames = gen_frames[:min_frames]
    
    if min_frames < 2:
        logging.warning(f"视频帧数过少: {ref_path} 和 {gen_path}")
        return None
    
    # 将帧转换为张量
    ref_tensor = torch.stack([torch.from_numpy(frame).permute(2, 0, 1).float() for frame in ref_frames]).to("cuda")  # [T, C, H, W]
    gen_tensor = torch.stack([torch.from_numpy(frame).permute(2, 0, 1).float() for frame in gen_frames]).to("cuda")  # [T, C, H, W]
    
    # 归一化到 [0, 1]
    ref_tensor = ref_tensor / 255.0
    gen_tensor = gen_tensor / 255.0
    
    # 计算PSNR和LPIPS（逐帧）
    psnr_values = []
    lpips_values = []
    for ref_frame, gen_frame in zip(ref_frames, gen_frames):
        try:
            psnr = compute_psnr(ref_frame, gen_frame)
            lpips_val = compute_lpips_metric(ref_frame, gen_frame, lpips_model)
            psnr_values.append(psnr)
            lpips_values.append(lpips_val)
        except Exception as e:
            logging.warning(f"帧处理失败: {e}")
            continue
    
    if not psnr_values:
        return None
    
    # 计算平均PSNR和LPIPS
    avg_psnr = np.mean(psnr_values)
    avg_lpips = np.mean(lpips_values)
    
    # 计算SSIM（使用pytorch_msssim）
    try:
        # 将张量形状调整为 [batch_size, C, T, H, W] 以适应pytorch_msssim的输入
        ref_tensor_ssim = ref_tensor.unsqueeze(0)  # [1, T, C, H, W]
        gen_tensor_ssim = gen_tensor.unsqueeze(0)  # [1, T, C, H, W]
        
        # 重排维度到 [batch_size, C, T, H, W]
        ref_tensor_ssim = ref_tensor_ssim.permute(0, 2, 1, 3, 4)
        gen_tensor_ssim = gen_tensor_ssim.permute(0, 2, 1, 3, 4)
        
        # 计算SSIM
        ssim_val = calc_ssim_func(ref_tensor_ssim, gen_tensor_ssim, data_range=1.0, size_average=True).item()
    except Exception as e:
        logging.error(f"SSIM计算失败: {ref_path} 和 {gen_path} - {e}")
        ssim_val = None
    
    return {
        'psnr': avg_psnr,
        'ssim': ssim_val,
        'lpips': avg_lpips,
    }

def main():
    args = parse_args()
    
    # 初始化CSV
    with open(args.csv_output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file_name', 'chunk_idx', 'tile_ci', 'psnr', 'ssim', 'lpips'])
    
    # 初始化LPIPS模型
    lpips_model = lpips.LPIPS(net='alex').eval().to("cuda")
    
    # 获取匹配的视频文件
    ref_videos = sorted(glob(os.path.join(args.root1, "*.mp4")))
    gen_videos = sorted(glob(os.path.join(args.root2, "*.mp4")))
    
    matched_pairs = []
    for gen_vid in gen_videos:
        gen_filename = os.path.basename(gen_vid)
        ref_vid = os.path.join(args.root1, gen_filename)
        if os.path.exists(ref_vid):
            matched_pairs.append((ref_vid, gen_vid))
        else:
            logging.warning(f"参考视频不存在: {ref_vid}")
    
    logging.info(f"总共找到 {len(matched_pairs)} 对匹配的视频。")
    
    # 处理每个视频对
    for ref_path, gen_path in tqdm(matched_pairs, desc="Processing videos"):
        try:
            # 解析元数据
            filename = os.path.basename(gen_path)
            meta = parse_filename(filename)
            if not meta:
                continue
                
            # 计算指标
            metrics = process_video_pair(ref_path, gen_path, lpips_model)
            if not metrics or metrics['ssim'] is None:
                logging.warning(f"指标计算失败: {ref_path} 和 {gen_path}")
                continue
                
            # 写入CSV
            with open(args.csv_output, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    meta['file_name'],
                    meta['chunk_idx'],
                    meta['tile_ci'],
                    round(metrics['psnr'], 4),
                    round(metrics['ssim'], 4),
                    round(metrics['lpips'], 4)
                ])
        except Exception as e:
            logging.error(f"处理失败: {gen_path} - {e}")
            continue

if __name__ == "__main__":
    main()

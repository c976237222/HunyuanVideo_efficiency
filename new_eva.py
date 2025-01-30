#!/usr/bin/env python3
# compute_metrics_per_tile.py

import argparse
import os
import pandas as pd
import torch
import cv2
from pytorch_msssim import ssim as calc_ssim_func
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def parse_args():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="计算视频 SSIM 并更新 CSV 文件。")
    parser.add_argument("--input_folder", type=str, required=True, help="输入视频文件夹路径。")
    parser.add_argument("--input_csv", type=str, required=True, help="输入 CSV 文件路径。")
    parser.add_argument("--output_csv", type=str, required=True, help="输出 CSV 文件路径。")
    parser.add_argument("--max_files", type=int, default=None, help="最大处理文件数量。设置为 None 处理所有文件。")
    return parser.parse_args()

def parse_filename(filename):
    """
    解析文件名，提取 file_name 和 chunk_idx。
    假设文件名格式为: file_name|chunk_idx|tiles_ci[0]|.mp4
    """
    parts = filename.split('|')
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {filename}")
    file_name = parts[0].strip()
    chunk_idx = int(parts[1].strip())  # 转换为整数
    return file_name, chunk_idx

def load_video_as_tensor(video_path):
    """
    使用 OpenCV 加载视频，并将其转换为 PyTorch 张量。
    返回的张量形状为 [T, C, H, W]，归一化到 [0, 1]。
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 转换为 RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转换为 PyTorch 张量，并调整维度顺序到 [C, H, W]
        frame = torch.from_numpy(frame).permute(2, 0, 1).float()  # [C, H, W]
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"No frames found in video: {video_path}")
    video_tensor = torch.stack(frames) / 255.0  # [T, C, H, W], 归一化到 [0,1]
    return video_tensor

def calculate_video_content_info_ssim(videos1):
    """
    计算视频的 SSIM 值。
    
    参数:
        videos1 (torch.Tensor): 视频张量，形状为 [batch_size, T, C, H, W]
    
    返回:
        List[float]: 每个视频的 SSIM 值
    """
    logging.info("开始计算 SSIM...")
    ci_ssim_results = []

    for video_num in tqdm(range(videos1.shape[0]), desc="计算 SSIM"):
        # 获取一个视频
        video1 = videos1[video_num].mul(255).add_(0.5).clamp_(0, 255)  # 反归一化到 [0, 255]

        img1 = video1[:-1]  # [T-1, C, H, W]
        img2 = video1[1:]   # [T-1, C, H, W]
        # 计算 SSIM，保持 batch_size=1
        device = "cuda"
        img1 = img1.to(device)
        img2 = img2.to(device)
        ssim_val = calc_ssim_func(img1, img2, data_range=255, size_average=True)
        logging.debug(f"Video {video_num + 1}: SSIM = {ssim_val.item()}")
        ci_ssim_results.append(ssim_val.item())
    return ci_ssim_results

def main():
    args = parse_args()

    # 打印参数以供调试
    logging.info(f"输入视频文件夹: {args.input_folder}")
    logging.info(f"输入 CSV 文件: {args.input_csv}")
    logging.info(f"输出 CSV 文件: {args.output_csv}")
    logging.info(f"最大处理文件数量: {args.max_files if args.max_files else '全部'}")

    # 获取文件夹中的所有 MP4 文件
    files1 = sorted([f for f in os.listdir(args.input_folder) if f.endswith('.mp4')])

    # 如果指定了最大文件数量，则进行切片
    if args.max_files is not None:
        files1 = files1[:args.max_files]
        logging.info(f"已限制处理的文件数量为: {args.max_files}")

    logging.info(f"总共找到 {len(files1)} 个文件用于处理。")

    # 读取原始的 CSV 文件
    try:
        df = pd.read_csv(args.input_csv)
        logging.info(f"成功读取 CSV 文件: {args.input_csv}")
    except Exception as e:
        logging.error(f"读取 CSV 文件失败: {args.input_csv} - {e}")
        return

    # 检查 CSV 文件中是否包含 'file_name' 和 'chunk_idx' 列
    required_columns = {'file_name', 'chunk_idx', 'tile_ci'}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        logging.error(f"CSV 文件中缺少必要的列：{missing}")
        return

    # 确保 'chunk_idx' 在 DataFrame 中是整数类型
    df['chunk_idx'] = df['chunk_idx'].astype(int)

    # 创建两个列表用于存储视频张量和对应的键
    video_tensors = []
    keys = []

    logging.info("加载视频并准备批次...")
    for filename in tqdm(files1, desc=f"加载视频 (最多 {len(files1)} 个)"):
        file1_path = os.path.join(args.input_folder, filename)

        try:
            file_name, chunk_idx = parse_filename(filename)
            logging.debug(f"解析文件名: file_name={file_name}, chunk_idx={chunk_idx}")
        except ValueError as e:
            logging.warning(e)
            continue

        try:
            video_tensor = load_video_as_tensor(file1_path)  # [T, C, H, W]
            video_tensors.append(video_tensor)
            keys.append((file_name, chunk_idx))
            logging.debug(f"成功加载视频: {file1_path}")
        except ValueError as e:
            logging.warning(e)
            continue

    if not video_tensors:
        logging.error("没有成功加载任何视频。")
        raise ValueError("没有成功加载任何视频。")

    # 找到所有视频的最小帧数
    min_frames = min([vid.shape[0] for vid in video_tensors])
    logging.info(f"所有视频的最小帧数: {min_frames}")

    # 截取所有视频到最小帧数
    video_tensors = [vid[:min_frames] for vid in video_tensors]

    # 堆叠成一个批次张量 [batch_size, T, C, H, W]
    logging.info("堆叠视频张量为一个批次...")
    try:
        batch_tensor = torch.stack(video_tensors)  # [batch_size, T, C, H, W]
        logging.debug(f"批次张量形状: {batch_tensor.shape}")
    except RuntimeError as e:
        logging.error(f"Error stacking videos: {e}")
        raise

    # 计算 SSIM
    ssim_vals = calculate_video_content_info_ssim(batch_tensor)

    # 创建一个 DataFrame 来存储计算得到的 SSIM 值
    ssim_df = pd.DataFrame(keys, columns=['file_name', 'chunk_idx'])
    ssim_df['tile_ci'] = ssim_vals

    logging.info("SSIM 计算完成，准备更新 CSV 文件。")

    # 确保 'chunk_idx' 在 ssim_df 中是整数类型
    ssim_df['chunk_idx'] = ssim_df['chunk_idx'].astype(int)

    # 合并原始 DataFrame 和 SSIM DataFrame
    # 以 'file_name' 和 'chunk_idx' 作为键
    try:
        df_updated = pd.merge(df, ssim_df, on=['file_name', 'chunk_idx'], how='left', suffixes=('', '_new'))
        logging.info("成功合并 DataFrame。")
    except Exception as e:
        logging.error(f"合并 DataFrame 失败: {e}")
        return

    # 检查哪些行成功匹配并更新
    matched = df_updated['tile_ci_new'].notna()
    logging.info(f"成功匹配并更新 {matched.sum()} 行。")
    logging.info(f"未匹配的行数: {len(df_updated) - matched.sum()}")

    # 使用新计算的 SSIM 值替换原有的 'tile_ci' 列
    df_updated.loc[matched, 'tile_ci'] = df_updated.loc[matched, 'tile_ci_new']

    # 删除辅助列
    df_updated.drop(columns=['tile_ci_new'], inplace=True)

    # 保存更新后的 CSV 文件
    try:
        df_updated.to_csv(args.output_csv, index=False)
        logging.info(f"已将更新后的数据保存到 {args.output_csv}")
    except Exception as e:
        logging.error(f"保存 CSV 文件失败: {args.output_csv} - {e}")

if __name__ == "__main__":
    main()

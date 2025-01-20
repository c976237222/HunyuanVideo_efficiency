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

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 指定两个文件夹路径（直接修改这里）
ROOT1 = "/home/hanling/HunyuanVideo_efficiency/video_data/video_data_100_240p"
ROOT2 = "/home/hanling/HunyuanVideo_efficiency/video_data/vae_output_videos"

# 结果存储路径
RESULTS_DIR = "/home/hanling/HunyuanVideo_efficiency/evaluation/results"

# 确保结果目录存在
os.makedirs(RESULTS_DIR, exist_ok=True)

# 计算 PSNR
def compute_psnr(img1, img2):
    mse = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

# 计算 SSIM
def compute_ssim(img1, img2):
    if np.all(img1 == img1[0, 0, 0]) or np.all(img2 == img2[0, 0, 0]):
        return 1.0
    return compare_ssim(img1, img2, data_range=img1.max() - img1.min(), channel_axis=-1)

# 计算 LPIPS
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

    img1_tensor = img1_tensor * 2 - 1  # 归一化到 [-1, 1]
    img2_tensor = img2_tensor * 2 - 1

    return loss_fn(img1_tensor, img2_tensor).item()

# 读取视频帧
def read_video(file_path):
    try:
        video = imageio.get_reader(file_path)
        frames = [frame for frame in video]
        video.close()
        return frames
    except Exception as e:
        logging.error(f"读取视频失败 {file_path}: {e}")
        return []

# 保存结果到带时间戳的文件
def save_results(results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成时间戳
    output_file = os.path.join(RESULTS_DIR, f"metrics_{timestamp}.txt")

    with open(output_file, "w") as f:
        f.write("\n")
        f.write(f"Root1: {ROOT1}\n")
        f.write(f"Root2: {ROOT2}\n")
        f.write(f"Timestamp: {timestamp}\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value}\n")
        f.write("\n")

    logging.info(f"结果已保存到 {output_file}")

# 主要处理逻辑
def main():
    # 获取两个文件夹的视频文件列表
    videos1 = sorted(glob(os.path.join(ROOT1, "*.mp4")))
    videos2 = sorted(glob(os.path.join(ROOT2, "*.mp4")))

    # 提取文件名（不含路径）
    filenames1 = {os.path.basename(video) for video in videos1}
    filenames2 = {os.path.basename(video) for video in videos2}

    # 取交集，找到匹配的文件
    common_filenames = sorted(filenames1 & filenames2)

    if not common_filenames:
        logging.error("未找到匹配的视频文件。")
        return

    logging.info(f"找到 {len(common_filenames)} 对匹配视频。")

    # 根据匹配文件名生成路径
    matched_videos1 = [os.path.join(ROOT1, filename) for filename in common_filenames]
    matched_videos2 = [os.path.join(ROOT2, filename) for filename in common_filenames]

    # Metrics storage
    metric_psnr = []
    metric_ssim = []
    metric_lpips = []

    # 初始化 LPIPS 模型
    lpips_model = lpips.LPIPS(net="alex").to("cuda:0")
    logging.info("已加载 LPIPS 模型 (AlexNet)。")

    # 逐对处理匹配的视频
    for vid1_path, vid2_path in tqdm(
        zip(matched_videos1, matched_videos2), total=len(common_filenames), desc="处理视频中"
    ):
        vid1_frames = read_video(vid1_path)
        vid2_frames = read_video(vid2_path)

        if not vid1_frames or not vid2_frames:
            logging.error(f"跳过无法读取的视频对: {vid1_path}, {vid2_path}")
            continue

        # 对两组帧进行对比
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
                metric_lpips.append(lpips_value)
            except Exception as e:
                logging.error(f"计算 LPIPS 失败: {e}")

    # 计算平均值
    results = {}
    if metric_psnr:
        results["PSNR"] = sum(metric_psnr) / len(metric_psnr)
    if metric_ssim:
        results["SSIM"] = sum(metric_ssim) / len(metric_ssim)
    if metric_lpips:
        results["LPIPS"] = sum(metric_lpips) / len(metric_lpips)

    # 输出结果
    logging.info(f"计算结果: {results}")
    save_results(results)

if __name__ == "__main__":
    main()

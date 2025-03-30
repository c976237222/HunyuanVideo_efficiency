import os
import argparse

# 提前解析参数
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0', help='CUDA device ID')
parser.add_argument('--ratio', type=str, default='4x', help='compression ratio')
parser.add_argument('--fps', type=str, default='15', help='video fps')
args = parser.parse_args()

# 设置显卡（必须在 import torch 前）
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

import csv
import imageio
import torch
import json
import numpy as np
from fvmd import fvmd
from fvmd.datasets.video_datasets import VideoDatasetNP
from fvmd.extract_motion_features import calc_hist
from fvmd.frechet_distance import calculate_fd_given_vectors
from fvmd.keypoint_tracking import track_keypoints
from calculate_fvd import calculate_fvd
from calculate_psnr import calculate_psnr
from calculate_lpips import calculate_lpips
from pytorch_msssim import ssim as calc_ssim_func
from tqdm import tqdm

def read_video_as_tensor(video_path, device, min_frames=10):
    if not os.path.exists(video_path):
        print(f"[警告] 视频文件不存在: {video_path}")
        return None

    try:
        reader = imageio.get_reader(video_path, 'ffmpeg')
        frames = [frame for frame in reader]
        reader.close()
    except Exception as e:
        print(f"[错误] 无法读取视频: {video_path}, 原因: {e}")
        return None

    if len(frames) < min_frames:
        print(f"[警告] 视频帧数({len(frames)})不足 {min_frames}，跳过计算 FVD")
        return None

    processed_frames = []
    for f in frames:
        if len(f.shape) == 2:
            f = f[..., None]

        f_tensor = torch.from_numpy(f).float() / 255.0
        if f_tensor.shape[-1] == 1:
            f_tensor = f_tensor.repeat(1, 1, 3)
        f_tensor = f_tensor.permute(2, 0, 1)
        processed_frames.append(f_tensor)

    video_tensor = torch.stack(processed_frames, dim=0).unsqueeze(0).to(device)
    return video_tensor

def compute_pytorchmsssim_SSIM(videos1, videos2):
    print("calculate_ssim...")
    v1 = videos1.permute(0, 2, 1, 3, 4)
    v2 = videos2.permute(0, 2, 1, 3, 4)

    with torch.no_grad():
        ssim_val = calc_ssim_func(v1, v2, data_range=1.0, size_average=True)
    return ssim_val.item()

def fvdm(video1, video2, log_dir):
    print("calculate_fvdm...")
    gt_dataset = VideoDatasetNP(video1.permute(0, 1, 3, 4, 2).cpu().numpy())
    gen_dataset = VideoDatasetNP(video2.permute(0, 1, 3, 4, 2).cpu().numpy())

    velo_gen, velo_gt, acc_gen, acc_gt = track_keypoints(log_dir=log_dir, gen_dataset=gen_dataset,
                                                          gt_dataset=gt_dataset, v_stride=1)
    B = velo_gen.shape[0]
    gt_v_hist = calc_hist(velo_gt).reshape(B, -1)
    gen_v_hist = calc_hist(velo_gen).reshape(B, -1)
    gt_a_hist = calc_hist(acc_gt).reshape(B, -1)
    gen_a_hist = calc_hist(acc_gen).reshape(B, -1)

    gt_hist = np.concatenate((gt_v_hist, gt_a_hist), axis=1)
    gen_hist = np.concatenate((gen_v_hist, gen_a_hist), axis=1)
    fvmd_value = calculate_fd_given_vectors(gt_hist, gen_hist)
    return fvmd_value

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[信息] 使用设备: {device}")

    label_dir = f"/mnt/public/wangsiyuan/HunyuanVideo_efficiency/analysis/30hz_240p_reconstructed_nothing_4x_label"
    recon_dir = f"/mnt/public/wangsiyuan/HunyuanVideo_efficiency/analysis/30hz_240p_reconstructed_nothing_4x"
    csv_output_path = f"{args.fps}hz_240p_{args.ratio}.csv"

    video_list = sorted([f for f in os.listdir(label_dir) if f.lower().endswith('.mp4')])
    results_for_csv = []

    for mp4_name in tqdm(video_list):
        label_path = os.path.join(label_dir, mp4_name)
        recon_path = os.path.join(recon_dir, mp4_name)

        print(f"\n[信息] 开始处理: {mp4_name}")

        videos1 = read_video_as_tensor(label_path, device)
        videos2 = read_video_as_tensor(recon_path, device)

        if videos1 is None or videos2 is None:
            print(f"[警告] 无法进行指标计算: {mp4_name}")
            continue

        T1, T2 = videos1.shape[1], videos2.shape[1]
        min_T = min(T1, T2)
        videos1 = videos1[:, :min_T]
        videos2 = videos2[:, :min_T]

        log_dir = "/mnt/public/wangsiyuan/HunyuanVideo_efficiency/rebuttal/common_metrics_on_video_quality/fvdm_log"
        fvdm_value = fvdm(videos1, videos2, log_dir)

        fvd_val = None
        if min_T >= 10:
            fvd_res = calculate_fvd(videos1, videos2, device, method='styleganv', only_final=True)
            fvd_val = float(fvd_res['value'][0])

        ssim_val = compute_pytorchmsssim_SSIM(videos1, videos2)
        psnr_val = float(calculate_psnr(videos1.cpu(), videos2.cpu(), only_final=True)['value'][0])
        lpips_val = float(calculate_lpips(videos1, videos2, device, only_final=True)['value'][0])

        results_for_csv.append([
            os.path.abspath(recon_path),
            ssim_val,
            psnr_val,
            lpips_val,
            fvd_val if fvd_val is not None else "",
            fvdm_value if fvdm_value is not None else ""
        ])

    with open(csv_output_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "ssim", "psnr", "lpips", "fvd", "fvdm"])
        writer.writerows(results_for_csv)

    print(f"\n✅ 所有视频处理完成，结果已保存至: {csv_output_path}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# infer.py

import argparse
import os
import torch
from torch.utils.data import DataLoader
from loguru import logger

from dataset_processor.dataset_loader import VideoTensorDataset
from hyvideo.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from hyvideo.vae import load_vae
from hyvideo.utils.file_utils import save_videos_grid

def fft_latents(latent, t_group_size=3, low_freq_factor=1.5, high_freq_factor=0.5):
    """
    对给定的潜在张量 `latent` (即 z) 按组（t_group_size）进行高低频分离，
    并通过系数调整高频和低频部分的强度。
    
    - latent: 输入的潜在张量
    - t_group_size: 每组时间步数
    - low_freq_factor: 调整低频部分的系数
    - high_freq_factor: 调整高频部分的系数
    """
    print(latent.shape)
    batch_size, num_channels, num_timesteps, height, width = latent.shape
    device = latent.device  # 使用输入张量的设备
    dtype = latent.dtype  # 使用输入张量的数据类型

    # 计算 T 维度需要多少个 generator
    num_t_groups = (num_timesteps + t_group_size - 1) // t_group_size  # 向上取整
    latents = torch.zeros(latent.shape, device=device, dtype=dtype)

    # 按组处理
    for t_group in range(num_t_groups):
        t_start = t_group * t_group_size
        t_end = min((t_group + 1) * t_group_size, num_timesteps)  # 确保不会超出 T 维度

        # 取当前组的 z
        group_latent = latent[:, :, t_start:t_end, :, :]

        # 1. **进行 3D FFT (T, H, W)**
        group_fft = torch.fft.fftn(group_latent, dim=(-3, -2, -1))  # 计算 T, H, W 维度的频域变换

        # 2. **分离低频和高频**
        cutoff_t = num_timesteps // 4  # 时间维度的低频截断
        cutoff_h = height // 4  # 空间维度的低频截断
        cutoff_w = width // 4

        mask = torch.zeros_like(group_fft)
        mask[:, :, :cutoff_t, :cutoff_h, :cutoff_w] = 1  # 低频区域（左上角）
        mask[:, :, -cutoff_t:, :cutoff_h, :cutoff_w] = 1  # 低频区域（右下角）
        mask[:, :, :cutoff_t, -cutoff_h:, :cutoff_w] = 1  # 低频区域（左下角）
        mask[:, :, -cutoff_t:, -cutoff_h:, :cutoff_w] = 1  # 低频区域（右上角）

        low_freq = group_fft * mask  # 低频部分
        high_freq = group_fft * (1 - mask)  # 高频部分

        # 3. **应用系数来调整高频和低频**
        low_freq = low_freq * low_freq_factor  # 增强/减少低频
        high_freq = high_freq * high_freq_factor  # 增强/减少高频

        # 4. **合并低频和高频部分**
        mixed_fft = low_freq + high_freq

        # 5. **逆傅里叶变换**（将频域信号转换回时域）
        new_group = torch.fft.ifftn(mixed_fft, dim=(-3, -2, -1)).real  # 变换回时域

        # 6. **归一化，确保高斯分布**
        new_group = (new_group - new_group.mean()) / (new_group.std() + 1e-6)

        # 将处理后的组存入 latents
        latents[:, :, t_start:t_end, :, :] = new_group

    return latents



def infer_vae(model: AutoencoderKLCausal3D,
              dataloader: DataLoader,
              device: str,
              output_dir: str,
              max_files: int = None,
              mp4: bool = True):
    """
    Perform inference using the VAE model on video tensors.
    """
    model.to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    for batch_idx, (video_tensor, file_name) in enumerate(dataloader):
        if max_files is not None and batch_idx >= max_files:
            break  # Stop processing after reaching the max number of files

        # 去掉 .pt 后缀
        file_name = file_name[0].replace(".pt", "")

        # Move to device
        video_tensor = video_tensor.to(device, dtype=torch.float16)
        logger.info(f"Processing {file_name}, video shape: {video_tensor.shape}")

        with torch.no_grad():
            posterior = model.encode(video_tensor, return_dict=False)[0]
            z = posterior.mode()
            reconstructed_video = model.decode(fft_latents(z.to(dtype=torch.float32),t_group_size=3, low_freq_factor=1.0, high_freq_factor=1.0).to(dtype=torch.float16), return_dict=False)[0]
            #reconstructed_video = model.decode(z, return_dict=False)[0]
        reconstructed_video = reconstructed_video.cpu().float()
        output_path = os.path.join(output_dir, f"{file_name}.pt")
        torch.save(reconstructed_video, output_path)
        logger.info(f"Saved reconstructed video to {output_path}, shape: {reconstructed_video.shape}")

        # Optionally save mp4
        if mp4:
            save_path = os.path.join(output_dir, f"{file_name}.mp4")
            save_videos_grid(reconstructed_video, save_path, fps=24, rescale=True)
            logger.info(f'Sample saved to: {save_path}')


def parse_args():
    parser = argparse.ArgumentParser(description="VAE Inference script for video tensors.")
    parser.add_argument("--tensor-dir", type=str,
                        help="Directory containing input .pt video tensors.")
    parser.add_argument("--output-dir", type=str,
                        help="Directory to save the reconstructed videos.")
    parser.add_argument("--vae-path", type=str, default="ckpts/hunyuan-video-t2v-720p/vae",
                        help="Path to VAE checkpoint directory (contains pytorch_model.pt).")
    parser.add_argument("--config-json", type=str, default="t_ops_config.json",
                        help="Path to the T-ops config JSON file.")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Max number of input files to process (for quick testing).")
    parser.add_argument("--mp4", action="store_true",
                        help="If set, also save outputs as .mp4 videos.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for data loader.")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers for data loader.")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info(f"Running inference with args: {args}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载 VAE 模型
    logger.info("Loading VAE...")
    vae, _, s_ratio, t_ratio = load_vae(
        vae_type="884-16c-hy",           # 这里和你们项目里保持一致
        vae_precision="fp16",
        logger=logger,
        vae_path=args.vae_path,
        device=device,
    )
    logger.info("VAE loaded.")

    # 如果你想启用 tiling，可在此处启用
    # vae.enable_tiling()

    # 加载数据集
    dataset = VideoTensorDataset("/home/siyuan/HunyuanVideo_efficiency/results/idea_0/vae/data/tensor")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # 运行推理
    infer_vae(vae, dataloader, device, "/home/siyuan/HunyuanVideo_efficiency/results/idea_0/vae/data", max_files=1)


if __name__ == "__main__":
    main()

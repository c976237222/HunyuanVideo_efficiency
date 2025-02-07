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


def save_model_architecture_to_file(module, file_path: str):
    """将模型架构保存到文件，示例功能不变，可选。"""
    with open(file_path, "w") as f:
        def write_full_model(module, indent: int = 0):
            prefix = " " * indent
            f.write(f"{prefix}{module.__class__.__name__}:\n")
            for name, sub_module in module.named_children():
                f.write(f"{prefix}  ({name}): {sub_module}\n")
                write_full_model(sub_module, indent + 4)
        write_full_model(module)


from hyvideo.vae.adaptive_temporal_tiling import AdaptiveTemporalTiling


def infer_vae_adaptive(model: AutoencoderKLCausal3D,
                       dataloader: DataLoader,
                       device: str,
                       output_dir: str,
                       max_files: int = None,
                       mp4: bool = False):
    """
    利用我们修改过的 temporal_tiled_encode / temporal_tiled_decode，
    在 encode 阶段对每个 tile 调用 ffprobe -> 选用相应 VAE -> encoder；
    在 decode 阶段，对应地选用相同 VAE -> decoder。
    """
    model.to(device)
    model.eval()

    # 初始化适配工具
    adaptor = AdaptiveTemporalTiling(
        vae_ckpt_path="ckpts/hunyuan-video-t2v-720p/vae",
        device=device,
        vae_precision="fp16",
    )

    os.makedirs(output_dir, exist_ok=True)

    for batch_idx, (video_tensor, file_name) in enumerate(dataloader):
        if max_files is not None and batch_idx >= max_files:
            break

        # 去掉 .pt 后缀
        file_name = file_name[0].replace(".pt", "")

        # Move to device
        video_tensor = video_tensor.to(device, dtype=torch.float16)
        logger.info(f"Processing {file_name}, video shape: {video_tensor.shape}")

        with torch.no_grad():
            # =========== Encode阶段 ===========
            posterior_out = model.temporal_tiled_encode(
                x=video_tensor,
                adaptor=adaptor,  # 传入我们自适应类
                return_dict=True
            )
            # posterior_out 是 AutoencoderKLOutput
            posterior = posterior_out.latent_dist  # 里面包含 mean/var
            print(posterior.mode().shape)
            # =========== Decode阶段 ===========
            reconstructed_video = model.temporal_tiled_decode(
                z=posterior.mode(),  # 或 sample() 取随机
                adaptor=adaptor,     # 同样传入
                return_dict=True
            ).sample

            # Save the reconstructed video
            reconstructed_video = reconstructed_video.cpu().float()
            output_path = os.path.join(output_dir, f"{file_name}.pt")
            torch.save(reconstructed_video, output_path)
            logger.info(f"Saved reconstructed video to {output_path}, shape: {reconstructed_video.shape}")

            # Optionally save mp4
            if mp4:
                save_path = os.path.join(output_dir, f"{file_name}.mp4")
                save_videos_grid(reconstructed_video, save_path, fps=15, rescale=True)
                logger.info(f'Sample saved to: {save_path}')

def infer_vae(model: AutoencoderKLCausal3D,
              dataloader: DataLoader,
              device: str,
              output_dir: str,
              max_files: int = None,
              mp4: bool = False):
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
            # Encode and decode video
            reconstructed_video = model(
                video_tensor,
                return_dict=False,
                return_posterior=True,
                sample_posterior=False
            )[0]

        # Save the reconstructed video in .pt format
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
    parser.add_argument("--tensor-dir", type=str, required=True,
                        help="Directory containing input .pt video tensors.")
    parser.add_argument("--output-dir", type=str, required=True,
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
    parser.add_argument("--use-adaptive", action="store_true", help="Use the adaptive VAE class.")
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
        t_ops_config_path=args.config_json,
        test=True,
    )
    logger.info("VAE loaded.")

    # 如果你想启用 tiling，可在此处启用
    vae.enable_tiling()
    vae.tile_overlap_factor=0
    # 加载数据集
    dataset = VideoTensorDataset(args.tensor_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 运行推理
    infer_vae_adaptive(vae, dataloader, device, args.output_dir, max_files=args.max_files, mp4=args.mp4)


if __name__ == "__main__":
    main()

from hyvideo.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from hyvideo.vae import load_vae
from loguru import logger
import torch
from torch.nn import Module
from video_dataloader import get_single_batch_dataloader
from torch.utils.data import DataLoader
import os
import os
import torch
import numpy as np
import os
from hyvideo.utils.file_utils import save_videos_grid

def save_model_architecture_to_file(module: Module, file_path: str):
    """将模型架构保存到文件"""
    with open(file_path, "w") as f:
        def write_full_model(module: Module, indent: int = 0):
            prefix = " " * indent
            f.write(f"{prefix}{module.__class__.__name__}:\n")
            for name, sub_module in module.named_children():
                f.write(f"{prefix}  ({name}): {sub_module}\n")
                write_full_model(sub_module, indent + 4)

        write_full_model(module)

def infer_vae(model: AutoencoderKLCausal3D, dataloader: DataLoader, device: str, output_dir: str, max_files: int = None, mp4: bool = False):
    """
    Perform inference using the VAE model on video tensors.
    Args:
        model: Pretrained VAE model.
        dataloader: DataLoader for video tensors.
        device: Device to run the inference on ('cuda' or 'cpu').
        output_dir: Directory to save reconstructed videos.
        max_files: Maximum number of files to process (if None, process all files).
    """
    model.to(device)
    model.eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for batch_idx, (video_tensor, _file_name) in enumerate(dataloader):
        if max_files is not None and batch_idx >= max_files:
            break  # Stop processing after reaching the max number of files

        # Move to device
        video_tensor = video_tensor.to(device)
        logger.info(f"Processing batch {batch_idx}, video shape: {video_tensor.shape}")
        video_tensor = video_tensor.to(device, dtype=torch.float16)
        with torch.no_grad():
            # Encode video
            reconstructed_video = model(video_tensor, return_dict=False, return_posterior=True, sample_posterior=False)[0]
            #posterior = model(video_tensor, return_dict=False, return_posterior=True, sample_posterior=True)[1]

            #posterior = model.encode(video_tensor).latent_dist
            #latents = posterior.sample()  # [B, latent_channels, T//t_ratio, H//s_ratio, W//s_ratio]

            ## Decode video
            #reconstructed_video = model.decode(latents).sample  # [B, C, T, H, W]
        # Save the reconstructed video with the same name as the input
        output_path = os.path.join(output_dir, f"{_file_name[0]}.pt")

        reconstructed_video = reconstructed_video.cpu().float()
        
        torch.save(reconstructed_video.cpu(), output_path)
        logger.info(f"Saved reconstructed video to {output_path}, shape is {reconstructed_video.shape}")
        if mp4:
            save_path = os.path.join(output_dir, f"{_file_name[0]}.mp4")
            save_videos_grid(reconstructed_video, save_path, fps=24, rescale=True)
            logger.info(f'Sample save to: {save_path}')

        
# 保存模型架构到文件
#save_model_architecture_to_file(vae, "vae_full_architecture.txt")
device = "cuda" if torch.cuda.is_available() else "cpu"
vae, _, s_ratio, t_ratio = load_vae(
    vae_type="884-16c-hy",
    vae_precision="fp16",
    logger=logger,
    vae_path=f"ckpts/hunyuan-video-t2v-720p/vae",
    device=device,
    t_ops_config_path="t_ops_config.json",
    test=True
)
vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}
vae.enable_tiling()
dataloader = get_single_batch_dataloader(
    tensor_dir="video_data/video_tensor",
    shuffle=False,
    num_workers=0
)
output_directory = "video_data/vae_interpolate_reconstructed_videos"  # 重建视频保存路径
infer_vae(vae, dataloader, device, output_directory, max_files=2, mp4=True)
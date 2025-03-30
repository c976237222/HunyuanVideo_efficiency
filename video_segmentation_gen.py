import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime
import torch

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.modules import load_model
from hyvideo.vae import load_vae
from hyvideo.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from hyvideo.diffusion.pipelines import HunyuanVideoPipeline
from hyvideo.vae.vae import DiagonalGaussianDistribution
from hyvideo.vae.adaptive_temporal_tiling import AdaptiveTemporalTiling

# hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)
VAE_PATH="ckpts/hunyuan-video-t2v-720p/vae"

t_ops_config_path="t_ops_config.json"

vae, _, s_ratio, t_ratio = load_vae(
            vae_path=VAE_PATH,
            vae_precision="fp16",
            logger=logger,
            device=device,
            t_ops_config_path=t_ops_config_path,
        )
vae.to(device)
vae.eval()
vae.enable_tiling()
model=vae
vae.tile_overlap_factor = 0
from torch.utils.data import DataLoader
from dataset_processor.dataset_loader import VideoTensorDataset

tensor_dir="/mnt/public/wangsiyuan/HunyuanVideo_efficiency/video_data/15hz_240p_tensors"
FPS=30
dataset = VideoTensorDataset(tensor_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

adaptor = AdaptiveTemporalTiling(
        vae_ckpt_path="ckpts/hunyuan-video-t2v-720p/vae",
        device=device,
        vae_precision="fp16",
        fps=FPS
    )

max_files = 22
output_dir = '/mnt/public/wangsiyuan/HunyuanVideo_efficiency/analysis/15hz_240p_reconstructed_nothing_4x'
label_dir = f"{output_dir}_label"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)
CHUNK_SIZE = 65
  
def split_tensor(tensor: torch.Tensor, chunk_size: int) -> list:
    """
    将视频张量按时间维度切分为多个固定帧数的片段。

    参数:
        tensor (torch.Tensor): 输入视频张量，形状为 [B, C, T, H, W]。
        chunk_size (int): 每个片段的帧数。

    返回:
        List[torch.Tensor]: 切分后的片段列表，形状为 [B, C, chunk_size, H, W]。
    """
    B, C, T, H, W = tensor.shape
    num_chunks = T // chunk_size
    if num_chunks == 0:
        return []
    chunks = torch.split(tensor[:, :, :num_chunks * chunk_size, :, :], chunk_size, dim=2)
    return list(chunks)

for batch_idx, (video_tensor, file_name) in enumerate(dataloader):
    if max_files is not None and batch_idx >= max_files:
        break

    # 去掉 .pt 后缀
    file_name = file_name[0].replace(".pt", "")

    # Move to device
    video_tensor = video_tensor.to(device, dtype=torch.float16)
    logger.info(f"Processing {file_name}, video shape: {video_tensor.shape}")
    input_list = split_tensor(video_tensor, CHUNK_SIZE)
    for chunk_idx, chunk in enumerate(input_list):
        with torch.no_grad():
            # =========== Encode阶段 ===========
            posterior_out = model.temporal_tiled_encode(
                x=chunk,
                adaptor=adaptor,  # 传入我们自适应类
                return_dict=True
            )
            # posterior_out 是 AutoencoderKLOutput
            posterior = posterior_out.latent_dist  # 里面包含 mean/var
            tiles_ci = posterior_out.tiles_ci
            # =========== Decode阶段 ===========
            #for tile_ind, (cur_tile, cur_ratio) in enumerate(row):
            #tile_ci = tiles_ci[tile_ind]
            #posterior = DiagonalGaussianDistribution(cur_tile)
            reconstructed_chunk = model.temporal_tiled_decode(
                z=posterior.mode(),  # 或 sample() 取随机
                adaptor=adaptor,     # 同样传入
                return_dict=True
            ).sample
        reconstructed_chunk = reconstructed_chunk.cpu().float()
        label_chunk = chunk.cpu().float()
        logger.info(f"Re: {reconstructed_chunk.shape}, Label: {label_chunk.shape}")
        save_path = os.path.join(output_dir, f"{file_name}|{chunk_idx}|{tiles_ci[0]}|.mp4")
        label_path = os.path.join(label_dir, f"{file_name}|{chunk_idx}|{tiles_ci[0]}|.mp4")
        save_videos_grid(reconstructed_chunk, save_path, fps=FPS, rescale=True)
        save_videos_grid(label_chunk, label_path, fps=FPS, rescale=True)
        logger.info(f'Sample saved to: {save_path}')
print("done")
from hyvideo.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from hyvideo.vae import load_vae
from loguru import logger
import torch
from torch.nn import Module
from dataset_processor.dataset_loader import VideoTensorDataset
from torch.utils.data import DataLoader
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
    """
    model.to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    for batch_idx, (video_tensor, file_name) in enumerate(dataloader):  # ✅ 现在获取文件名
        if max_files is not None and batch_idx >= max_files:
            break  # Stop processing after reaching the max number of files

        # 获取原始文件名（去掉 .pt 后缀）
        file_name = file_name[0].replace(".pt", "")

        # Move to device
        video_tensor = video_tensor.to(device, dtype=torch.float16)
        logger.info(f"Processing {file_name}, video shape: {video_tensor.shape}")

        with torch.no_grad():
            # Encode and decode video
            reconstructed_video = model(video_tensor, return_dict=False, return_posterior=True, sample_posterior=False)[0]

        # Save the reconstructed video
        reconstructed_video = reconstructed_video.cpu().float()
        output_path = os.path.join(output_dir, f"{file_name}.pt")  # ✅ 使用原始文件名
        torch.save(reconstructed_video, output_path)
        logger.info(f"Saved reconstructed video to {output_path}, shape: {reconstructed_video.shape}")

        if mp4:
            save_path = os.path.join(output_dir, f"{file_name}.mp4")  # ✅ 使用原始文件名
            save_videos_grid(reconstructed_video, save_path, fps=24, rescale=True)
            logger.info(f'Sample saved to: {save_path}')


# ✅ 加载 VAE 模型
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
#vae.enable_tiling()

# ✅ 加载数据集
tensor_dir = "video_data/video_data_100_240p_tensor"  # 你的 Tensor 目录
dataset = VideoTensorDataset(tensor_dir)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

# ✅ 运行推理
output_directory = "video_data/vae_output_videos"
infer_vae(vae, dataloader, device, output_directory, max_files=10, mp4=True)

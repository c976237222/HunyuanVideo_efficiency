# demo_dynamic.py
from dlfr_vae import AutoencoderKLCausal3D
from loguru import logger
import torch
from pathlib import Path

# 假设本地目录中存放有 config.json 和 pytorch_model.pt，确保它们匹配！
# 如果你的 config.json 与权重文件在同一目录（例如 "ckpts"），则：
vae_path = "/home/hanling/HunyuanVideo_efficiency/ckpts/hunyuan-video-t2v-720p/vae"

# 使用 AutoencoderKLCausal3D.load_config 加载配置（本地加载）
config = AutoencoderKLCausal3D.load_config(vae_path)

# 通过 from_config 创建 DynamicVAE 实例（内部构造将根据 config.json 来）
vae = AutoencoderKLCausal3D.from_config(config)

vae_ckpt = Path(vae_path) / "pytorch_model.pt"
assert vae_ckpt.exists(), f"VAE checkpoint not found: {vae_ckpt}"
ckpt = torch.load(vae_ckpt, map_location="cuda", weights_only=False)
if "state_dict" in ckpt:
    ckpt = ckpt["state_dict"]

if any(k.startswith("vae.") for k in ckpt.keys()):
    ckpt = {k.replace("vae.", ""): v for k, v in ckpt.items() if k.startswith("vae.")}
vae.load_state_dict(ckpt)

vae = vae.to("cuda", dtype=torch.float16)
vae.requires_grad_(False)
vae.eval()

# 构造一个随机输入 (B, C, T, H, W)
tensor_random = torch.randn((1, 16, 30, 10, 10)).to("cuda", dtype=torch.float16)
ratio_list = [4,8,16] #read from dlfr_config
decoded_out = vae.temporal_tiled_decode(
    z=tensor_random, 
    ratio_list=ratio_list,
    return_dict=True
).sample

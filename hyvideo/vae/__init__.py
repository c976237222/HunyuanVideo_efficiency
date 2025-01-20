from pathlib import Path

import torch

from .autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from ..constants import VAE_PATH, PRECISION_TO_TYPE

from pathlib import Path
from loguru import logger
import torch
import json
from .autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from ..constants import VAE_PATH, PRECISION_TO_TYPE

def _apply_t_ops_config_to_vae(vae: AutoencoderKLCausal3D, t_ops_config: dict):
    """
    将 t_ops_config 注入到 vae.encoder.down_blocks / vae.encoder.mid_block / vae.decoder.up_blocks / vae.decoder.mid_block.
    """
    # -------------- 1) encoder --------------
    enc_cfg = t_ops_config.get("encoder", {})

    # (1a) 遍历 down_blocks
    down_blocks_cfg = enc_cfg.get("down_blocks", [])
    for block_cfg in down_blocks_cfg:
        idx = block_cfg["block_index"]
        if 0 <= idx < len(vae.encoder.down_blocks):
            down_block = vae.encoder.down_blocks[idx]
            if hasattr(down_block, "apply_t_ops_config"):
                down_block.apply_t_ops_config(block_cfg)
            else:
                print(f"[Warning] down_block at index {idx} lacks apply_t_ops_config().")
        else:
            print(f"[Warning] down_block index {idx} out of range of encoder.down_blocks.")

    # (1b) encoder.mid_block
    enc_mid_cfg = enc_cfg.get("mid_block", {})  # <=== 新增
    if hasattr(vae.encoder, "mid_block") and hasattr(vae.encoder.mid_block, "apply_t_ops_config_midblock"):
        vae.encoder.mid_block.apply_t_ops_config_midblock(enc_mid_cfg)
    else:
        print("[Warning] encoder.mid_block not found or has no apply_t_ops_config_midblock method.")

    # -------------- 2) decoder --------------
    dec_cfg = t_ops_config.get("decoder", {})

    # (2a) 遍历 up_blocks
    up_blocks_cfg = dec_cfg.get("up_blocks", [])
    for block_cfg in up_blocks_cfg:
        idx = block_cfg["block_index"]
        if 0 <= idx < len(vae.decoder.up_blocks):
            up_block = vae.decoder.up_blocks[idx]
            if hasattr(up_block, "apply_t_ops_config"):
                up_block.apply_t_ops_config(block_cfg)
            else:
                print(f"[Warning] up_block at index {idx} lacks apply_t_ops_config().")
        else:
            print(f"[Warning] up_block index {idx} out of range of decoder.up_blocks.")

    # (2b) decoder.mid_block
    dec_mid_cfg = dec_cfg.get("mid_block", {})  # <=== 新增
    if hasattr(vae.decoder, "mid_block") and hasattr(vae.decoder.mid_block, "apply_t_ops_config_midblock"):
        vae.decoder.mid_block.apply_t_ops_config_midblock(dec_mid_cfg)
    else:
        print("[Warning] decoder.mid_block not found or has no apply_t_ops_config_midblock method.")


def load_t_ops_config(json_path: str) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)

def load_vae(
    vae_type: str="884-16c-hy",
    vae_precision: str=None,
    sample_size: tuple=None,
    vae_path: str=None,
    logger=None,
    device=None,
    t_ops_config_path: str = None,
    test: bool = False,
):
    """
    Load the 3D VAE model.
    """
    if vae_path is None:
        vae_path = VAE_PATH[vae_type]

    if logger is not None:
        logger.info(f"Loading 3D VAE model ({vae_type}) from: {vae_path}")
    config = AutoencoderKLCausal3D.load_config(vae_path)
    if sample_size:
        vae = AutoencoderKLCausal3D.from_config(config, sample_size=sample_size)
    else:
        vae = AutoencoderKLCausal3D.from_config(config)

    vae_ckpt = Path(vae_path) / "pytorch_model.pt"
    assert vae_ckpt.exists(), f"VAE checkpoint not found: {vae_ckpt}"

    ckpt = torch.load(vae_ckpt, map_location=vae.device, weights_only=False)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if any(k.startswith("vae.") for k in ckpt.keys()):
        ckpt = {k.replace("vae.", ""): v for k, v in ckpt.items() if k.startswith("vae.")}
    vae.load_state_dict(ckpt)

    spatial_compression_ratio = vae.config.spatial_compression_ratio
    time_compression_ratio = vae.config.time_compression_ratio

    if vae_precision is not None:
        vae = vae.to(dtype=PRECISION_TO_TYPE[vae_precision])

    vae.requires_grad_(False)

    if logger is not None:
        logger.info(f"VAE to dtype: {vae.dtype}")

    if device is not None:
        vae = vae.to(device)

    vae.eval()

    # ============ 如果传入了 t_ops_config_path，并且 test=True，就调用 _apply_t_ops_config_to_vae ============
    if t_ops_config_path is not None and test:
        my_t_ops_config = load_t_ops_config(t_ops_config_path)
        if logger is not None:
            logger.info("Applying T-pool/pad configs to the loaded VAE.")
        _apply_t_ops_config_to_vae(vae, my_t_ops_config)

    return vae, vae_path, spatial_compression_ratio, time_compression_ratio
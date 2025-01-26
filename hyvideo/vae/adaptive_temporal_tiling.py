# adaptive_temporal_tiling.py

import os
import tempfile
import subprocess
import torch
from loguru import logger

# 假设 load_vae 是您项目中已有的加载函数
# from hyvideo.vae import load_vae
from hyvideo.vae import load_vae

################################################################
# 可选的视频写盘库，这里仅作示例，您可用 imageio、moviepy 等
################################################################
try:
    import imageio.v3 as iio
    HAVE_IMAGEIO = True
except ImportError:
    HAVE_IMAGEIO = False
    logger.warning("imageio.v3 not installed, ffprobe demo code might fail to create mp4 from tensor.")


class AdaptiveTemporalTiling:
    """
    在这里加载 3 个不同配置(4x/2x/1x)的 VAE，并提供:
      1) compute_tile_bitrate(tile_tensor): 用 ffprobe 得到码率
      2) decide_compression_ratio(bitrate_value): 根据码率映射到 4/2/1
      3) get_vae_for_ratio(ratio): 返回对应 VAE
    """
    def __init__(self, 
                 vae_ckpt_path="ckpts/hunyuan-video-t2v-720p/vae", 
                 device="cuda",
                 vae_precision="fp16"):
        self.device = device
        self.vae_precision = vae_precision
        self.vae_ckpt_path = vae_ckpt_path
        
        # 针对您给出的 3 个 JSON 文件
        self.config_4x = "/home/hanling/HunyuanVideo_efficiency/analysis/config_stride2_json/exp_262.json"
        self.config_2x = "/home/hanling/HunyuanVideo_efficiency/analysis/config_stride_json/exp_20.json"
        self.config_1x = "/home/hanling/HunyuanVideo_efficiency/t_ops_config.json"

        logger.info("Loading 4x-time-compression VAE ...")
        self.vae_4x, _, _, _ = load_vae(
            vae_type="884-16c-hy",
            vae_precision=self.vae_precision,
            vae_path=self.vae_ckpt_path,
            device=self.device,
            t_ops_config_path=self.config_4x,
            test=True
        )
        
        logger.info("Loading 2x-time-compression VAE ...")
        self.vae_2x, _, _, _ = load_vae(
            vae_type="884-16c-hy",
            vae_precision=self.vae_precision,
            vae_path=self.vae_ckpt_path,
            device=self.device,
            t_ops_config_path=self.config_2x,
            test=True
        )
        
        logger.info("Loading 1x(no)-time-compression VAE ...")
        self.vae_1x, _, _, _ = load_vae(
            vae_type="884-16c-hy",
            vae_precision=self.vae_precision,
            vae_path=self.vae_ckpt_path,
            device=self.device,
            t_ops_config_path=self.config_1x,
            test=True
        )

        self.vae_4x.eval()
        self.vae_2x.eval()
        self.vae_1x.eval()

    def compute_tile_bitrate(self, tile: torch.Tensor, fps: int = 25) -> float:
        """
        将 tile (形状 [B, C, T, H, W], 其中 B=1) 写为临时 mp4，再调用 ffprobe 得到码率 (kbps)。
        实际中最好是从原视频文件截取对应时间段进行探测，或者直接对输入文件做处理，这里仅为演示。
        """
        # 检查 batch_size==1，否则需要拆开多 batch
        assert tile.shape[0] == 1, "演示代码仅支持 B=1，若有需要可自行拆分"

        if not HAVE_IMAGEIO:
            logger.error("imageio.v3 not installed, compute_tile_bitrate will return 0.0 forcibly.")
            return 0.0

        # 1) 写到临时 mp4
        tmp_dir = tempfile.mkdtemp(prefix="tile_ffprobe_")
        mp4_path = os.path.join(tmp_dir, "temp_tile.mp4")
        
        # tile: [1, C, T, H, W] => [T, H, W, C], C=3
        # 如果 C!=3 也需要做相应处理
        _, c, t, h, w = tile.shape
        if c != 3:
            logger.warning(f"compute_tile_bitrate expects tile with 3 channels, but got c={c}. We'll clamp to 3 if possible.")
        
        # 简易转换 + 写盘
        tile_np = tile[0].permute(1, 2, 3, 0).detach().cpu().numpy()  # => [T, H, W, C]
        # 如果 channels>3，需要截取前3通道
        if c > 3:
            tile_np = tile_np[..., :3]
        
        iio.imwrite(
            uri=mp4_path,      # 比如 "temp_tile.mp4"
            image=tile_np,     # numpy 数组，形状 [T, H, W, C]
            fps=fps,
            plugin="pyav",
            codec="libx264",   # 显式指定编解码器
        )

        # 2) 调用 ffprobe 获取码率
        cmd = [
            "ffprobe", "-v", "error", 
            "-show_entries", "format=bit_rate", 
            "-of", "default=noprint_wrappers=1:nokey=1",
            mp4_path
        ]
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            bit_rate_str = output.decode().strip()
            bit_rate_bps = float(bit_rate_str)  # 单位：bps
            bit_rate_kbps = bit_rate_bps / 1000.0

            logger.info(f"ffprobe bit_rate(kbps) = {bit_rate_kbps}")
        except Exception as e:
            logger.error(f"ffprobe failed: {e}")
            bit_rate_kbps = 0.0
        
        # 清理临时文件夹
        try:
            os.remove(mp4_path)
            os.rmdir(tmp_dir)
        except:
            pass
        
        return bit_rate_kbps
    
    def decide_compression_ratio(self, bitrate_kbps: float):
        """
        用码率档位做一个映射：<1000 => 4x, [1000,2000) => 2x, >=2000 => 1x
        """
        if bitrate_kbps < 1000:
            return 4
        elif bitrate_kbps < 2000:
            return 2
        else:
            return 1

    def get_vae_for_ratio(self, ratio: int):
        """
        返回对应 ratio 的 VAE 对象。
        """
        if ratio == 4:
            return self.vae_4x
        elif ratio == 2:
            return self.vae_2x
        else:
            return self.vae_1x

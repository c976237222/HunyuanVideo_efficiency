# adaptive_temporal_tiling.py

import os
import tempfile
import subprocess
import torch
from loguru import logger
import numpy as np
from skimage.metrics import structural_similarity as ssim_metric
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import math
# 假设 load_vae 是您项目中已有的加载函数
# from hyvideo.vae import load_vae
from hyvideo.vae import load_vae
from pytorch_msssim import ssim as calc_ssim_func
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
                 vae_precision="fp16",
                 fps=None):
        self.device = device
        self.vae_precision = vae_precision
        self.vae_ckpt_path = vae_ckpt_path
        self.fps = fps
        
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
        
    def compute_tile_entropy_interframe(self, video_tensor: torch.Tensor) -> float:
        """
        计算视频张量的帧间差异熵（适配输入范围为 [-1, 1] 的情况）
        """
        video_tensor = video_tensor[0]  # => [C, T, H, W]
        c, t, h, w = video_tensor.shape

        if c > 3:
            video_tensor = video_tensor[:3]

        video_np = video_tensor.permute(1, 2, 3, 0).detach().cpu().numpy()  # => [T, H, W, C]
        
        diff_entropies = []
        for i in range(t - 1):
            frame_a = video_np[i]      # [H, W, C]
            frame_b = video_np[i + 1]  # [H, W, C]

            # ------------ 关键修改点 ------------
            # 将 [-1, 1] 的输入映射到 [0, 255] 的 uint8
            frame_a = (frame_a + 1.0) * 0.5 * 255.0  # 映射到 [0, 255]
            frame_a = np.clip(frame_a, 0, 255).astype(np.uint8)
            frame_b = (frame_b + 1.0) * 0.5 * 255.0
            frame_b = np.clip(frame_b, 0, 255).astype(np.uint8)

            # 转灰度（原逻辑保持）
            if frame_a.shape[-1] == 3:
                gray_a = 0.299 * frame_a[..., 0] + 0.587 * frame_a[..., 1] + 0.114 * frame_a[..., 2]
                gray_b = 0.299 * frame_b[..., 0] + 0.587 * frame_b[..., 1] + 0.114 * frame_b[..., 2]
            else:
                gray_a = frame_a[..., 0]
                gray_b = frame_b[..., 0]

            # 计算差分（uint8 范围 0~255，无需再 *255）
            diff = np.abs(gray_b - gray_a)  # 结果自动在 0~255 范围内

            # 统计直方图
            hist, _ = np.histogram(diff, bins=256, range=(0, 255))
            p = hist / hist.sum()
            p = p[p > 0]
            entropy = -np.sum(p * np.log2(p))

            diff_entropies.append(entropy)

        return float(np.mean(diff_entropies)) if diff_entropies else 0.0

    def compute_tile_psnr(self, tile: torch.Tensor) -> float:

        if tile.dim() != 5:
            raise ValueError(f"Expected tile with 5 dimensions [B, C, T, H, W], but got {tile.dim()} dimensions.")

        tile = tile[0]  # => [C, T, H, W]
        c, t, h, w = tile.shape
        if c > 3:
            tile = tile[:3]
            c = 3  # 更新通道数

        # 将 tile 从 [-1, 1] 映射到 [0, 255]
        tile_np = ((tile.permute(1, 2, 3, 0).detach().cpu().numpy() + 1.0) * 127.5).astype(np.float32)  # => [T, H, W, C]

        psnr_list = []
        for i in range(t - 1):
            frame_a = tile_np[i]  # [H, W, C]
            frame_b = tile_np[i + 1]  # [H, W, C]
            psnr_val = compare_psnr(frame_a, frame_b, data_range=255)
            psnr_list.append(psnr_val)

        if not psnr_list:
            return 100.0

        return float(np.mean(psnr_list))
    
    def compute_tile_ssim(self, tile: torch.Tensor) -> float:
        """
        计算视频的 SSIM 值。
        
        参数:
            videos1 (torch.Tensor): 视频张量，形状为 [batch_size, T, C, H, W]
        
        返回:
            List[float]: 每个视频的 SSIM 值
        """
        ci_ssim_results = []
        tile = tile[0]
        tile = (tile + 1) * 127.5  # 先映射到 [0, 255]
        tile = tile.clamp(0, 255)
        img1 = tile[:-1]  # [T-1, C, H, W]
        img2 = tile[1:]   # [T-1, C, H, W]
        # 计算 SSIM，保持 batch_size=1
        ssim_val = calc_ssim_func(img1, img2, data_range=255, size_average=True)
        return ssim_val.item()

    def compute_tile_ssim_1(self, tile: torch.Tensor) -> float: #弃用
        """
        计算 tile 中相邻帧的平均 SSIM.
        tile shape: [B, C, T, H, W], 默认 B=1, C=3.
        返回: 0~1之间的 float, 越大表示帧间越相似.
        """
        tile = tile[0]  # => [C, T, H, W]
        c, t, h, w = tile.shape
        # 若有多于3通道，可根据需要处理，这里先截取前3通道
        if c > 3:
            tile = tile[:3]
            c = 3

        # 转成 numpy: [T, H, W, C]
        tile_np = tile.permute(1, 2, 3, 0).detach().cpu().numpy()  # => [T,H,W,C]

        ssim_values = []
        for i in range(t - 1):
            frame_a = tile_np[i]
            frame_b = tile_np[i + 1]
            # skimage ssim: 如果是多通道彩色图，需要 multichannel=True
            ssim_val = ssim_metric(
                frame_a, 
                frame_b, 
                channel_axis=-1,
                data_range=frame_b.max() - frame_b.min()
            )
            ssim_values.append(ssim_val)

        if not ssim_values:
            return 1.0  # 只有1帧或0帧的情况
        
        return float(np.mean(ssim_values))

    def compute_tile_bitrate(self, tile: torch.Tensor) -> float:
        """
        将 tile (形状 [B, C, T, H, W], 其中 B=1) 写为临时 mp4，再调用 ffprobe 得到码率 (kbps)。
        实际中最好是从原视频文件截取对应时间段进行探测，或者直接对输入文件做处理，这里仅为演示。
        """
        # 检查 batch_size==1，否则需要拆开多 batch
        assert tile.shape[0] == 1, "演示代码仅支持 B=1，若有需要可自行拆分"
        tile_np = ((tile_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
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
            fps=self.fps,
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
    
    def decide_compression_ratio_bitrate_15fps(self, bitrate_kbps: float):
        """
        用码率档位做一个映射：<1000 => 4x, [1000,2000) => 2x, >=2000 => 1x
        """
        if 0 <= bitrate_kbps < 500.0:
            return 4
        elif 500.0 <= bitrate_kbps < 900.0:
            return 4
        else:
            return 4
        
    def decide_compression_ratio_ssim(self, ssim: float):
        """
        用码率档位做一个映射：<1000 => 4x, [1000,2000) => 2x, >=2000 => 1x
        """
        if ssim >= 0.75:
            return 1
        elif 0.75 > ssim > 0.4:
            return 1
        else:
            return 1
        
    def decide_compression_ratio_psnr(self, psnr: float):
        """
        用码率档位做一个映射：<1000 => 4x, [1000,2000) => 2x, >=2000 => 1x
        """
        if psnr > 22.0:
            return 4
        elif 22.0 >= psnr > 17.0:
            return 4
        else:
            return 4
        
    def decide_compression_ratio_entropy_interframe(self, result: float):
        """
        用码率档位做一个映射：<1000 => 4x, [1000,2000) => 2x, >=2000 => 1x
        """
        if result > 4.5:
            return 4
        elif 4.5 >= result >= 3:
            return 4
        else:
            return 4

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

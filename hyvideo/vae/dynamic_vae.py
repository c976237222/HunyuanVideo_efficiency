# autoencoder_kl_causal_3d.py

import subprocess
import tempfile
import torch
import torch.nn.functional as F
from loguru import logger
from .autoencoder_kl_causal_3d import AutoencoderKLCausal3D, AutoencoderKLOutput, DiagonalGaussianDistribution

class AutoencoderKLCausal3DAdaptive(AutoencoderKLCausal3D):
    """
    一个示例子类, 展示:
    1) temporal_tiled_encode 阶段: 
       - 对每个 tile 用 ffprobe 计算码率 -> 选定配置(4x/2x/1x)
       - load 对应 JSON -> 做 encode + quant_conv
       - 将该 tile 的配置名记录进 self._tile_config_history 里
    2) temporal_tiled_decode 阶段:
       - 不再判码率, 直接读 self._tile_config_history[i] -> load 同样 JSON -> decode
       - 保证与 encode 完全一致
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 假设三档的配置信息(示例路径):
        self.json_config_4x = "/home/hanling/HunyuanVideo_efficiency/analysis/config_stride2_json/exp_262.json"  # 4倍
        self.json_config_2x = "/home/hanling/HunyuanVideo_efficiency/analysis/config_stride_json/exp_20.json"    # 2倍
        self.json_config_1x = "/path/to/your/no_compress_config.json"  # 不压缩(1×), 需要你自己准备对应 JSON

        # 这里用一个 list 来记录每个 tile 用的 config 名称
        self._tile_config_history = []  # 在一次 encode/decode 流程中，每个 tile 的配置依序放这里

    ######################################
    #     1) 用 ffprobe 判码率 (encode)   #
    ######################################
    def compute_tile_info_by_ffprobe(self, tile_tensor: torch.Tensor) -> float:
        """
        (示例) 用 ffprobe 计算 tile 对应的平均码率 (单位可以是kbps).
        实际实现需要先把 tile_tensor 转成视频文件, 然后 ffprobe 分析它.
        """
        with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp_video_file:
            tmp_name = tmp_video_file.name
            # TODO: 把 tile_tensor 存成 mp4
            # save_tensor_to_mp4(tile_tensor, tmp_name, fps=10)  # 你自己实现

            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_entries", "format=bit_rate",
                tmp_name
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                # TODO: 解析 result.stdout, 获取平均码率
                average_bitrate = 1500.0  # 示例: 这里写死 1500, 实际上要 parse JSON
            except subprocess.CalledProcessError as e:
                logger.error(f"ffprobe error: {e.stderr}")
                average_bitrate = 0.0
        return average_bitrate

    # 可以另外留空两个方法, 以方便后续扩展
    def compute_tile_info_method2(self, tile_tensor: torch.Tensor) -> float:
        pass

    def compute_tile_info_method3(self, tile_tensor: torch.Tensor) -> float:
        pass

    ################################################
    # 2) 动态加载/应用 新的 t_ops_config JSON (通用) #
    ################################################
    def load_new_t_ops_config(self, json_path: str):
        from ..vae import load_t_ops_config, _apply_t_ops_config_to_vae

        logger.info(f"[Adaptive] 切换 t_ops_config: {json_path}")
        new_cfg = load_t_ops_config(json_path)
        _apply_t_ops_config_to_vae(self, new_cfg)

    ########################################################
    # 3) temporal_tiled_encode: encode 阶段记录 tile 配置  #
    ########################################################
    def temporal_tiled_encode(self, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
        """
        沿时间维度分块:
          - 对每个 tile 用 ffprobe 判码率, 决定使用 "4x" / "2x" / "1x" 配置
          - load 对应 JSON
          - encode + quant_conv
        最后把 tile 拼接起来, 得到 moments -> posterior.
        同时把每个 tile 的配置名写进 self._tile_config_history 里.
        """

        # 先清空旧的记录
        self._tile_config_history = []

        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_tsize * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_tsize * self.tile_overlap_factor)
        t_limit = self.tile_latent_min_tsize - blend_extent

        row = []
        for i in range(0, T, overlap_size):
            tile = x[:, :, i : i + self.tile_sample_min_tsize + 1, :, :]
            logger.info(f"[Adaptive Encode] tile {i} shape: {tile.shape}")

            # 计算码率 -> 分类
            avg_bitrate = self.compute_tile_info_by_ffprobe(tile)
            if avg_bitrate < 1000:
                config_name = "4x"
                self.load_new_t_ops_config(self.json_config_4x)
            elif avg_bitrate < 2000:
                config_name = "2x"
                self.load_new_t_ops_config(self.json_config_2x)
            else:
                config_name = "1x"
                self.load_new_t_ops_config(self.json_config_1x)

            # 记录当前 tile 的 config
            self._tile_config_history.append(config_name)

            # encode
            if self.use_spatial_tiling and (tile.shape[-1] > self.tile_sample_min_size or tile.shape[-2] > self.tile_sample_min_size):
                tile = self.spatial_tiled_encode(tile, return_moments=True)
            else:
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)

            # 去重叠帧(去掉与上个 tile 重叠的 1 帧)
            if i > 0:
                tile = tile[:, :, 1:, :, :]

            row.append(tile)

        # blend + 拼接
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
                result_row.append(tile[:, :, :t_limit, :, :])
            else:
                result_row.append(tile[:, :, : t_limit + 1, :, :])

        moments = torch.cat(result_row, dim=2)
        posterior = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    #######################################################
    # 4) temporal_tiled_decode: decode 时直接读 config    #
    #######################################################
    def temporal_tiled_decode(self, z: torch.FloatTensor, return_dict: bool = True):
        """
        沿时间维度分块:
          - 对每个 tile 从 self._tile_config_history 里读出配置名
          - 加载对应 JSON
          - decode (post_quant_conv + decoder)
        不再判码率, 完全跟 encode 保持一致。
        """

        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_tsize * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_tsize * self.tile_overlap_factor)
        t_limit = self.tile_sample_min_tsize - blend_extent

        row = []
        # 这里假设 encode 和 decode 的分块大小是一致的, 并且 tile 个数也一样。
        # self._tile_config_history 的长度应 == 分块数
        tile_count = 0

        for i in range(0, T, overlap_size):
            tile = z[:, :, i : i + self.tile_latent_min_tsize + 1, :, :]
            logger.info(f"[Adaptive Decode] tile {i} shape: {tile.shape}")

            # 从记录里读取配置名
            if tile_count < len(self._tile_config_history):
                config_name = self._tile_config_history[tile_count]
            else:
                # 如果因为某些原因 tile_count 超出范围, 这里简单处理
                logger.warning(f"No config recorded for tile index={tile_count}, fallback to 1x.")
                config_name = "1x"

            # 根据 config_name load JSON
            if config_name == "4x":
                self.load_new_t_ops_config(self.json_config_4x)
            elif config_name == "2x":
                self.load_new_t_ops_config(self.json_config_2x)
            else:
                self.load_new_t_ops_config(self.json_config_1x)

            # decode
            if self.use_spatial_tiling and (tile.shape[-1] > self.tile_latent_min_size or tile.shape[-2] > self.tile_latent_min_size):
                decoded = self.spatial_tiled_decode(tile, return_dict=True).sample
            else:
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile).sample

            # 去重叠帧
            if i > 0:
                decoded = decoded[:, :, 1:, :, :]

            row.append(decoded)
            tile_count += 1

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
                result_row.append(tile[:, :, :t_limit, :, :])
            else:
                result_row.append(tile[:, :, : t_limit + 1, :, :])

        dec = torch.cat(result_row, dim=2)

        # 返回一个与 decode 一致的结构
        from diffusers.models.modeling_outputs import BaseOutput
        class DecoderOutput2(BaseOutput):
            sample: torch.FloatTensor

        if not return_dict:
            return dec
        return DecoderOutput2(sample=dec)

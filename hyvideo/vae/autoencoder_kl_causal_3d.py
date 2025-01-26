# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified from diffusers==0.29.2
#
# ==============================================================================
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config

try:
    # This diffusers is modified and packed in the mirror.
    from diffusers.loaders import FromOriginalVAEMixin
except ImportError:
    # Use this to be compatible with the original diffusers.
    from diffusers.loaders.single_file_model import FromOriginalModelMixin as FromOriginalVAEMixin
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.attention_processor import (
    ADDED_KV_ATTENTION_PROCESSORS,
    CROSS_ATTENTION_PROCESSORS,
    Attention,
    AttentionProcessor,
    AttnAddedKVProcessor,
    AttnProcessor,
)
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from .vae import DecoderCausal3D, BaseOutput, DecoderOutput, DiagonalGaussianDistribution, EncoderCausal3D


@dataclass
class DecoderOutput2(BaseOutput):
    sample: torch.FloatTensor
    posterior: Optional[DiagonalGaussianDistribution] = None


class AutoencoderKLCausal3D(ModelMixin, ConfigMixin, FromOriginalVAEMixin):
    r"""
    A VAE model with KL loss for encoding images/videos into latents and decoding latent representations into images/videos.

    This model inherits from [ModelMixin]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlockCausal3D",),
        up_block_types: Tuple[str] = ("UpDecoderBlockCausal3D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        sample_tsize: int = 64,
        scaling_factor: float = 0.18215,
        force_upcast: float = True,
        spatial_compression_ratio: int = 8,
        time_compression_ratio: int = 4,
        mid_block_add_attention: bool = True,
    ):
        super().__init__()

        self.time_compression_ratio = time_compression_ratio
        self._used_tile_ratios_encode = []
        self.encoder = EncoderCausal3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True, #如果为True，将z的维度扩大一倍,2*latent_channels
            time_compression_ratio=time_compression_ratio,
            spatial_compression_ratio=spatial_compression_ratio,
            mid_block_add_attention=mid_block_add_attention,
        )

        self.decoder = DecoderCausal3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            time_compression_ratio=time_compression_ratio,
            spatial_compression_ratio=spatial_compression_ratio,
            mid_block_add_attention=mid_block_add_attention,
        )
        #只是简单的线性变化
        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)

        self.use_slicing = False
        self.use_spatial_tiling = False
        self.use_temporal_tiling = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_tsize = sample_tsize
        self.tile_latent_min_tsize = sample_tsize // time_compression_ratio

        self.tile_sample_min_size = self.config.sample_size
        sample_size = (
            self.config.sample_size[0]
            if isinstance(self.config.sample_size, (list, tuple))
            else self.config.sample_size
        )
        self.tile_latent_min_size = int(sample_size / (2 ** (len(self.config.block_out_channels) - 1)))#是指tile经过encode以后的大小
        self.tile_overlap_factor = 0.25

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (EncoderCausal3D, DecoderCausal3D)):
            module.gradient_checkpointing = value

    def enable_temporal_tiling(self, use_tiling: bool = True):
        self.use_temporal_tiling = use_tiling

    def disable_temporal_tiling(self):
        self.enable_temporal_tiling(False)

    def enable_spatial_tiling(self, use_tiling: bool = True):
        self.use_spatial_tiling = use_tiling

    def disable_spatial_tiling(self):
        self.enable_spatial_tiling(False)

    def enable_tiling(self, use_tiling: bool = True):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger videos.
        """
        self.enable_spatial_tiling(use_tiling)
        self.enable_temporal_tiling(use_tiling)

    def disable_tiling(self):
        r"""
        Disable tiled VAE decoding. If enable_tiling was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.disable_spatial_tiling()
        self.disable_temporal_tiling()
    
    def enable_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self):
        r"""
        Disable sliced VAE decoding. If enable_slicing was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    @property
    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            dict of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]], _remove_lora=False
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (dict of AttentionProcessor or only AttentionProcessor):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** Attention layers.

                If processor is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor, _remove_lora=_remove_lora)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"), _remove_lora=_remove_lora)

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in ADDED_KV_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnAddedKVProcessor()
        elif all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call set_default_attn_processor when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor, _remove_lora=True)

    @apply_forward_hook
    def encode(
        self, x: torch.FloatTensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of images/videos into latents.

        Args:
            x (torch.FloatTensor): Input batch of images/videos.
            return_dict (bool, *optional*, defaults to True):
                Whether to return a [~models.autoencoder_kl.AutoencoderKLOutput] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images/videos. If return_dict is True, a
                [~models.autoencoder_kl.AutoencoderKLOutput] is returned, otherwise a plain tuple is returned.
        """
        assert len(x.shape) == 5, "The input tensor should have 5 dimensions."
        # (B, C, T, H ,W) 

        if self.use_temporal_tiling and x.shape[2] > self.tile_sample_min_tsize: #时间压缩中可以包含空间压缩
            return self.temporal_tiled_encode(x, return_dict=return_dict) #只返回分布 不返回数值

        if self.use_spatial_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
            return self.spatial_tiled_encode(x, return_dict=return_dict) #只返回分布 不返回数值
        
        if self.use_slicing and x.shape[0] > 1:#为了节省内存，将输入切片
            encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self.encoder(x)

        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        assert len(z.shape) == 5, "The input tensor should have 5 dimensions."

        if self.use_temporal_tiling and z.shape[2] > self.tile_latent_min_tsize:
            return self.temporal_tiled_decode(z, return_dict=return_dict)

        if self.use_spatial_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            return self.spatial_tiled_decode(z, return_dict=return_dict)

        z = self.post_quant_conv(z) #post_quant_conv是一个1x1的卷积,直线线性变换
        dec = self.decoder(z) #decoder是一个解码器，将z解码成图片 此时channel是3

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
    #实际应用的decode
    @apply_forward_hook
    def decode(
        self, z: torch.FloatTensor, return_dict: bool = True, generator=None
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        """
        Decode a batch of images/videos.

        Args:
            z (torch.FloatTensor): Input batch of latent vectors.
            return_dict (bool, *optional*, defaults to True):
                Whether to return a [~models.vae.DecoderOutput] instead of a plain tuple.

        Returns:
            [~models.vae.DecoderOutput] or tuple:
                If return_dict is True, a [~models.vae.DecoderOutput] is returned, otherwise a plain tuple is
                returned.

        """
        if self.use_slicing and z.shape[0] > 1:#这段是为了节省内存，将输入切片
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)
    #上下混合 #B, C, T, H, W
    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (y / blend_extent)
        return b
    #左右混合
    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (x / blend_extent)
        return b

    def _blend_t_partial(
        self,
        a: torch.Tensor,  # [B, C, T_a, H, W]
        b: torch.Tensor,  # [B, C, T_b, H, W]
        blend_extent_a: int,
        blend_extent_b: int,
    ) -> torch.Tensor:
        """
        在时间维度上对 a、b 做融合，分为三种情况:
        1) T_a > T_b: 对 a 的末 blend_extent_a 帧做 downsample 到 T_b，再融合
        2) T_a = T_b: 直接融合
        3) T_a < T_b: 对 a 的末 blend_extent_a 帧做 upsample 到 T_b，再融合

        其中, "downsample" 采用 area 插值, "upsample" 可采用 nearest/linear/bilinear 等.
        您也可根据需要自行选择插值模式.

        步骤:
        (1) a_seg = a[:, :, (T_a - blend_extent_a) : T_a, ...]
            b_seg = b[:, :, :blend_extent_b, ...]
        (2) 若 blend_extent_a 与 blend_extent_b 不同:
            - 下采样 (area) 或上采样 (nearest 或 linear 等)
            - 获得 a_seg_res 形状 [B, C, blend_extent_b, H, W]
        (3) 对 a_seg_res, b_seg 逐帧线性混合
            for x in range(blend_extent_b):
                alpha = x / blend_extent_b
                b_seg[:, :, x, ...] = a_seg_res[:, :, x, ...]*(1-alpha) + b_seg[:, :, x, ...]*alpha
        (4) 将融合后的 b_seg 写回 b 的前 blend_extent_b 帧.
        (5) 返回 b

        说明:
        - 只改动 b 的 overlap 区域, a 不动.
        - 如果要同时更新 a, 可在混合时一并改 a_seg_res.
        - "不要有重叠pooling/插值操作" 即对 a_seg 整段一次插值到与 b_seg 同样帧数.
        """

        # 1) 截取 a末 blend_extent_a、b前 blend_extent_b
        T_a = a.shape[2]
        T_b = b.shape[2]

        if blend_extent_a <= 0 or blend_extent_b <= 0:
            # 无可混合
            return b

        a_seg = a[:, :, T_a - blend_extent_a : T_a, :, :]  # a的末blend_extent_a帧
        b_seg = b[:, :, :blend_extent_b, :, :]             # b的前blend_extent_b帧

        # 2) 都是对齐前一段的末尾
        if blend_extent_a > blend_extent_b: 
            # 对 a_seg 做 downsample, 缩到 blend_extent_b
            scale_factor = int(blend_extent_a / blend_extent_b)
            a_seg_res = F.avg_pool3d(
                a_seg,
                kernel_size=(scale_factor, 1, 1),
                stride=(scale_factor, 1, 1)
            )
        elif blend_extent_a < blend_extent_b:
            # 对 a_seg 做 upsample, 拉到 blend_extent_b
            scale_factor = int(blend_extent_b / blend_extent_a)
            a_seg_res = F.interpolate(
                a_seg,
                scale_factor=(scale_factor, 1, 1),
                mode="nearest"  # or 'linear', 看您喜好
            )
        else:
            # blend_extent_a == blend_extent_b, 直接用 a_seg
            a_seg_res = a_seg

        # 此时 a_seg_res.shape[2] == blend_extent_b
        # b_seg.shape[2] == blend_extent_b
        # 方便做逐帧混合
        T_r = b_seg.shape[2]  # == blend_extent_b
        for x in range(T_r):
            alpha = x / T_r
            b_frame = b_seg[:, :, x, :, :]
            a_frame = a_seg_res[:, :, x, :, :]
            blended = a_frame * (1 - alpha) + b_frame * alpha
            b_seg[:, :, x, :, :] = blended

        # 4) 将融合后的 b_seg 写回 b
        b[:, :, :blend_extent_b, :, :] = b_seg

        return b

    def spatial_tiled_encode(self, x: torch.FloatTensor, ratio: int = None, return_dict: bool = True, return_moments: bool = False) -> AutoencoderKLOutput:
        r"""Encode a batch of images/videos using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image/videos size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (torch.FloatTensor): Input batch of images/videos.
            return_dict (bool, *optional*, defaults to True):
                Whether or not to return a [~models.autoencoder_kl.AutoencoderKLOutput] instead of a plain tuple.

        Returns:
            [~models.autoencoder_kl.AutoencoderKLOutput] or tuple:
                If return_dict is True, a [~models.autoencoder_kl.AutoencoderKLOutput] is returned, otherwise a plain
                tuple is returned.
        """

        #滑动步长
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor)) #tile_sample_min_size = 256
        #重叠区域
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        #有效区域
        row_limit = self.tile_latent_min_size - blend_extent
        # Split video into tiles and encode them separately.
        rows = []#B, C, T, H, W
        for i in range(0, x.shape[-2], overlap_size):
            row = []
            for j in range(0, x.shape[-1], overlap_size):
                tile = x[:, :, :, i: i + self.tile_sample_min_size, j: j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    #上面的tile和当前tile混合
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    #左边的tile和当前tile混合
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))

        moments = torch.cat(result_rows, dim=-2) #经过空间压缩后的小feature map
        if return_moments:
            return moments

        posterior = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def spatial_tiled_decode(self, z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Decode a batch of images/videos using a tiled decoder.

        Args:
            z (torch.FloatTensor): Input batch of latent vectors.
            return_dict (bool, *optional*, defaults to True):
                Whether or not to return a [~models.vae.DecoderOutput] instead of a plain tuple.

        Returns:
            [~models.vae.DecoderOutput] or tuple:
                If return_dict is True, a [~models.vae.DecoderOutput] is returned, otherwise a plain tuple is
                returned.
        """
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))#滑动步长
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)#重叠区域
        row_limit = self.tile_sample_min_size - blend_extent#有效区域

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[-2], overlap_size):#z的维度是B, C, T, H, W
            row = []
            for j in range(0, z.shape[-1], overlap_size):
                tile = z[:, :, :, i: i + self.tile_latent_min_size, j: j + self.tile_latent_min_size]
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                # 以下是要解决原始大小的feature map重叠的问题
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=-2)
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def _get_tile_latent_min_tsize_for_ratio(self, ratio: int) -> int:
        """
        示例: ratio=1 => 16, ratio=2 => 8, ratio=4 => 4.
        可根据您的实际网络结构情况灵活改动。
        """
        if ratio == 1:
            return 16
        elif ratio == 2:
            return 8
        elif ratio == 4:
            return 4
        else:
            raise ValueError(f"Unsupported ratio: {ratio}")

    def _compute_blend_extent(self, latent_tsize: int, overlap_factor: float) -> int:
        """
        根据该 tile 的 latent 尺寸 latent_tsize 以及 overlap_factor(如0.25) 
        来计算 blend_extent = int(latent_tsize * overlap_factor).
        """
        return int(latent_tsize * overlap_factor)

    def temporal_tiled_encode(
        self, 
        x: torch.FloatTensor, 
        adaptor=None,             # 新增：可传入 AdaptiveTemporalTiling
        return_dict: bool = True, 
        return_moments: bool = False
    ):
        """
        在原先的 temporal_tiled_encode 基础上做修改:
          - 若 adaptor 不为 None, 则对每个 tile 用 ffprobe 得到码率 -> ratio -> vae_for_tile
          - 调用 vae_for_tile.encoder or vae_for_tile.spatial_tiled_encode
          - 同时记录下 ratio
        其余步骤按原先的拼接逻辑。
        """
        B, C, T, H, W = x.shape
        #self.tile_sample_min_tsize=64, self.tile_overlap_factor=0.25,tile_latent_min_tsize=16
        overlap_size = int(self.tile_sample_min_tsize * (1 - self.tile_overlap_factor))
        #blend_extent = int(self.tile_latent_min_tsize * self.tile_overlap_factor)
        #t_limit = self.tile_latent_min_tsize - blend_extent
        
        row = []
        
        # 清空一下上一次的记录
        self._used_tile_ratios_encode = []
        
        for i in range(0, T, overlap_size):
            tile = x[:, :, i: i + self.tile_sample_min_tsize + 1, :, :]
            

            # ============ 新增部分：决定要用哪个 VAE ============
            if adaptor is not None:
                # 先算码率
                tile_bitrate = adaptor.compute_tile_bitrate(tile)
                ratio = adaptor.decide_compression_ratio(tile_bitrate)
                vae_for_tile = adaptor.get_vae_for_ratio(ratio)
                logger.info(f"[Encode] tile range=({i},{i + self.tile_sample_min_tsize + 1}), shape={tile.shape}, ratio={ratio}")
                # 记录该 tile 用到的 ratio（以便 decode 时还原）
                self._used_tile_ratios_encode.append(ratio)
            else:
                # 没有传 adaptor，就走默认（1x）
                ratio = 1
                vae_for_tile = self  # 即当前这个 VAE 本身
            # =================================================
            
            # 空间 tlie 还是走 if self.use_spatial_tiling ...
            if self.use_spatial_tiling and (tile.shape[-1] > self.tile_sample_min_size or tile.shape[-2] > self.tile_sample_min_size):
                # 要用相应的 VAE 的 spatial_tiled_encode
                tile = vae_for_tile.spatial_tiled_encode(tile, ratio, return_moments=True)
            else:
                # 常规 encode
                tile = vae_for_tile.encoder(tile)
                tile = vae_for_tile.quant_conv(tile)

            # 跟原逻辑相同：如果不是第一个 tile，则去掉第一帧
            if i > 0:
                tile = tile[:, :, 1:, :, :]
            
            row.append((tile,ratio))
        
        # 同原逻辑：把 row 里所有 tile 做 blend + cat
        result_row = []
        for i, (cur_tile, cur_ratio) in enumerate(row):
            # 获取本 tile 的latent目标长度(如 ratio=1 =>16,2=>8,4=>4)
            cur_latent_tsize = self._get_tile_latent_min_tsize_for_ratio(cur_ratio)
            cur_blend_extent = self._compute_blend_extent(cur_latent_tsize, self.tile_overlap_factor)
            cur_t_limit = cur_latent_tsize - cur_blend_extent

            # 跟前一个 tile 做部分 overlap blend
            if i > 0:
                prev_tile, prev_ratio = row[i-1]
                prev_latent_tsize = self._get_tile_latent_min_tsize_for_ratio(prev_ratio)
                prev_blend_extent = self._compute_blend_extent(prev_latent_tsize, self.tile_overlap_factor)

                # 先做 blend
                # 只混合 overlap_len = min(prev_blend_extent, cur_blend_extent, prev_tile.shape[2], cur_tile.shape[2])
                cur_tile = self._blend_t_partial(prev_tile, cur_tile, prev_blend_extent, cur_blend_extent)

                # 最后裁剪到 cur_t_limit
                clipped_tile = cur_tile[:, :, :cur_t_limit, :, :]
            else:
                # 对第0块, 不需要 blend, 只需剪到 cur_t_limit+1 (跟原逻辑对齐)
                clipped_tile = cur_tile[:, :, : (cur_t_limit + 1), :, :]

            result_row.append(clipped_tile)

        # 最终拼接
        if len(result_row) > 1:
            moments = torch.cat(result_row, dim=2)
        else:
            moments = result_row[0]


        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def temporal_tiled_decode(self, z: torch.FloatTensor, return_dict: bool = True, adaptor=None,) -> Union[DecoderOutput, torch.FloatTensor]:
        # Split z into overlapping tiles and decode them separately.

        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_tsize * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_tsize * self.tile_overlap_factor)
        t_limit = self.tile_sample_min_tsize - blend_extent
        i = 0
        row = []
        tile_index = 0
        num_tiles = len(self._used_tile_ratios_encode)
        for tile_index in range(num_tiles):
            ratio_cur = self._used_tile_ratios_encode[tile_index]
            # 若还有下一个 tile，就去取 ratio_next, 否则直接等于 ratio_cur
            if tile_index < num_tiles - 1:
                ratio_next = self._used_tile_ratios_encode[tile_index + 1]
                tile_latent_size = int(self.tile_latent_min_tsize * ((1 / ratio_cur) * (1 - self.tile_overlap_factor) + (1 / ratio_next) * self.tile_overlap_factor))
                tile = z[:, :, i : i + tile_latent_size + 1, :, :]
                start_next = int((1.0 / ratio_next) * self.tile_overlap_factor * self.tile_latent_min_tsize)
                if ratio_cur < ratio_next: #上采样
                    partial_tile = tile[:, :, start_next:, :, :]
                    scale_factor_t = int(ratio_next / ratio_cur)
                    partial_tile_up = F.interpolate(
                        partial_tile,
                        scale_factor=(scale_factor_t, 1, 1),
                        mode='nearest'
                    )
                    tile = torch.cat(
                        [tile[:, :, :start_next, :, :], partial_tile_up],
                        dim=2
                    )
                elif ratio_cur > ratio_next: #下采样
                    partial_tile = tile[:, :, start_next:, :, :]
                    scale_factor_t = int(ratio_cur / ratio_next)
                    partial_tile_down = F.avg_pool3d(
                        partial_tile,
                        kernel_size=(scale_factor_t, 1, 1),
                        stride=(scale_factor_t, 1, 1)
                    )
                    tile = torch.cat(
                        [tile[:, :, :start_next, :, :], partial_tile_down],
                        dim=2
                    )
                else:
                    pass
                logger.info(f"[Decode tile#{tile_index}] ratio_cur={ratio_cur}, ratio_next={ratio_next}, tile_size={tile.shape}")  
            else:
                ratio_next = None
                tile_latent_size = int(self.tile_latent_min_tsize * (1 / ratio_cur))
                tile = z[:, :, i : i + tile_latent_size + 1, :, :]
            
            i += int(self.tile_latent_min_tsize * (1 / ratio_cur) * (1 - self.tile_overlap_factor))
            
            if adaptor is not None and tile_index < len(self._used_tile_ratios_encode):
                ratio = self._used_tile_ratios_encode[tile_index]
                vae_for_tile = adaptor.get_vae_for_ratio(ratio)
            else:
                ratio = 1
                vae_for_tile = self
            
            
            
            # 走 if self.use_spatial_tiling ...
            if self.use_spatial_tiling and (tile.shape[-1] > self.tile_latent_min_size or tile.shape[-2] > self.tile_latent_min_size):
                decoded = vae_for_tile.spatial_tiled_decode(tile, return_dict=True).sample
            else:
                tile = vae_for_tile.post_quant_conv(tile)
                decoded = vae_for_tile.decoder(tile)
            
            if tile_index > 0:
                decoded = decoded[:, :, 1:, :, :]
            
            row.append(decoded)
            tile_index += 1

        # 同样做 blend + cat
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                logger.info(f"[Decode] tile#{i}, shape0={row[i - 1].shape}, shape1={tile.shape}")
                tile = self.blend_t(row[i - 1], tile, blend_extent)
            result_row.append(tile[:, :, :t_limit, :, :] if i>0 else tile[:, :, :t_limit+1, :, :])
        
        dec = torch.cat(result_row, dim=2)
        
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        return_posterior: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput2, torch.FloatTensor]:
        r"""
        Args:
            sample (torch.FloatTensor): Input sample.
            sample_posterior (bool, *optional*, defaults to False):
                Whether to sample from the posterior.
            return_dict (bool, *optional*, defaults to True):
                Whether or not to return a [DecoderOutput] instead of a plain tuple.
        """
        x = sample
        # latent_dist=posterior是一个分布
        # encode输出的是一个分布,其中均值维度是B, C, T, H, W,其中C是latent_channels,方差也是这个维度. 
        # 这就代表vae学习到的是latent_channels纬度的高斯分布
        posterior = self.encode(x).latent_dist 
        if sample_posterior:
            z = posterior.sample(generator=generator) #z的形状是B, C, T, H, W,其中C具体数值是latent_channels=4
        else:
            z = posterior.mode()
        dec = self.decode(z).sample #这时候channel是3

        if not return_dict:
            if return_posterior:
                return (dec, posterior)
            else:
                return (dec,)
        if return_posterior:
            return DecoderOutput2(sample=dec, posterior=posterior)
        else:
            return DecoderOutput2(sample=dec)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query,
        key, value) are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("fuse_qkv_projections() is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

    # Copied from diffusers.models.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)
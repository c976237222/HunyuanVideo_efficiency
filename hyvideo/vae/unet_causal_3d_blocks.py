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

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from diffusers.utils import logging
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import SpatialNorm
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import AdaGroupNorm
from diffusers.models.normalization import RMSNorm
from loguru import logger as _logger
import sys

logger_ = logging.get_logger(__name__)  # pylint: disable=invalid-name

def prepare_causal_attention_mask(n_frame: int, n_hw: int, dtype, device, batch_size: int = None):
    seq_len = n_frame * n_hw
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    for i in range(seq_len):
        i_frame = i // n_hw
        mask[i, : (i_frame + 1) * n_hw] = 0
    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class CausalConv3d(nn.Module):
    """
    Implements a causal 3D convolution layer where each position only depends on previous timesteps and current spatial locations.
    This maintains temporal causality in video generation tasks.
    """

    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,#不开启膨胀卷积
        pad_mode='replicate',
        **kwargs
    ):
        super().__init__()

        self.pad_mode = pad_mode
        padding = (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size - 1, 0)  # W, H, T
        self.time_causal_padding = padding

        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)


class UpsampleCausal3D(nn.Module):
    """
    A 3D upsampling layer with an optional convolution.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size: Optional[int] = None,
        padding=1,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        interpolate=True,
        upsample_factor=(2, 2, 2),
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate
        self.upsample_factor = upsample_factor

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        conv = None
        if use_conv_transpose:
            raise NotImplementedError
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = CausalConv3d(self.channels, self.out_channels, kernel_size=kernel_size, bias=bias)

        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_size: Optional[int] = None,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            raise NotImplementedError

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if output_size is passed we force the interpolation output
        # size and do not make use of scale_factor=2
        if self.interpolate:
            B, C, T, H, W = hidden_states.shape
            first_h, other_h = hidden_states.split((1, T - 1), dim=2)
            if output_size is None:
                if T > 1:
                    other_h = F.interpolate(other_h, scale_factor=self.upsample_factor, mode="nearest")

                first_h = first_h.squeeze(2)
                first_h = F.interpolate(first_h, scale_factor=self.upsample_factor[1:], mode="nearest")
                first_h = first_h.unsqueeze(2)
            else:
                raise NotImplementedError

            if T > 1:
                hidden_states = torch.cat((first_h, other_h), dim=2)
            else:
                hidden_states = first_h

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states


class DownsampleCausal3D(nn.Module):
    """
    A 3D downsampling layer with an optional convolution.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size=3,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        stride=2,#1pad,2stride,3kernel可以实现减半
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding #没用上这个参数
        stride = stride
        self.name = name
        #都是在通道维度
        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        if use_conv:
            conv = CausalConv3d(
                self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, bias=bias
            )
        else:
            raise NotImplementedError

        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states: torch.FloatTensor, scale: float = 1.0) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels #[B, C, H, W]

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states)

        return hidden_states


class ResnetBlockCausal3D(nn.Module):
    r"""
    A Resnet block.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        # default, scale_shift, ada_group, spatial
        time_embedding_norm: str = "default",
        kernel: Optional[torch.FloatTensor] = None,
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_3d_out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        linear_cls = nn.Linear

        if groups_out is None:
            groups_out = groups

        if self.time_embedding_norm == "ada_group":
            self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
        elif self.time_embedding_norm == "spatial":
            self.norm1 = SpatialNorm(in_channels, temb_channels)
        else:
            self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        #第一次卷积 T,H,W维度没有发生变化,仅channel有可能变化
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, stride=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = linear_cls(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = linear_cls(temb_channels, 2 * out_channels)
            elif self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
                self.time_emb_proj = None
            else:
                raise ValueError(f"Unknown time_embedding_norm : {self.time_embedding_norm} ")
        else:
            self.time_emb_proj = None

        if self.time_embedding_norm == "ada_group":
            self.norm2 = AdaGroupNorm(temb_channels, out_channels, groups_out, eps=eps)
        elif self.time_embedding_norm == "spatial":
            self.norm2 = SpatialNorm(out_channels, temb_channels)
        else:
            self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(dropout)
        conv_3d_out_channels = conv_3d_out_channels or out_channels
        #第二次卷积 H,W维度没有发生变化,仅channel有可能变化
        self.conv2 = CausalConv3d(out_channels, conv_3d_out_channels, kernel_size=3, stride=1)

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None
        if self.up:
            self.upsample = UpsampleCausal3D(in_channels, use_conv=False)
        elif self.down:
            self.downsample = DownsampleCausal3D(in_channels, use_conv=False, name="op")

        self.use_in_shortcut = self.in_channels != conv_3d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = CausalConv3d(
                in_channels,
                conv_3d_out_channels,
                kernel_size=1,
                stride=1,
                bias=conv_shortcut_bias,
            )

    def forward(
        self,
        input_tensor: torch.FloatTensor,
        temb: torch.FloatTensor,
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        hidden_states = input_tensor

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm1(hidden_states, temb)
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = (
                self.upsample(input_tensor, scale=scale)
            )
            hidden_states = (
                self.upsample(hidden_states, scale=scale)
            )
        elif self.downsample is not None:
            input_tensor = (
                self.downsample(input_tensor, scale=scale)
            )
            hidden_states = (
                self.downsample(hidden_states, scale=scale)
            )

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = (
                self.time_emb_proj(temb, scale)[:, :, None, None]
            )

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm2(hidden_states, temb)
        else:
            hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = (
                self.conv_shortcut(input_tensor)
            )
        #残差
        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor

def get_down_block3d(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    downsample_stride: int,
    resnet_eps: float,
    resnet_act_fn: str,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    downsample_padding: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    downsample_type: Optional[str] = None,
    dropout: float = 0.0,
):
    # If attn head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        logger_.warning(
            f"It is recommended to provide attention_head_dim when calling get_down_block. Defaulting attention_head_dim to {num_attention_heads}."
        )
        attention_head_dim = num_attention_heads

    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownEncoderBlockCausal3D":
        return DownEncoderBlockCausal3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            downsample_stride=downsample_stride,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    raise ValueError(f"{down_block_type} does not exist.")

def get_up_block3d(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    prev_output_channel: int,
    temb_channels: int,
    add_upsample: bool,
    upsample_scale_factor: Tuple,
    resnet_eps: float,
    resnet_act_fn: str,
    resolution_idx: Optional[int] = None,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    upsample_type: Optional[str] = None,
    dropout: float = 0.0,
) -> nn.Module:
    # If attn head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        logger_.warning(
            f"It is recommended to provide attention_head_dim when calling get_up_block. Defaulting attention_head_dim to {num_attention_heads}."
        )
        attention_head_dim = num_attention_heads

    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpDecoderBlockCausal3D":
        return UpDecoderBlockCausal3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            upsample_scale_factor=upsample_scale_factor,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temb_channels=temb_channels,
        )
    raise ValueError(f"{up_block_type} does not exist.")

class UNetMidBlockCausal3D(nn.Module):
    """
    中间模块，包含 (num_layers + 1) 个 ResnetBlockCausal3D 以及可选多头 Attention。
    """

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attn_groups: int = None,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
    ):
        super().__init__()
        self.add_attention = add_attention
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        if attn_groups is None:
            attn_groups = resnet_groups if resnet_time_scale_shift == "default" else None

        # 第 0 个 Resnet
        resnets = [
            ResnetBlockCausal3D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        if attention_head_dim is None:
            logger_.warning(
                f"It is not recommended to pass attention_head_dim=None. Defaulting attention_head_dim to {in_channels}."
            )
            attention_head_dim = in_channels

        # 后面 num_layers 次：Attention + Resnet
        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=attn_groups,
                        spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        # 总计 (num_layers + 1) 个 ResnetBlock
        self.num_resblocks = num_layers + 1

        # 用于存储每个resnet是否在前/后做 pooling/padding
        self.resnet_pool_configs = [None] * self.num_resblocks
        self.resnet_pad_configs  = [None] * self.num_resblocks

    def apply_t_ops_config_midblock(self, config: dict):
        if not isinstance(config, dict):
            return

        epb = config.get("enable_t_pool_before_block", [])
        epa = config.get("enable_t_pool_after_block", [])

        if any(len(lst) != self.num_resblocks for lst in [epb, epa]):
            raise ValueError(
                f"[UNetMidBlockCausal3D] T-ops config mismatch: we have {self.num_resblocks} ResnetBlock(s), "
                f"but got list lengths: {list(map(len, [epb, epa, edb, eda]))}"
            )

        pool_k = config.get("pool_t_kernel", 2)
        pool_s = config.get("pool_t_stride", 2)


        for i in range(self.num_resblocks):
            self.resnet_pool_configs[i] = {
                "enable_before": epb[i],
                "enable_after":  epa[i],
                "kernel": pool_k,
                "stride": pool_s
            }

    def forward(self, hidden_states: torch.FloatTensor, temb: torch.FloatTensor = None) -> torch.FloatTensor:
        """
        先经过第 0 个 ResnetBlock（无 attention），然后再重复(Attention + ResnetBlock) num_layers次。
        在每个 ResnetBlock 前/后，根据 self.resnet_pool_configs[i] / self.resnet_pad_configs[i] 决定是否做 replicate pad + avg_pool3d。
        """
        for i in range(self.num_resblocks):
            # 如果这是第 0 个 resnet，就没有 attention；否则 i>=1 时注意 handle attention
            if i > 0:
                # 先做 attention
                attn = self.attentions[i - 1]  # 第 (i-1) 个 attention
                if attn is not None:
                    B, C, T, H, W = hidden_states.shape
                    hidden_states = rearrange(hidden_states, "b c f h w -> b (f h w) c")
                    attention_mask = prepare_causal_attention_mask(T, H*W, hidden_states.dtype, hidden_states.device, batch_size=B)
                    hidden_states = attn(hidden_states, temb=temb, attention_mask=attention_mask)
                    hidden_states = rearrange(hidden_states, "b (f h w) c -> b c f h w", f=T, h=H, w=W)
            
            pool_conf = self.resnet_pool_configs[i] or {}
            if pool_conf.get("enable_before", False):
                k, s = pool_conf["kernel"], pool_conf["stride"]
                hidden_states = F.pad(hidden_states, pad=(0,0,0,0,k-1,0), mode='replicate')
                hidden_states = F.avg_pool3d(hidden_states, kernel_size=(k,1,1), stride=(s,1,1))
            # 再做 ResnetBlock
            hidden_states = self.resnets[i](hidden_states, temb=temb)

            # 是否在 ResnetBlock 之后做 pool/pad
            if pool_conf.get("enable_after", False):
                k, s = pool_conf["kernel"], pool_conf["stride"]
                hidden_states = F.pad(hidden_states, pad=(0,0,0,0,k-1,0), mode='replicate')
                hidden_states = F.avg_pool3d(hidden_states, kernel_size=(k,1,1), stride=(s,1,1))

        return hidden_states

class DownEncoderBlockCausal3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_stride: int = 2,
        downsample_padding: int = 1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    DownsampleCausal3D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                        stride=downsample_stride,
                    )
                ]
            )
        else:
            self.downsamplers = None
        self.resnet_pool_configs = [None] * num_layers
        
    def apply_t_ops_config(self, block_config: dict):
        if "downsample_stride" in block_config:
            ds_stride = tuple(block_config["downsample_stride"])  # e.g. [1, 2, 2] -> (1, 2, 2)
            if self.downsamplers is not None:
                for ds in self.downsamplers:
                    ds.conv.conv.stride = ds_stride
                    
        num_resnet = len(self.resnets)
        epb = block_config.get("enable_t_pool_before_block", [])
        epa = block_config.get("enable_t_pool_after_block", [])
        if any(len(x) != num_resnet for x in [epb, epa]):
            raise ValueError(
                f"[DownEncoderBlockCausal3D] config mismatch: expecting {num_resnet} bools in each list."
            )

        pool_k = block_config.get("pool_t_kernel", 2)
        pool_s = block_config.get("pool_t_stride", 2)

        for i in range(num_resnet):
            pool_conf = {
                "enable_before": epb[i],
                "enable_after":  epa[i],
                "kernel": pool_k,
                "stride": pool_s
            }
            self.resnet_pool_configs[i] = pool_conf
            
    def forward(self, hidden_states: torch.FloatTensor, scale: float = 1.0, index: int = None) -> torch.FloatTensor:
        for i, resnet in enumerate(self.resnets):
            pool_conf = self.resnet_pool_configs[i] or {}
            if pool_conf.get("enable_before", False):
                k, s = pool_conf["kernel"], pool_conf["stride"]
                _logger.info(f"DownEncoderBlockCausal3D Pooling before ResnetBlock: kernel={k}, stride={s}, hidden_states.shape={hidden_states.shape}, layer={i}, index={index}")
                padding = (k-1, 0)  # 仅在时间维度前向填充 k-1 个像素
                hidden_states = F.pad(hidden_states, pad=(0, 0, 0, 0, padding[0], padding[1]), mode='replicate')
                hidden_states = F.avg_pool3d(hidden_states, kernel_size=(k, 1, 1), stride=(s, 1, 1))      

            # 2) ResnetBlock
            hidden_states = resnet(hidden_states, temb=None, scale=scale)

            # 3) pool/pad AFTER
            if pool_conf.get("enable_after", False):
                k, s = pool_conf["kernel"], pool_conf["stride"]
                padding = (k-1, 0)  # 仅在时间维度前向填充 k-1 个像素
                hidden_states = F.pad(hidden_states, pad=(0, 0, 0, 0, padding[0], padding[1]), mode='replicate')
                hidden_states = F.avg_pool3d(hidden_states, kernel_size=(k, 1, 1), stride=(s, 1, 1))      
                _logger.info(f"DownEncoderBlockCausal3D Pooling after ResnetBlock: kernel={k}, stride={s}, hidden_states.shape={hidden_states.shape}, layer={i}, index={index}")

        # 下采样
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, scale)

        return hidden_states

class UpDecoderBlockCausal3D(nn.Module):
    """
    上采样解码块, 包含 num_layers 个 ResnetBlockCausal3D, 以及可选 UpsampleCausal3D。
    支持在每个 resnet 前/后插入 T 维度 pool/pad。
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        upsample_scale_factor=(2, 2, 2),
        temb_channels: Optional[int] = None,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            block_in_ch = in_channels if i == 0 else out_channels
            block = ResnetBlockCausal3D(
                in_channels=block_in_ch,
                out_channels=out_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
            resnets.append(block)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([
                UpsampleCausal3D(
                    out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    upsample_factor=upsample_scale_factor,
                )
            ])
        else:
            self.upsamplers = None

        self.resolution_idx = resolution_idx

        # 为每个 ResnetBlock 存储 pool/pad 配置
        self.resnet_interp_configs = [None] * num_layers
        self.resolution_idx = resolution_idx
        self.num_layers = num_layers


    def apply_t_ops_config(self, block_config: dict):
        num_resnet = len(self.resnets)
        eib = block_config.get("enable_t_interp_before_block", [])
        eia = block_config.get("enable_t_interp_after_block", [])
        if any(len(x) != num_resnet for x in [eib, eia]):
            raise ValueError(
                f"[UpDecoderBlockCausal3D] config mismatch: expecting {num_resnet} bools in each list."
            )

        interp_scale = block_config.get("interp_t_scale_factor", 2)
        interp_mode  = block_config.get("interp_mode", "nearest")
        for i in range(num_resnet):
            interp_conf = {
                "enable_before": eib[i],
                "enable_after":  eia[i],
                "scale_factor":  interp_scale,
                "mode":          interp_mode
            }
            self.resnet_interp_configs[i] = interp_conf

    def forward(
        self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None, scale: float = 1.0
    ) -> torch.FloatTensor:
        """
        在每个 ResnetBlock 前/后，看情况做:
          - avg_pool3d (pool)
          - replicate pad (pad)
          - F.interpolate (插值)
        然后再做 block 本身。
        """
        for i, resnet in enumerate(self.resnets):
            interp_conf = self.resnet_interp_configs[i] or {}

            # 1) 先看看有没有 "插值前"：
            if interp_conf.get("enable_before", False):
                sc = interp_conf["scale_factor"]
                mode = interp_conf["mode"]
                # 只对 time 维度插值 (dim=2)
                # nearest插值非平滑, 但支持反向传播
                if hidden_states.shape[2] > 0:
                    hidden_states = F.interpolate(
                        hidden_states,
                        scale_factor=(sc, 1, 1),  # 仅沿时间维度放大 sc 倍
                        mode=mode
                    )

            hidden_states = resnet(hidden_states, temb=temb, scale=scale)

            if interp_conf.get("enable_after", False):
                sc = interp_conf["scale_factor"]
                mode = interp_conf["mode"]
                if hidden_states.shape[2] > 0:
                    hidden_states = F.interpolate(
                        hidden_states,
                        scale_factor=(sc, 1, 1),
                        mode=mode
                    )

        # upsample (原逻辑)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states
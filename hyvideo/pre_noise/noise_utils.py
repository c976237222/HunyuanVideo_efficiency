import torch

def randn_tensor_tile(
    shape: tuple,
    generator: torch.Generator = None,
    device: torch.device = None,
    dtype: torch.dtype = None,
    layout: torch.layout = None,
    t_group_size: int = 3
):
    """
    生成一个随机张量，并在 T 维度每 `t_group_size` 个时间步使用一个新的 `torch.Generator`，
    但仅复制低频部分，高频部分重新随机生成，确保时域高斯分布。

    - shape: (B, C, T, H, W)
    - generator: torch.Generator
    - device: 目标设备
    - dtype: 数据类型
    - layout: 存储格式
    - t_group_size: 每几个 T 共享一个 Generator
    """
    batch_size, num_channels, num_timesteps, height, width = shape
    layout = layout or torch.strided
    device = device or torch.device("cpu")
    dtype = dtype or torch.float32

    # 计算 t_group 数量
    num_t_groups = (num_timesteps + t_group_size - 1) // t_group_size

    if generator is None:
        generator = torch.Generator(device=device).manual_seed(42)

    latents = torch.zeros(shape, device=device, dtype=dtype, layout=layout)

    for t_group in range(num_t_groups):
        t_start = t_group * t_group_size
        t_end = min((t_group + 1) * t_group_size, num_timesteps)

        group_shape = (batch_size, num_channels, t_end - t_start, height, width)

        if t_group == 0:
            latents[:, :, t_start:t_end, :, :] = torch.randn(
                group_shape,
                generator=generator,
                device=device,
                dtype=dtype,
                layout=layout
            )
        else:
            prev_group = latents[:, :, t_start - t_group_size:t_start, :, :].clone().to(torch.float32)
            # === Step 1: FFT ===
            prev_fft = torch.fft.fftn(prev_group, dim=(-3, -2, -1))

            # === Step 2: 高频噪声 FFT ===
            local_gen = torch.Generator(device=device).manual_seed(43 + t_group)
            noise = torch.randn(prev_group.shape, generator=local_gen, device=prev_group.device, dtype=prev_group.dtype)

            noise_fft = torch.fft.fftn(noise, dim=(-3, -2, -1))

            # === Step 3: 构建频率 mask（椭球形低频区域） ===
            T, H, W = prev_group.shape[-3:]
            t_freq = torch.fft.fftfreq(T, d=1.0).to(device)
            h_freq = torch.fft.fftfreq(H, d=1.0).to(device)
            w_freq = torch.fft.fftfreq(W, d=1.0).to(device)

            grid_t, grid_h, grid_w = torch.meshgrid(t_freq, h_freq, w_freq, indexing='ij')
            # 可调参数，越小保留频率越少
            cutoff_t, cutoff_h, cutoff_w = 0.01, 0.02, 0.02 #0.01几乎为正常，0.05有伪影
            radius = (grid_t / cutoff_t) ** 2 + (grid_h / cutoff_h) ** 2 + (grid_w / cutoff_w) ** 2
            freq_mask = (radius <= 1).float()  # 低频区域为1，其余为0
            mask = freq_mask[None, None, :, :, :]  # 扩展到 [1,1,T,H,W]

            # === Step 4: 混合低频 + 高频 ===
            low_freq = prev_fft * mask
            high_freq = noise_fft * (1 - mask)
            mixed_fft = low_freq + high_freq

            # === Step 5: IFFT 回到时域 + 标准化 ===
            new_group = torch.fft.ifftn(mixed_fft, dim=(-3, -2, -1)).real
            new_group = (new_group - new_group.mean()) / (new_group.std() + 1e-6)

            latents[:, :, t_start:t_end, :, :] = new_group.to(torch.float16)

    return latents


def enhance_latents_with_low_freq(
    latents: torch.Tensor,
    device: torch.device = None,
    dtype: torch.dtype = None,
    layout: torch.layout = None,
    cutoff_t: float = 0.01,
    cutoff_h: float = 0.02,
    cutoff_w: float = 0.02,
    low_freq_boost: float = 1.0,
    high_freq_seed: int = 43,
):
    """
    对整个 latents 进行全局 FFT，增强低频，并用高频随机噪声补全，保持时域高斯分布。
    
    参数：
        latents (torch.Tensor): 输入张量，形状为 (B, C, T, H, W)
        cutoff_* (float): 控制低频掩码的频率阈值
        low_freq_boost (float): 低频增强系数（>1 会增强低频）
        high_freq_seed (int): 高频噪声的种子
    """
    B, C, T, H, W = latents.shape
    device = device or latents.device
    dtype = dtype or latents.dtype
    layout = layout or latents.layout

    latents = latents.to(torch.float32)

    # === Step 1: FFT ===
    latents_fft = torch.fft.fftn(latents, dim=(-3, -2, -1))

    # === Step 2: 构建频率掩码 ===
    t_freq = torch.fft.fftfreq(T, d=1.0).to(device)
    h_freq = torch.fft.fftfreq(H, d=1.0).to(device)
    w_freq = torch.fft.fftfreq(W, d=1.0).to(device)

    grid_t, grid_h, grid_w = torch.meshgrid(t_freq, h_freq, w_freq, indexing='ij')
    radius = (grid_t / cutoff_t) ** 2 + (grid_h / cutoff_h) ** 2 + (grid_w / cutoff_w) ** 2
    freq_mask = (radius <= 1).float()  # 低频区域为1，其余为0
    mask = freq_mask[None, None, :, :, :]  # 扩展到 [1,1,T,H,W]

    # === Step 3: 高频噪声生成 ===
    generator = torch.Generator(device=device).manual_seed(high_freq_seed)
    noise = torch.randn(
            latents.shape,
            generator=generator,
            device=device,
            dtype=torch.float32
        )
    noise_fft = torch.fft.fftn(noise, dim=(-3, -2, -1))

    # === Step 4: 混合频率域 ===
    low_freq = latents_fft * mask * low_freq_boost
    high_freq = noise_fft * (1 - mask)
    mixed_fft = low_freq

    # === Step 5: 反变换 & 标准化 ===
    enhanced_latents = torch.fft.ifftn(mixed_fft, dim=(-3, -2, -1)).real
    enhanced_latents = (enhanced_latents - enhanced_latents.mean()) / (enhanced_latents.std() + 1e-6)

    return enhanced_latents.to(dtype)

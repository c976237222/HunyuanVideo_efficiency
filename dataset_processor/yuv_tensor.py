import os
import re
import argparse
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------------------------------
# 命令行参数配置
# -------------------------------------------------
parser = argparse.ArgumentParser(description='YUV视频处理与Tensor转换')
parser.add_argument('--target_height', type=int, default=None,
                    help='目标垂直分辨率（如720/360/240），默认None表示保持原始尺寸')
parser.add_argument('--start_frame', type=int, default=None,
                    help='起始帧（包含），默认从第0帧开始')
parser.add_argument('--end_frame', type=int, default=None,
                    help='结束帧（不包含），默认处理到最后一帧')
parser.add_argument('--yuv_format', type=str, default='I420',
                    choices=['I420', 'NV12', 'YV12'],
                    help='输入 YUV 文件的格式 (默认: I420)')
args = parser.parse_args()

# -------------------------------------------------
# 日志配置
# -------------------------------------------------
logging.basicConfig(
    filename='yuv_processor.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# -------------------------------------------------
# 自动解析文件名中的 "WxH" 和 "fps" 片段
# 示例： "foo_15fps_360-1920x1080.yuv" => fps=15, width=1920, height=1080
# -------------------------------------------------
def parse_fps_width_height_from_filename(yuv_filename):
    """
    从文件名中提取 帧率(fps), 宽(width), 高(height).
    匹配模式： (\d+)fps 和 (\d+)x(\d+)
    """
    fps_pattern = re.compile(r'(\d+)fps')
    resolution_pattern = re.compile(r'(\d+)x(\d+)')

    fps_match = fps_pattern.search(yuv_filename)
    resolution_match = resolution_pattern.search(yuv_filename)

    if not fps_match or not resolution_match:
        raise ValueError(
            f"无法从文件名解析到帧率或分辨率信息：{yuv_filename}\n"
            f"请确保文件名包含类似 '15fps' 和 '1920x1080' 的字样。"
        )

    fps = float(fps_match.group(1))
    width, height = map(int, resolution_match.groups())

    return fps, width, height

# -------------------------------------------------
# 路径配置
# -------------------------------------------------
base_dir = "/mnt/public/wangsiyuan/k8bfn0qsj9fs1rwnc2x75z6t7/BVI-HFR"
output_base = "/mnt/public/wangsiyuan/HunyuanVideo_efficiency/video_data"
resolution_tag = f"{args.target_height}p" if args.target_height else "original"
hz = "30hz"
video_dir = os.path.join(base_dir, f"{hz}")  # 假设此目录下存放 .yuv 文件
output_video_dir = os.path.join(output_base, f"{hz}_{resolution_tag}_videos")
output_tensor_dir = os.path.join(output_base, f"{hz}_{resolution_tag}_tensors")

# 确保输出目录存在
os.makedirs(output_video_dir, exist_ok=True)
os.makedirs(output_tensor_dir, exist_ok=True)

# -------------------------------------------------
# 1) 读取原始 YUV420 文件为 BGR 格式帧
# -------------------------------------------------
def read_yuv_frames(yuv_path, width, height, start_frame=None, end_frame=None):
    """
    从指定的 YUV (I420/NV12/YV12) 文件中读取帧并转换为 BGR 格式（OpenCV 常用）。
    - yuv_path: YUV 文件路径
    - width, height: 原始视频帧尺寸
    - start_frame, end_frame: 帧范围过滤
    返回: [frame0, frame1, ...]，每个 frame 为 np.array(H, W, 3) in BGR
    """
    if args.yuv_format == 'I420' or args.yuv_format == 'YV12':
        frame_size = width * height * 3 // 2  # I420/YV12: 1.5 * width * height
    elif args.yuv_format == 'NV12':
        frame_size = width * height * 3 // 2  # NV12 也是 1.5 * width * height
    else:
        raise ValueError(f"不支持的 YUV 格式：{args.yuv_format}")

    file_size = os.path.getsize(yuv_path)
    total_frames = file_size // frame_size

    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames
    if start_frame is None:
        start_frame = 0

    if start_frame >= end_frame:
        return []

    frames = []

    with open(yuv_path, 'rb') as f:
        # 跳过 start_frame 之前的所有数据
        skip_bytes = start_frame * frame_size
        f.seek(skip_bytes, 0)

        # 连续读取 [start_frame, end_frame) 区间的帧
        for frame_idx in range(start_frame, end_frame):
            yuv_data = f.read(frame_size)
            if len(yuv_data) < frame_size:
                logging.warning(f"文件 {yuv_path} 的帧 {frame_idx} 数据不完整，已跳过。")
                break  # 防止文件尾部不完整

            try:
                # 将 YUV 数据重塑为 (height * 3 // 2, width)
                if args.yuv_format == 'I420' or args.yuv_format == 'YV12':
                    yuv_i420 = np.frombuffer(yuv_data, dtype=np.uint8).reshape((height * 3 // 2, width))
                elif args.yuv_format == 'NV12':
                    yuv_i420 = np.frombuffer(yuv_data, dtype=np.uint8).reshape((height * 3 // 2, width))
                
                # 调用 OpenCV 进行颜色空间转换到 BGR
                if args.yuv_format == 'YV12':
                    bgr_frame = cv2.cvtColor(yuv_i420, cv2.COLOR_YUV2BGR_YV12)
                elif args.yuv_format == 'NV12':
                    bgr_frame = cv2.cvtColor(yuv_i420, cv2.COLOR_YUV2BGR_NV12)
                else:  # I420
                    bgr_frame = cv2.cvtColor(yuv_i420, cv2.COLOR_YUV2BGR_I420)

                frames.append(bgr_frame)
            except Exception as e:
                logging.error(f"处理文件 {yuv_path} 的帧 {frame_idx} 时出错：{e}")
                continue

    return frames

# -------------------------------------------------
# 2) 写出帧到 MP4 文件（可选分辨率变换）
# -------------------------------------------------
def write_frames_to_mp4(frames, output_path, fps, target_size=None):
    """
    给定一批 BGR 格式帧，将其写到 MP4 文件。
    - frames: list of np.array(H, W, 3) in BGR
    - fps: 帧率
    - target_size: (width, height) or None
    """
    if not frames:
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    if target_size is None:
        H, W, _ = frames[0].shape
        out_size = (W, H)
    else:
        out_size = target_size

    out = cv2.VideoWriter(output_path, fourcc, fps, out_size)

    for frame in frames:
        if target_size is not None:
            frame = cv2.resize(frame, out_size)
        out.write(frame)

    out.release()

# -------------------------------------------------
# 3) 核心处理函数
# -------------------------------------------------
def process_yuv_video(yuv_file):
    """
    对单个 .yuv 文件进行处理：
      1. 解析帧率、宽高(从文件名推断)
      2. 读取并可选做分辨率缩放（--target_height）
      3. 写出 .mp4（若需要）
      4. 转换成 PyTorch Tensor 并保存 (.pt)
    """
    input_path = os.path.join(video_dir, yuv_file)
    output_mp4_path = os.path.join(output_video_dir, yuv_file.replace(".yuv", ".mp4"))
    tensor_path = os.path.join(output_tensor_dir, yuv_file.replace(".yuv", ".pt"))

    # 1) 自动解析帧率和分辨率
    try:
        fps, width, height = parse_fps_width_height_from_filename(yuv_file)
        logging.info(f"解析文件 {yuv_file} 的分辨率为 {width}x{height}，帧率为 {fps}fps。")
    except ValueError as e:
        logging.error(f"无法处理文件 {yuv_file}，原因：{e}")
        return f"❌ 无法处理：{yuv_file}\n原因：{e}"

    # 2) 读取帧
    frames_bgr = read_yuv_frames(
        yuv_path=input_path,
        width=width,
        height=height,
        start_frame=args.start_frame,
        end_frame=args.end_frame
    )
    if not frames_bgr:
        logging.warning(f"文件 {yuv_file} 没有有效帧，已跳过。")
        return f"⚠️ 跳过空视频：{yuv_file}"

    # 3) 目标分辨率变换（若指定 --target_height）
    if args.target_height:
        new_height = args.target_height
        aspect_ratio = width / height
        new_width = int(new_height * aspect_ratio)
        new_width = (new_width // 2) * 2  # 偶数化

        # 写出mp4
        try:
            write_frames_to_mp4(
                frames=frames_bgr,
                output_path=output_mp4_path,
                fps=fps,  # 使用解析的 fps
                target_size=(new_width, new_height)
            )
            logging.info(f"写出 MP4 文件 {output_mp4_path}，尺寸为 {new_width}x{new_height}。")
        except Exception as e:
            logging.error(f"写出 MP4 文件 {output_mp4_path} 时出错：{e}")
            return f"❌ 写出 MP4 文件时出错：{yuv_file}\n原因：{e}"

        final_size = (new_width, new_height)
    else:
        # 不做缩放
        final_size = (width, height)

    # 4) 转换为 PyTorch Tensor
    frames_tensor = []
    for idx, f_bgr in enumerate(frames_bgr):
        try:
            # BGR => RGB
            rgb_frame = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB)
            tensor_frame = transforms.ToTensor()(rgb_frame)  # [3, H, W], [0,1]
            frames_tensor.append(tensor_frame)
        except Exception as e:
            logging.error(f"转换帧 {idx} 为 Tensor 时出错：{e}")
            continue

    if not frames_tensor:
        logging.warning(f"文件 {yuv_file} 没有有效的 Tensor 帧，已跳过。")
        return f"⚠️ 文件 {yuv_file} 没有有效 Tensor 帧，已跳过。"

    try:
        video_tensor = torch.stack(frames_tensor, dim=1)  # => [3, T, H, W]
        video_tensor = 2.0 * video_tensor - 1.0          # 归一化到 [-1, 1]
        torch.save(video_tensor, tensor_path)
        logging.info(f"保存 Tensor 文件 {tensor_path}，形状为 {video_tensor.shape}。")
    except Exception as e:
        logging.error(f"保存 Tensor 文件 {tensor_path} 时出错：{e}")
        return f"❌ 保存 Tensor 文件时出错：{yuv_file}\n原因：{e}"

    C, T, H, W = video_tensor.shape
    return (
        f"处理成功：{yuv_file}\n"
        f"├─ 解析到原始尺寸：{width}x{height}\n"
        f"├─ 最终输出尺寸：{final_size[0]}x{final_size[1]}\n"
        f"├─ Tensor形状：C={C}, T={T}, H={H}, W={W}\n"
        f"└─ 数值范围：[-1, 1], dtype={video_tensor.dtype}"
    )

# -------------------------------------------------
# 4) 多线程执行
# -------------------------------------------------
if __name__ == "__main__":
    # 假设目录下都是 .yuv 文件（或者您可自行筛选）
    yuv_files = [f for f in os.listdir(video_dir) if f.endswith(".yuv")]

    if not yuv_files:
        logging.error("没有找到 .yuv 文件，请检查 video_dir 路径是否正确。")
        print("❌ 没有找到 .yuv 文件，请检查 video_dir 路径是否正确。")
    else:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_yuv_video, f): f for f in yuv_files}

            for future in tqdm(as_completed(futures), total=len(yuv_files)):
                try:
                    result = future.result()
                    tqdm.write("\n" + "="*50)
                    tqdm.write(result)
                except Exception as e:
                    logging.error(f"处理文件时出错：{e}")
                    tqdm.write("\n" + "="*50)
                    tqdm.write(f"❌ 处理文件时出错：{e}")

        print("✅ 所有视频处理完成！")
        logging.info("所有视频处理完成。")

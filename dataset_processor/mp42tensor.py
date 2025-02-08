import os
import torch
import cv2
import argparse
import torchvision.transforms as transforms
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 🛠️ 命令行参数配置
parser = argparse.ArgumentParser(description='视频处理与Tensor转换')
parser.add_argument('--target_height', type=int, default=None,
                    help='目标垂直分辨率（如720/360/240），默认None表示保持原始尺寸')
parser.add_argument('--start_time', type=float, default=None,
                    help='起始时间（秒），默认从第0秒开始')
parser.add_argument('--end_time', type=float, default=None,
                    help='结束时间（秒），默认处理到视频的最后一秒')
args = parser.parse_args()

# 📂 动态生成输出路径
base_dir = "/home/hanling/HunyuanVideo_efficiency/video_data"
resolution_tag = f"{args.target_height}p" if args.target_height else "original"

video_dir = os.path.join(base_dir, "large_motion2")
output_video_dir = os.path.join(base_dir, f"large_motion2_{resolution_tag}_videos")
output_tensor_dir = os.path.join(base_dir, f"large_motion2_{resolution_tag}_tensors")

# 确保输出目录存在
os.makedirs(output_video_dir, exist_ok=True)
os.makedirs(output_tensor_dir, exist_ok=True)

# 🎥 视频处理函数
def process_video(video_file):
    """完整的视频处理流水线"""
    input_path = os.path.join(video_dir, video_file)
    output_path = os.path.join(output_video_dir, video_file)
    tensor_path = os.path.join(output_tensor_dir, video_file.replace(".mp4", ".pt"))
    
    # 读取原始视频参数
    cap = cv2.VideoCapture(input_path)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps  # 视频总时长（秒）
    
    # 打印日志帮助调试
    print(f"正在处理视频: {video_file}")
    print(f"视频总帧数: {total_frames}, FPS: {fps}, 总时长: {total_duration:.2f}秒")
    
    # 计算开始和结束的帧数
    start_frame = 0
    end_frame = total_frames
    
    if args.start_time is not None:
        start_frame = int(args.start_time * fps)
    if args.end_time is not None:
        end_frame = int(args.end_time * fps)
    
    # 确保起始帧和结束帧的范围是有效的
    start_frame = max(0, start_frame)
    end_frame = min(total_frames, end_frame)

    # 打开视频流重新进行读取
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # ===== 阶段1：分辨率处理 =====
    if args.target_height:
        # 计算新尺寸（保持宽高比）
        new_height = args.target_height
        new_width = int(orig_width * (new_height / orig_height))
        new_width = new_width // 2 * 2  # 宽度调整为偶数
        
        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
        
        current_frame = start_frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or current_frame >= end_frame: break
            
            # 调整分辨率并写入
            resized_frame = cv2.resize(frame, (new_width, new_height))
            out.write(resized_frame)
            current_frame += 1
        
        cap.release()
        out.release()
        video_path = output_path
        final_size = (new_width, new_height)
    else:
        # 直接使用原始视频
        video_path = input_path
        cap.release()  # 先关闭视频读取器，后面再重新打开
        cap = cv2.VideoCapture(video_path)
        final_size = (int(cap.get(3)), int(cap.get(4)))  # (width, height)

    # ===== 阶段2：Tensor转换 =====
    cap = cv2.VideoCapture(video_path)
    frames = []
    current_frame = start_frame
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or current_frame >= end_frame: break
        
        # 转换为Tensor并标准化
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor_frame = transforms.ToTensor()(frame)  # [0,1]范围
        frames.append(tensor_frame)
        current_frame += 1
    
    cap.release()
    
    if not frames:
        return f"⚠️ 跳过空视频：{video_file}"
    
    # 组合Tensor
    video_tensor = torch.stack(frames)          # (T, C, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3)  # (C, T, H, W)
    video_tensor = 2 * video_tensor - 1         # [-1, 1]范围
    
    # 保存结果
    torch.save(video_tensor, tensor_path)
    
    # 返回处理信息
    C, T, H, W = video_tensor.shape
    return (
        f"处理成功：{video_file}\n"
        f"├─ 视频尺寸：{final_size[0]}x{final_size[1]}\n"
        f"├─ Tensor形状：C={C}, T={T}, H={H}, W={W}\n"
        f"└─ 数值范围：[-1, 1] (dtype: {video_tensor.dtype})"
    )

# ⚙️ 多线程执行
if __name__ == "__main__":
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = {executor.submit(process_video, f): f for f in video_files}
        
        for future in tqdm(as_completed(futures), total=len(video_files)):
            result = future.result()
            tqdm.write("\n" + "="*50)
            tqdm.write(result)

    print("✅ 所有视频处理完成！")

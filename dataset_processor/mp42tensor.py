import os
import torch
import cv2
import torchvision.transforms as transforms
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# 📂 文件路径
video_dir = "/mnt/public/wangsiyuan/HunyuanVideo_efficiency/video_data/video_data_5000"
output_video_dir = "/mnt/public/wangsiyuan/HunyuanVideo_efficiency/video_data/video_data_5000_240p"
output_tensor_dir = "/mnt/public/wangsiyuan/HunyuanVideo_efficiency/video_data/video_data_5000_240p_tensor"

# 确保输出目录存在
os.makedirs(output_video_dir, exist_ok=True)
os.makedirs(output_tensor_dir, exist_ok=True)

# 获取所有 MP4 文件
video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

# 线程数（根据 CPU 核心数调整）
NUM_THREADS = 50

# 统计跳过的文件数
skipped_count = 0

# 预处理转换
transform = transforms.ToTensor()

def resize_video(input_path, output_path, target_height=240):
    """ 将视频调整为 240p 并存储 """
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 输出 MP4 格式
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 仅调整大于 target_height 的视频
    if height > target_height:
        new_width = int(width * (target_height / height))  # 按比例计算新宽度
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, target_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, (new_width, target_height))  # 调整分辨率
            out.write(frame_resized)

        cap.release()
        out.release()
        return True
    else:
        cap.release()
        return False

def video_to_tensor(video_path):
    """ 读取 240p 视频并转换为 Tensor，数值范围映射至 [-1, 1] """
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # OpenCV 读取的是 BGR，需要转换为 RGB
        tensor_frame = transform(frame)  # [0,255] → [0,1]，(H, W, C) → (C, H, W)
        frames.append(tensor_frame)

    cap.release()

    if len(frames) == 0:
        return None  # 处理空视频情况

    video_tensor = torch.stack(frames)  # (T, C, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3)  # (T, C, H, W) → (C, T, H, W)
    
    # 映射到 [-1, 1]
    video_tensor = 2 * video_tensor - 1  # [0,1] → [-1,1]
    
    return video_tensor

def process_video(video_file):
    """ 处理单个视频文件：调整尺寸 + 转换为 Tensor """
    video_path = os.path.join(video_dir, video_file)
    resized_video_path = os.path.join(output_video_dir, video_file)
    tensor_path = os.path.join(output_tensor_dir, video_file.replace(".mp4", ".pt"))

    # 读取视频高度
    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if height > 240:
        success = resize_video(video_path, resized_video_path, target_height=240)

        if success:
            video_tensor = video_to_tensor(resized_video_path)
            if video_tensor is not None:
                torch.save(video_tensor, tensor_path)
                return f"✅ {video_file} | shape: {video_tensor.shape}, dtype: {video_tensor.dtype}, range ~ [-1,1]"
            else:
                return f"⚠️ Skipping empty video: {video_file}"
        else:
            return f"❌ Failed to resize: {video_file}"
    else:
        return f"🔹 Skipping video (<= 240p): {video_file}"

# 使用多线程并行处理视频
with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    futures = {executor.submit(process_video, video): video for video in video_files}

    for future in tqdm(as_completed(futures), total=len(video_files), desc="Processing Videos"):
        tqdm.write(future.result())

print("✅ 所有视频处理完成！")

import os
import torch
import cv2
import torchvision.transforms as transforms
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

video_dir = "/home/hanling/HunyuanVideo_efficiency/video_data/video_data_100_no_240p_double"
output_tensor_dir = "/home/hanling/HunyuanVideo_efficiency/video_data/video_data_100_no_240p_double_trnsor"

# 确保输出目录存在
os.makedirs(output_tensor_dir, exist_ok=True)

# 获取所有 MP4 文件
video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

# 线程数配置
NUM_THREADS = 128

def video_to_tensor(video_path):
    """读取视频并转换为[-1,1]范围的Tensor"""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor_frame = transforms.ToTensor()(frame)  # [0,1]范围
        frames.append(tensor_frame)

    cap.release()

    if len(frames) == 0:
        return None

    video_tensor = torch.stack(frames)          # (T, C, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3)  # (C, T, H, W)
    return 2 * video_tensor - 1                 # 映射到[-1,1]

def process_video(video_file):
    """处理单个视频的主逻辑"""
    input_path = os.path.join(video_dir, video_file)
    tensor_path = os.path.join(output_tensor_dir, video_file.replace(".mp4", ".pt"))

    # 获取视频元数据
    cap = cv2.VideoCapture(input_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    ## 跳过条件检查
    #if frame_count <= 64:
    #    return None  # 不再输出跳过信息

    try:
        # 转换为Tensor
        video_tensor = video_to_tensor(input_path)
        if video_tensor is None:
            return None  # 不输出空文件信息

        # 保存结果
        torch.save(video_tensor, tensor_path)
        return f"✅ 处理成功 | 尺寸: {video_tensor.shape} | 范围: [-1,1] | 文件: {video_file}"
    
    except Exception as e:
        return None  # 不输出错误信息

# 使用多线程并行处理
with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    futures = {executor.submit(process_video, f): f for f in video_files}
    
    # 进度条显示
    progress_bar = tqdm(total=len(video_files), desc="处理进度")
    for future in as_completed(futures):
        result = future.result()
        if result:  # 只打印成功信息
            tqdm.write(result)
        progress_bar.update(1)
    progress_bar.close()

print("✅ 所有视频处理完成！")
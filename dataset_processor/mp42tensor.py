import os
import torch
import cv2
import torchvision.transforms as transforms
from tqdm import tqdm  # 引入 tqdm 进度条库

# 📂 文件路径
video_dir = "/mnt/public/wangsiyuan/HunyuanVideo_efficiency/video_data/video_data_100"
output_video_dir = "/mnt/public/wangsiyuan/HunyuanVideo_efficiency/video_data/video_data_100_240p"
output_tensor_dir = "/mnt/public/wangsiyuan/HunyuanVideo_efficiency/video_data/video_data_100_240p_tensor"

# 确保输出目录存在
os.makedirs(output_video_dir, exist_ok=True)
os.makedirs(output_tensor_dir, exist_ok=True)

# 获取所有 MP4 文件
video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

# 统计跳过的文件数
skipped_count = 0

# 预处理转换（只转换为 Tensor，不调整大小）
# transforms.ToTensor() 会把像素值从 [0,255] 缩放到 [0,1]
transform = transforms.ToTensor()

def resize_video(input_path, output_path, target_height=240):
    """ 将视频调整为 240p 并存储 """
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 输出 MP4 格式
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"fps: {fps}, width: {width}, height: {height}")
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
        # [0,255] → [0,1]
        tensor_frame = transform(frame)  # (H, W, C) → (C, H, W)
        frames.append(tensor_frame)

    cap.release()

    if len(frames) == 0:
        return None  # 处理空视频情况

    video_tensor = torch.stack(frames)  # (T, C, H, W)

    # 调整为 (C, T, H, W)
    video_tensor = video_tensor.permute(1, 0, 2, 3)  # 将维度从 (T, C, H, W) 调整为 (C, T, H, W)
    
    # 映射到 [-1, 1]
    video_tensor = 2 * video_tensor - 1  # [0,1] → [-1,1]
    
    return video_tensor

# 遍历所有视频文件，并加入 tqdm 进度条
for video_file in tqdm(video_files, desc="Processing Videos", unit="file"):
    video_path = os.path.join(video_dir, video_file)
    resized_video_path = os.path.join(output_video_dir, video_file)
    tensor_path = os.path.join(output_tensor_dir, video_file.replace(".mp4", ".pt"))

    # 读取视频高度
    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # 仅调整大于 240p 的视频
    if height > 240:
        tqdm.write(f"Resizing: {video_file} (Original height: {height} → 240p)")
        success = resize_video(video_path, resized_video_path, target_height=240)

        if success:
            # 读取调整后的视频并转换为 Tensor
            video_tensor = video_to_tensor(resized_video_path)

            if video_tensor is not None:
                torch.save(video_tensor, tensor_path)
                tqdm.write(f"Saved: {tensor_path}, shape: {video_tensor.shape}, dtype: {video_tensor.dtype}, range ~ [-1,1]")
            else:
                tqdm.write(f"Skipping empty video after resizing: {video_file}")
        else:
            tqdm.write(f"Failed to resize: {video_file}")

    else:
        tqdm.write(f"Skipping video (<= 240p): {video_file}")
        skipped_count += 1

print(f"✅ 所有视频处理完成！跳过了 {skipped_count} 个视频")

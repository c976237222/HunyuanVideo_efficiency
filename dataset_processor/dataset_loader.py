import os
import torch
from torch.utils.data import Dataset, DataLoader

# 📂 设置数据路径
tensor_dir = "/home/hanling/HunyuanVideo_efficiency/video_data/video_data_100_240p_tensor"

# ✅ 自定义 Dataset
class VideoTensorDataset(Dataset):
    def __init__(self, tensor_dir):
        self.tensor_dir = tensor_dir
        self.tensor_files = [f for f in os.listdir(tensor_dir) if f.endswith(".pt")]
        self.tensor_files.sort()  # 确保加载顺序一致

    def __len__(self):
        return len(self.tensor_files)

    def __getitem__(self, idx):
        tensor_path = os.path.join(self.tensor_dir, self.tensor_files[idx])
        video_tensor = torch.load(tensor_path)  # (T, C, H, W)

        return video_tensor  # 可按需求返回 (video_tensor, self.tensor_files[idx])

# ✅ 创建 DataLoader
batch_size = 4  # 根据显存调整
shuffle = True  # 是否随机打乱数据
num_workers = 4  # 适用于多线程加载

dataset = VideoTensorDataset(tensor_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# ✅ 测试 DataLoader
for batch in dataloader:
    print(f"Batch Shape: {batch.shape}")  # (B, T, C, H, W)
    break  # 只打印一个 batch

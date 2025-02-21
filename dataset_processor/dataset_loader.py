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
        video_tensor = torch.load(tensor_path, weights_only=False)  # (C, T, H, W)

        return video_tensor, self.tensor_files[idx]  # 返回 (Tensor, 文件名)




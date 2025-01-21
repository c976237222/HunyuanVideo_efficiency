import os
import subprocess
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 设置路径
video_dir = "/home/hanling/HunyuanVideo_efficiency/video_data/video_data_100_240p"
output_file = "video_bitrate_100.txt"
MAX_VIDEOS = 5000  # 最多处理 1w 个视频
NUM_THREADS = 30  # 线程数，可根据 CPU 适当调整

# 获取所有 MP4 文件
video_files = sorted(glob(os.path.join(video_dir, "*.mp4")))[:MAX_VIDEOS]

# 计算码率的函数
def get_bitrate(video_path):
    """ 使用 ffprobe 计算视频的平均码率（kbps） """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "format=bit_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        bitrate = int(result.stdout.strip()) / 1000  # 转换为 kbps
        return os.path.realpath(video_path), f"{bitrate:.2f} kbps"
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return os.path.realpath(video_path), "ERROR"

# 多线程处理视频码率计算
bitrate_results = []
with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    future_to_video = {executor.submit(get_bitrate, video): video for video in video_files}
    
    for future in tqdm(as_completed(future_to_video), total=len(video_files), desc="Processing Videos"):
        bitrate_results.append(future.result())

# 将结果写入文件
with open(output_file, "w") as f:
    for path, bitrate in bitrate_results:
        f.write(f"{path} {bitrate}\n")

print(f"✅ 码率计算完成，结果已保存到: {output_file}")

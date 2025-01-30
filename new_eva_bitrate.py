import argparse
import os
import pandas as pd
import subprocess
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def parse_args():
    parser = argparse.ArgumentParser(description="计算视频 tile 码率并更新 CSV 文件。")
    parser.add_argument("--input_folder", type=str, required=True, help="输入视频文件夹路径。")
    parser.add_argument("--input_csv", type=str, required=True, help="输入 CSV 文件路径。")
    parser.add_argument("--output_csv", type=str, required=True, help="输出 CSV 文件路径。")
    parser.add_argument("--max_files", type=int, default=None, help="最大处理文件数量。设置为 None 处理所有文件。")
    return parser.parse_args()

def parse_filename(filename):
    parts = filename.split('|')
    if len(parts) < 3:
        raise ValueError(f"Unexpected filename format: {filename}")
    file_name = parts[0].strip()
    chunk_idx = int(parts[1].strip())
    return file_name, chunk_idx

def compute_tile_ci(video_path: str) -> float:
    """
    计算视频的比特率 (kbps)，直接计算原始文件的比特率。
    """
    cmd = [
        "ffprobe", "-v", "error", 
        "-select_streams", "v:0",
        "-show_entries", "stream=bit_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        bit_rate_kbps = float(output.decode().strip()) / 1000.0  # 转换为 kbps
        logging.info(f"{video_path} bit_rate(kbps) = {bit_rate_kbps}")
    except Exception as e:
        logging.error(f"ffprobe failed for {video_path}: {e}")
        bit_rate_kbps = 0.0
    
    return bit_rate_kbps

def main():
    args = parse_args()
    
    logging.info(f"输入视频文件夹: {args.input_folder}")
    logging.info(f"输入 CSV 文件: {args.input_csv}")
    logging.info(f"输出 CSV 文件: {args.output_csv}")
    logging.info(f"最大处理文件数量: {args.max_files if args.max_files else '全部'}")
    
    # 获取所有 .mp4 文件
    files = sorted(set(f for f in os.listdir(args.input_folder) if f.endswith('.mp4')))
    if args.max_files is not None:
        files = files[:args.max_files]
        logging.info(f"限制处理的文件数量为: {args.max_files}")
    
    # 读取 CSV 文件
    df = pd.read_csv(args.input_csv)
    required_columns = {'file_name', 'chunk_idx', 'tile_ci'}
    if not required_columns.issubset(df.columns):
        logging.error(f"CSV 文件缺少必要列：{required_columns - set(df.columns)}")
        return
    df['chunk_idx'] = df['chunk_idx'].astype(int)
    df.drop_duplicates(subset=['file_name', 'chunk_idx'], inplace=True)
    
    bitrate_vals = []
    keys = []
    
    # 遍历所有视频文件，直接计算原始视频的码率
    for filename in tqdm(files, desc="计算码率"):
        file_path = os.path.join(args.input_folder, filename)
        try:
            file_name, chunk_idx = parse_filename(filename)
            if (file_name, chunk_idx) in keys:
                continue
            # 直接计算原始视频的比特率
            bit_rate_kbps = compute_tile_ci(file_path)
            keys.append((file_name, chunk_idx))
            bitrate_vals.append((file_name, chunk_idx, bit_rate_kbps))
        except ValueError as e:
            logging.warning(e)
            continue
    
    # 创建新的 DataFrame 以存储比特率
    bitrate_df = pd.DataFrame(bitrate_vals, columns=['file_name', 'chunk_idx', 'tile_ci'])
    
    # 合并到原 CSV 数据
    df_updated = pd.merge(df, bitrate_df, on=['file_name', 'chunk_idx'], how='left', suffixes=('', '_new'))
    matched = df_updated['tile_ci_new'].notna()
    logging.info(f"更新 {matched.sum()} 行 tile_ci 值。")
    df_updated.loc[matched, 'tile_ci'] = df_updated.loc[matched, 'tile_ci_new']
    df_updated.drop(columns=['tile_ci_new'], inplace=True)
    df_updated.drop_duplicates(inplace=True)
    
    # 保存 CSV
    df_updated.to_csv(args.output_csv, index=False)
    logging.info(f"更新后的数据已保存到 {args.output_csv}")

if __name__ == "__main__":
    main()

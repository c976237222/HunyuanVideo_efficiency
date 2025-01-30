#!/usr/bin/env python3
# analyze_metrics.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="分析视频质量指标")
    parser.add_argument("--csv-input", type=str, required=True, help="输入CSV文件路径")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    return parser.parse_args()

def plot_metrics(df, output_dir):
    """绘制指标散点图"""
    plt.figure(figsize=(15, 5))

    # PSNR
    plt.subplot(1, 3, 1)
    sns.scatterplot(data=df, x='tile_ci', y='psnr', alpha=0.7)  # 使用散点图
    plt.title("PSNR vs Tile CI")
    plt.xlabel("Tile CI")
    plt.ylabel("PSNR (dB)")

    # SSIM
    plt.subplot(1, 3, 2)
    sns.scatterplot(data=df, x='tile_ci', y='ssim', alpha=0.7)  # 使用散点图
    plt.title("SSIM vs Tile CI")
    plt.xlabel("Tile CI")
    plt.ylim(0, 1)

    # LPIPS
    plt.subplot(1, 3, 3)
    sns.scatterplot(data=df, x='tile_ci', y='lpips', alpha=0.7)  # 使用散点图
    plt.title("LPIPS vs Tile CI")
    plt.xlabel("Tile CI")
    plt.ylim(0, 1)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "metrics_scatter.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"散点图已保存至: {plot_path}")


def main():
    args = parse_args()

    
    # 读取数据
    df = pd.read_csv(args.csv_input)
    
    # 数据清洗
    df = df.dropna()
    df = df[(df['psnr'] > 0) & (df['ssim'] > 0) & (df['lpips'] >= 0)]
    
    # 按tile_ci分组统计
    grouped = df.groupby('tile_ci').agg({
        'psnr': ['mean', 'std'],
        'ssim': ['mean', 'std'],
        'lpips': ['mean', 'std']
    }).reset_index()
    
    # 保存统计结果
    stats_path = os.path.join(args.output_dir, "metrics_stats.csv")
    grouped.to_csv(stats_path, index=False)
    print(f"统计结果已保存至: {stats_path}")
    
    # 绘制趋势图
    plot_metrics(df, args.output_dir)
    
    # 保存原始数据副本
    rawdata_path = os.path.join(args.output_dir, "raw_metrics.csv")
    df.to_csv(rawdata_path, index=False)

if __name__ == "__main__":
    main()
#!/bin/bash

ORIGINAL_VIDEOS="/mnt/public/wangsiyuan/HunyuanVideo_efficiency/video_data/video_data_5000_240p"
stride_BASE="/home/hanling/HunyuanVideo_efficiency/analysis/stride"
METRICS_BASE="/mnt/public/wangsiyuan/HunyuanVideo_efficiency/analysis/stride_metrics"
PY_METRICS="/home/hanling/HunyuanVideo_efficiency/evaluation/compute_metrics_threads.py"
PYTHON="python3"

# 遍历stride目录下的所有子文件夹
for category_dir in "$stride_BASE"/*; do
    category=$(basename "$category_dir")
    echo "=> Processing category: $category"
    
    # 创建对应类别的metrics目录
    mkdir -p "$METRICS_BASE/$category"
        
        # 执行命令（注意参数对齐）
        $PYTHON "$PY_METRICS" \
            --root1 "$ORIGINAL_VIDEOS" \
            --root2 "$category_dir" \
            --results-dir "$METRICS_BASE/$category" \
            --batch-size 2048
done

echo "All metrics computed!"
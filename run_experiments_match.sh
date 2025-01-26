#!/usr/bin/env bash

set -e
set -u

##############################################
# 1. 基础配置
##############################################

PY_INFER="infer_dynamic_separate.py"
VAE_PATH="ckpts/hunyuan-video-t2v-720p/vae"

GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
BATCH_SIZE_loop=4  # 一次同时并行多少个任务

BATCH_SIZE=1
NUM_WORKERS=4
NUM_VIDEOS_PER_TXT=1
CONFIG_JSON_DIR="/home/hanling/HunyuanVideo_efficiency/t_ops_config.json"
OUT_BASE="/mnt/public/wangsiyuan/HunyuanVideo_efficiency/analysis/stride"
EXP_OUT_DIR="/home/hanling/HunyuanVideo_efficiency/analysis/720p_tmp_stride"
TENSOR_DIR="/home/hanling/HunyuanVideo_efficiency/video_data/processed_720p_tensors"
rm -rf $EXP_OUT_DIR/*
python "$PY_INFER" \
    --tensor-dir "$TENSOR_DIR" \
    --output-dir "$EXP_OUT_DIR" \
    --vae-path "$VAE_PATH" \
    --config-json "$CONFIG_JSON_DIR" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --max-files "$NUM_VIDEOS_PER_TXT" \
    --mp4 \
    --use-adaptive
echo "[INFO] All tasks finished!"

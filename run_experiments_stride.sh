#!/usr/bin/env bash

set -e  # 遇到错误中断
set -u  # 使用未定义变量中断

##############################################
# 1. 基础配置: 请根据实际项目修改
##############################################

T_OPS_CONFIG="t_ops_config.json"                # 原始基础 config, 只做参考
PY_DYNAMIC_ENUM="dynamic_enumeration_stride_2.py" # 这次用新的脚本
PY_INFER="infer.py"
PY_METRICS="evaluation/compute_metrics.py"

# 数据与模型路径
TENSOR_DIR="/mnt/public/wangsiyuan/HunyuanVideo_efficiency/video_data/15hz_540p_tensors"
VAE_PATH="ckpts/hunyuan-video-t2v-720p/vae"
ORIGINAL_VIDEOS="/mnt/public/wangsiyuan/HunyuanVideo_efficiency/video_data/15hz_540p_videos"

# 输出目录改为 *_stride
OUT_BASE="analysis/two_true_stride"
METRICS_BASE="analysis/two_true_stride_metrics"

MAX_FILES=10
BATCH_SIZE=1
NUM_WORKERS=4

PYTHON="python"

# 确保 config_json_stride 目录存在；若已有残留则清理
CONFIG_JSON_DIR="/mnt/public/wangsiyuan/HunyuanVideo_efficiency/analysis/config_stride2_json"
mkdir -p "$CONFIG_JSON_DIR"
rm -rf "$CONFIG_JSON_DIR/exp_*.json"
rm -rf "$OUT_BASE"/*
rm -rf "$METRICS_BASE"/*
##############################################
# 2. 生成 JSON 配置
##############################################
echo "[INFO] Generating JSON combos with dynamic_enumeration_stride.py ..."
$PYTHON "$PY_DYNAMIC_ENUM" "$T_OPS_CONFIG" "$CONFIG_JSON_DIR"

# 检查 JSON 生成成功
count_json=$(ls "$CONFIG_JSON_DIR"/exp_*.json 2>/dev/null | wc -l || true)
if [[ "$count_json" -eq 0 ]]; then
  echo "[ERROR] No exp_*.json found in $CONFIG_JSON_DIR."
  exit 1
fi
echo "[INFO] Total $count_json config files found in $CONFIG_JSON_DIR."

##############################################
# 3. 遍历所有 exp_{n}.json, 逐个执行推理和指标计算
##############################################

mkdir -p "$OUT_BASE"
mkdir -p "$METRICS_BASE"

idx=0
for CONFIG_JSON in "$CONFIG_JSON_DIR"/exp_*.json; do
  idx=$((idx + 1))
  NAME="$(basename "$CONFIG_JSON" .json)"
  echo "-------------------------------------------"
  echo "[INFO] [$idx/$count_json] Running pipeline for $CONFIG_JSON => $NAME"
  echo "-------------------------------------------"

  EXP_OUT_DIR="${OUT_BASE}/${NAME}"
  EXP_METRICS_DIR="${METRICS_BASE}/${NAME}"
  mkdir -p "$EXP_OUT_DIR" "$EXP_METRICS_DIR"

  # 1) 运行推理
  echo "=> Inference: $PY_INFER"
  $PYTHON "$PY_INFER" \
    --tensor-dir "$TENSOR_DIR" \
    --output-dir "$EXP_OUT_DIR" \
    --vae-path "$VAE_PATH" \
    --config-json "$CONFIG_JSON" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --max-files "$MAX_FILES" \
    --mp4 \
    || { echo "[ERROR] infer.py failed for $CONFIG_JSON"; exit 1; }

  # 2) 计算指标
  echo "=> Compute Metrics: $PY_METRICS"
  $PYTHON "$PY_METRICS" \
    --root1 "$ORIGINAL_VIDEOS" \
    --root2 "$EXP_OUT_DIR" \
    --results-dir "$EXP_METRICS_DIR" \
    || { echo "[ERROR] compute_metrics.py failed for $CONFIG_JSON"; exit 1; }

  echo "[INFO] Done for $CONFIG_JSON"
  echo
done

echo "[INFO] All experiments finished!"

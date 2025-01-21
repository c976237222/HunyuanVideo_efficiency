#!/usr/bin/env bash

set -e  # 遇到错误中断
set -u  # 使用未定义变量中断

##############################################
# 1. 基础配置: 请根据实际项目修改
##############################################

T_OPS_CONFIG="t_ops_config.json"            
PY_DYNAMIC_ENUM="dynamic_enumeration.py"    
PY_INFER="infer.py"
PY_METRICS="evaluation/compute_metrics_threads.py"

TENSOR_DIR="video_data/video_data_100_240p_tensor"
VAE_PATH="ckpts/hunyuan-video-t2v-720p/vae"
ORIGINAL_VIDEOS="video_data/video_data_100_240p"

OUT_BASE="analysis/one_true_pool"
METRICS_BASE="analysis/one_true_pool_metrics"

MAX_FILES=100
BATCH_SIZE=1
NUM_WORKERS=4

PYTHON="python"

# 确保 config_json 目录存在
CONFIG_JSON_DIR="/mnt/public/wangsiyuan/HunyuanVideo_efficiency/analysis/config_pool_json"
mkdir -p "$CONFIG_JSON_DIR"

# 若 exp_*.json 已残留, 先清理
rm -rf "$CONFIG_JSON_DIR/exp_*.json"
rm -rf "$OUT_BASE"/*
rm -rf "$METRICS_BASE"/*
##############################################
# 2. 生成 JSON 配置
##############################################

echo "[INFO] Generating JSON combos with dynamic_enumeration.py ..."
$PYTHON "$PY_DYNAMIC_ENUM" "$T_OPS_CONFIG"

# 检查 JSON 生成成功
count_json=$(ls "$CONFIG_JSON_DIR"/exp_*.json 2>/dev/null | wc -l || true)
if [[ "$count_json" -eq 0 ]]; then
  echo "[ERROR] No exp_*.json found in $CONFIG_JSON_DIR."
  exit 1
fi
echo "[INFO] Total $count_json config files found in $CONFIG_JSON_DIR."
##############################################
# 3. 遍历所有 exp_{n}.json, 分批并行执行
##############################################

mkdir -p "$OUT_BASE"
mkdir -p "$METRICS_BASE"

# 预先定义可用的 GPU 列表（假设你有 4 块 GPU）
GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}   # 一般=4

# 把所有 JSON 文件放到一个数组中，方便按索引访问
CONFIG_FILES=("$CONFIG_JSON_DIR"/exp_*.json)
TOTAL_CONFIG=${#CONFIG_FILES[@]}
if [[ "$TOTAL_CONFIG" -eq 0 ]]; then
  echo "[ERROR] No exp_*.json found in $CONFIG_JSON_DIR."
  exit 1
fi

# 设置批大小，每批同时运行 4 个
BATCH_SIZE_loop=4

echo "[INFO] Found $TOTAL_CONFIG config files, running in batches of $BATCH_SIZE_loop."

# i 用来迭代所有 config 索引
i=0
while [[ $i -lt $TOTAL_CONFIG ]]; do

  # 一次启动 BATCH_SIZE 个并行任务
  for (( j=0; j<$BATCH_SIZE_loop; j++ )); do
    # 如果 config 数量不足以凑满最后一批，则要检查是否越界
    if [[ $i -ge $TOTAL_CONFIG ]]; then
      break
    fi

    # 选定要使用的 GPU
    GPU_ID=${GPUS[$j]}  # 注意 j < NUM_GPUS，否则要再写 % 4 之类
    CONFIG_JSON="${CONFIG_FILES[$i]}"
    NAME="$(basename "$CONFIG_JSON" .json)"

    echo "-------------------------------------------"
    echo "[INFO] Launching job $((i+1))/$TOTAL_CONFIG on GPU=$GPU_ID => $CONFIG_JSON => $NAME"
    echo "-------------------------------------------"

    EXP_OUT_DIR="${OUT_BASE}/${NAME}"
    EXP_METRICS_DIR="${METRICS_BASE}/${NAME}"
    mkdir -p "$EXP_OUT_DIR" "$EXP_METRICS_DIR"

    (
      export CUDA_VISIBLE_DEVICES="$GPU_ID"

      # 这里直接调用 python
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
    ) &  # 在子进程中后台执行

    # 增加 i，继续处理下一个文件
    i=$((i + 1))
  done

  # 等待这一批（最多 4 个）全部执行完，再进下一批
  wait
done

echo "[INFO] All experiments finished!"


## 2) 计算指标
#echo "=> Compute Metrics: $PY_METRICS"
#$PYTHON "$PY_METRICS" \
#  --root1 "$ORIGINAL_VIDEOS" \
#  --root2 "$OUT_BASE" \
#  --results-dir "$METRICS_BASE" \
#  --num-threads 100
#  --batch-size 3072

echo "[INFO] All experiments finished!"

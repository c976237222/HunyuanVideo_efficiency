#!/usr/bin/env bash

set -e
set -u

##############################################
# 1. 基础配置
##############################################

PY_INFER="infer.py"
VAE_PATH="ckpts/hunyuan-video-t2v-720p/vae"

GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
BATCH_SIZE_loop=4  # 一次同时并行多少个任务

BATCH_SIZE=1
NUM_WORKERS=4

CONFIG_JSON_DIR="/home/hanling/HunyuanVideo_efficiency/analysis/config_pool_json"
OUT_BASE="/mnt/public/wangsiyuan/HunyuanVideo_efficiency/analysis/pool"

VIDEO_BUCKET_DIR="/mnt/public/wangsiyuan/HunyuanVideo_efficiency/video_buckets"

TXT_FILES=(
  "Low_500_kbps.txt"
  "Medium_1500_2000_kbps.txt"
  "High_2500_3000_kbps.txt"
  "High_3000_3500_kbps.txt"
)

NUM_VIDEOS_PER_TXT=10

# .pt 文件所在主目录
PT_ROOT="/mnt/public/wangsiyuan/HunyuanVideo_efficiency/video_data/video_data_5000_240p_tensor"

##############################################
# 2. 先给每个 bucket 创建一个固定软链接目录 (只做一次)
##############################################

declare -A SYMLINK_DIR_MAP
mkdir -p /tmp/video_list_temp

for txt_file in "${TXT_FILES[@]}"; do
  bucket_name="${txt_file%.txt}"
  bucket_name="${bucket_name// /_}"
  bucket_name="${bucket_name//(/_}"
  bucket_name="${bucket_name//)/_}"
  bucket_name="${bucket_name//</_}"
  bucket_name="${bucket_name//>/_}"

  tmp_list="/tmp/video_list_temp/${bucket_name}.list"
  head -n "$NUM_VIDEOS_PER_TXT" "${VIDEO_BUCKET_DIR}/${txt_file}" > "$tmp_list"

  # 固定的 symlink 目录
  symlink_dir="/home/hanling/HunyuanVideo_efficiency/video_data/video_data_5000_240p_tensor/symlink_${bucket_name}"
  rm -rf "$symlink_dir"
  mkdir -p "$symlink_dir"

  # 填充软链接
  while IFS= read -r mp4_path; do
    file="$(basename "$mp4_path")"
    pt_file="${file%.mp4}.pt"
    real_pt_path="${PT_ROOT}/${pt_file}"
    ln -sf "$real_pt_path" "${symlink_dir}/${pt_file}"
  done < "$tmp_list"

  SYMLINK_DIR_MAP["$bucket_name"]="$symlink_dir"
done

echo "[INFO] Created symlink dirs for each bucket:"
for bname in "${!SYMLINK_DIR_MAP[@]}"; do
  echo "  $bname -> ${SYMLINK_DIR_MAP[$bname]}"
done

##############################################
# 3. 收集所有 exp_{i}.json + bucket 的任务列表
##############################################

CONFIG_FILES=("$CONFIG_JSON_DIR"/exp_*.json)
TOTAL_CONFIG=${#CONFIG_FILES[@]}
if [[ "$TOTAL_CONFIG" -eq 0 ]]; then
  echo "[ERROR] No exp_*.json found in $CONFIG_JSON_DIR."
  exit 1
fi

# 准备一个 tasks 数组，每个元素形如:  "config_path|bucket_name"
tasks=()
for config_file in "${CONFIG_FILES[@]}"; do
  for bucket_name in "${!SYMLINK_DIR_MAP[@]}"; do
    tasks+=("$config_file|$bucket_name")
  done
done

TOTAL_TASKS=${#tasks[@]}
echo "[INFO] We have $TOTAL_CONFIG configs * ${#SYMLINK_DIR_MAP[@]} buckets = $TOTAL_TASKS tasks."
echo "[INFO] Will run them in batches of $BATCH_SIZE_loop."

##############################################
# 4. 并行调度: 一次开 BATCH_SIZE_loop 个任务
##############################################

i=0
while [[ $i -lt $TOTAL_TASKS ]]; do
  for (( j=0; j<$BATCH_SIZE_loop; j++ )); do
    if [[ $i -ge $TOTAL_TASKS ]]; then
      break
    fi

    # 解析出 config_file 和 bucket_name
    task_str="${tasks[$i]}"
    config_file="${task_str%%|*}"
    bucket_name="${task_str#*|}"

    # GPU 分配
    GPU_ID=${GPUS[$((j % NUM_GPUS))]}

    # 获取 exp_xxx
    NAME="$(basename "$config_file" .json)"

    echo "-------------------------------------------"
    echo "[INFO] Launching job $((i+1))/$TOTAL_TASKS on GPU=$GPU_ID => $config_file => bucket=$bucket_name => $NAME"
    echo "-------------------------------------------"

    (
      export CUDA_VISIBLE_DEVICES="$GPU_ID"

      symlink_dir="${SYMLINK_DIR_MAP[$bucket_name]}"
      if [[ -z "$symlink_dir" ]]; then
        echo "[ERROR] symlink_dir not found for bucket=$bucket_name"
        exit 1
      fi

      # 推理输出
      EXP_OUT_DIR="${OUT_BASE}/${bucket_name}/${NAME}"
      mkdir -p "$EXP_OUT_DIR"

      echo "[INFO] Inference => config=$NAME, bucket=$bucket_name, symlink=$symlink_dir"

      python "$PY_INFER" \
        --tensor-dir "$symlink_dir" \
        --output-dir "$EXP_OUT_DIR" \
        --vae-path "$VAE_PATH" \
        --config-json "$config_file" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --max-files "$NUM_VIDEOS_PER_TXT" \
        --mp4 \
        || {
          echo "[ERROR] infer.py failed for $config_file on bucket=$bucket_name"
          exit 1
        }
    ) &

    i=$((i + 1))
  done

  wait
done

# 可选：rm -rf /tmp/video_list_temp
echo "[INFO] All tasks finished!"

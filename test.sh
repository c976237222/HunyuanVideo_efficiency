T_OPS_CONFIG="t_ops_config.json"            
PY_DYNAMIC_ENUM="dynamic_enumeration.py"    
PY_INFER="infer_batch.py"
PY_METRICS="evaluation/compute_metrics_threads.py"

TENSOR_DIR="video_data/video_data_100_240p_tensor"
VAE_PATH="ckpts/hunyuan-video-t2v-720p/vae"
ORIGINAL_VIDEOS="video_data/video_data_100_240p"

OUT_BASE="analysis/one_true_pool"
METRICS_BASE="analysis/one_true_pool_metrics"

MAX_FILES=100
BATCH_SIZE=2
NUM_WORKERS=4

PYTHON="python"

# 确保 config_json 目录存在
CONFIG_JSON_DIR="/mnt/public/wangsiyuan/HunyuanVideo_efficiency/analysis/config_pool_json"
mkdir -p "$CONFIG_JSON_DIR"
## 2) 计算指标
#echo "=> Compute Metrics: $PY_METRICS"
#$PYTHON "$PY_METRICS" \
#  --root1 "$ORIGINAL_VIDEOS" \
#  --root2 "$OUT_BASE" \
#  --results-dir "$METRICS_BASE" \
#  --num-threads 100
#  --batch-size 3072

echo "=> Inference: $PY_INFER"
$PYTHON "$PY_INFER" \
--tensor-dir "$TENSOR_DIR" \
--output-dir "/mnt/public/wangsiyuan/HunyuanVideo_efficiency/tmp" \
--vae-path "$VAE_PATH" \
--config-json "/home/hanling/HunyuanVideo_efficiency/t_ops_config.json" \
--batch-size "$BATCH_SIZE" \
--num-workers "$NUM_WORKERS" \
--max-files "$MAX_FILES" \
--mp4
ORIGINAL_VIDEOS="/mnt/public/wangsiyuan/HunyuanVideo_efficiency/video_data/video_data_100_240p"
OUT_BASE="/mnt/public/wangsiyuan/HunyuanVideo_efficiency/analysis/one_true_pool"
METRICS_BASE="/mnt/public/wangsiyuan/HunyuanVideo_efficiency/analysis/one_true_pool_metrics"
PY_METRICS="/home/hanling/HunyuanVideo_efficiency/evaluation/compute_metrics_threads.py"
PYTHON="python3"
# 2) 计算指标
echo "=> Compute Metrics: $PY_METRICS"
$PYTHON "$PY_METRICS" \
 --root1 "$ORIGINAL_VIDEOS" \
 --root2 "$OUT_BASE" \
 --results-dir "$METRICS_BASE" \
 --batch-size 2048
ORIGINAL_VIDEOS="video_data/video_data_100_240p"
OUT_BASE="analysis/one_true_pool"
METRICS_BASE="analysis/one_true_pool_metrics"
PY_METRICS="/home/hanling/HunyuanVideo_efficiency/evaluation/compute_metrics_threads.py"
PYTHON="python3"
# 2) 计算指标
echo "=> Compute Metrics: $PY_METRICS"
$PYTHON "$PY_METRICS" \
 --root1 "$ORIGINAL_VIDEOS" \
 --root2 "$OUT_BASE" \
 --results-dir "$METRICS_BASE" \
 --num-threads 1 \
 --batch-size 5096

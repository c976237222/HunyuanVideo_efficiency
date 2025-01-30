#ORIGINAL_VIDEOS="/mnt/public/wangsiyuan/HunyuanVideo_efficiency/video_data/video_data_100_240p"
#OUT_BASE="/mnt/public/wangsiyuan/HunyuanVideo_efficiency/analysis/one_true_pool"
#METRICS_BASE="/mnt/public/wangsiyuan/HunyuanVideo_efficiency/analysis/one_true_pool_metrics"
#PY_METRICS="/home/hanling/HunyuanVideo_efficiency/evaluation/compute_metrics_threads.py"
#PYTHON="python3"
## 2) 计算指标
#echo "=> Compute Metrics: $PY_METRICS"
#$PYTHON "$PY_METRICS" \
# --root1 "$ORIGINAL_VIDEOS" \
# --root2 "$OUT_BASE" \
# --results-dir "$METRICS_BASE" \
# --batch-size 2048
exp=nothing_1x
hz_p=15hz_720p
echo "当前实验为${exp},数据为:${hz_p}"
output=/home/hanling/HunyuanVideo_efficiency/analysis_ci/analysis_results_15hz_${exp}_720p
mkdir -p $output
label_path=/home/hanling/HunyuanVideo_efficiency/analysis/${hz_p}_reconstructed_${exp}_label
vae_output=/home/hanling/HunyuanVideo_efficiency/analysis/${hz_p}_reconstructed_${exp}
#python evaluation/compute_metrics_ci.py \
#  --root1 $label_path \
#  --root2 $vae_output \
#  --csv-output $output/metrics.csv
#只有ssim需要 用新指标计算ssim
python new_eva.py --input_folder $label_path \
                                   --input_csv $output/metrics.csv\
                                   --output_csv $output/raw_metrics_v2.csv \
                                   --max_files 88
#只有bitrate 需要
#python new_eva_bitrate.py --input_folder $label_path \
#                                   --input_csv $output/raw_metrics_v2.csv \
#                                   --output_csv $output/raw_metrics_v5.csv \
#                                   --max_files 88

#python evaluation/analyze_metrics.py \
#  --csv-input $output/raw_metrics_v2.csv \
#  --output-dir $output
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
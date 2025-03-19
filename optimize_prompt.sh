#!/bin/bash

# 检查是否提供 GPU 号参数
if [ -z "$1" ]; then
    echo "Usage: $0 <GPU_ID>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$1"

animal="cat"
prompts=(
    "A cat runs with quick, dynamic motion, grass sways."
)

output_dir="./results/idea_0"
mkdir -p "$output_dir"

for i in "${!prompts[@]}"; do
    prompt="${prompts[$i]}"
    echo "Generating video for prompt ${i}: $prompt on GPU $CUDA_VISIBLE_DEVICES"

    # Run the Python script with the specified GPU
    python3 sample_video.py \
        --video-size 360 640 \
        --video-length 129 \
        --infer-steps 50 \
        --prompt "$prompt" \
        --seed 42 \
        --embedded-cfg-scale 6.0 \
        --flow-shift 7.0 \
        --flow-reverse \
        --use-cpu-offload \
        --save-path "$output_dir" &
done

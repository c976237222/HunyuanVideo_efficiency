#!/bin/bash
# Description: This script demonstrates how to inference videos based on HunyuanVideo model with different prompts.
# The script runs each video generation in groups of 4 in parallel on different GPUs.
#A cat lunges wildly, momentum fading into tentative pawsteps amid bent blades.            tensor_1
#A cat careens through thrashing grass, chaotic energy settling into twitching vigilance.  tensor_2
#A cat surges through grass, tumult diminishing to hushed, rippling waves. tensor_4
#A cat sprints through flattened grass, strides shortening to a curious halt. tensor_6
# Define an array of 60 prompts
export CUDA_VISIBLE_DEVICES=2
animal="cat"
prompts=(
    "A cat erupts in wild bounds, settling to subtle grass tremors."
)
#A cat strolling through a meadow, small steps at first, growing faster.                    tensor
#A cat walking gently, slow motion initially, then larger strides.                          tensor_3
#A cat walking, grass barely shifting, realistic motion. tensor_5
#A cat walking slowly across grass, subtle motion, gradually increasing. tensor_7
#A cat erupts in wild bounds, settling to subtle grass tremors. tensor_8
# Loop through the prompts array and run in groups of 4
for i in "${!prompts[@]}"; do
    prompt="${prompts[$i]}"
    
    # Set GPU device (cycling through cuda devices 0,1,2,3)
    gpu_id=$((i % 4))  # This will cycle between 0, 1, 2, 3
    
    echo "Generating video for prompt ${i}: $prompt"
    
    # Create a directory for the result
    output_dir="./results/epoch_n/prompt_3"
    mkdir -p "$output_dir"
    
    # Run the python script in the background with the selected GPU
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

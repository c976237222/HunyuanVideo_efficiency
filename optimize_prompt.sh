#!/bin/bash

#A cat lunges wildly, momentum fading into tentative pawsteps amid bent blades.            tensor_1
#A cat careens through thrashing grass, chaotic energy settling into twitching vigilance.  tensor_2
#A cat surges through grass, tumult diminishing to hushed, rippling waves. tensor_4
#A cat sprints through flattened grass, strides shortening to a curious halt. tensor_6
export CUDA_VISIBLE_DEVICES=5
animal="cat"
prompts=(
    "A cat sprints through flattened grass, strides shortening to a curious halt."
)
#A cat strolling through a meadow, small steps at first, growing faster.                    tensor
#A cat walking gently, slow motion initially, then larger strides.                          tensor_3
#A cat walking, grass barely shifting, realistic motion. tensor_5
#A cat walking slowly across grass, subtle motion, gradually increasing. tensor_7
#A cat erupts in wild bounds, settling to subtle grass tremors. tensor_8
for i in "${!prompts[@]}"; do
    prompt="${prompts[$i]}"
    echo "Generating video for prompt ${i}: $prompt"
    output_dir="./results/idea_0"
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
        --save-path "$output_dir"
done

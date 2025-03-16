#!/bin/bash

# Configuration
contrast_group_file="conf/dict/labels/color_pairs_groups/color_pairs_group01.txt"
gpu=0
num_epochs=2

# Data paths (high priority defaults)
train_dir="/data/share/bin-wu/data/other/image/imagenet/sample_imagenet/test"
dev_dir="/data/share/bin-wu/data/other/image/imagenet/sample_imagenet/val"
mtest_dir=""

# Model and output paths
pretrained_model="exp/fl_ce_imagenet/test/imagenet_small/model.pt" # epoch 100
result_dir="exp/sandbox/fl_ce_imagenet/all_pairs/test"

# Model parameters (from fl_ce_color_contrast.py defaults)
batch_size=128
image_size=224
patch_size=16
dim=768
depth=12
heads=12
mlp_dim=3072
dropout=0.1
mae_pretrained=""
mae_train_layers=-1

# Training parameters (from fl_ce_color_contrast.py defaults)
base_lr=1.5e-4
weight_decay=0.05
warmup_epochs=10
total_epochs=100
save_epoch_interval=10
save_image_interval=5

# Statistical parameters
pixel_variance=0.05
cond_entropy_bits_clean="-0.8287"
num_save_images=8
seed=2025

# Other settings
merge_type="mask"

# Parse options
. local/scripts/parse_options.sh || exit 1

# Read contrast groups from file and process each line
while IFS= read -r contrast_pair; do
    # Skip empty lines
    [ -z "$contrast_pair" ] && continue

    # Clean the contrast pair string
    contrast_pair=$(echo "${contrast_pair}" | tr -d '\n\r')

    # Extract colors from contrast pair
    color_from=$(echo "${contrast_pair}" | cut -d' ' -f1)
    color_to=$(echo "${contrast_pair}" | cut -d' ' -f2)

    # Create result directory name from the contrast pair
    pair_dir=$(echo "${contrast_pair}" | tr ' ' '_')

    # Run Python script with parameters
    python local/fl_ce_color_contrast.py \
        --train_dir "${train_dir}" \
        --dev_dir "${dev_dir}" \
        --mtest_dir "${mtest_dir}" \
        --batch_size "${batch_size}" \
        --image_size "${image_size}" \
        --patch_size "${patch_size}" \
        --dim "${dim}" \
        --depth "${depth}" \
        --heads "${heads}" \
        --mlp_dim "${mlp_dim}" \
        --dropout "${dropout}" \
        --mae_pretrained "${mae_pretrained}" \
        --mae_train_layers "${mae_train_layers}" \
        --base_lr "${base_lr}" \
        --weight_decay "${weight_decay}" \
        --warmup_epochs "${warmup_epochs}" \
        --total_epochs "${total_epochs}" \
        --save_epoch_interval "${save_epoch_interval}" \
        --save_image_interval "${save_image_interval}" \
        --pixel_variance "${pixel_variance}" \
        --num_save_images "${num_save_images}" \
        --seed "${seed}" \
        --pretrained_model "${pretrained_model}" \
        --contrast_type "pair" \
        --color_from "${color_from}" \
        --color_to "${color_to}" \
        --merge_type "${merge_type}" \
        --num_epochs "${num_epochs}" \
        --result_dir "${result_dir}/${pair_dir}" \
        --cond_entropy_bits_clean "${cond_entropy_bits_clean}" \
        --gpu "$gpu" \
        --exit \
        --overwrite

done < "${contrast_group_file}"

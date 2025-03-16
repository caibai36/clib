#!/bin/bash

# Configuration
contrast_group_file="conf/dict/labels/initial_pairs_groups/initial_pairs_group01.txt"
label2id_file="conf/dict/chinese_all_label2id.txt"
gpu=3
num_epochs=2

# Data paths
train_corpus="exp/data/sample_tv07_phoneme.txt"
dev_corpus="exp/data/sample_tv07_phoneme.txt"
mtest_corpus=""

# Model and output paths
pretrained_model="exp/fl_ce_chinese/test/tv07_syllable/model.ckpt"
result_dir="exp/sandbox/fl_ce_chinese/all_pairs/test"

# Model parameters
hidden_size=384
num_layers=6
num_heads=6
dropout=0.1
batch_size=128
max_seq_length=512

# Training parameters
base_lr=1e-4
weight_decay=0.01
warmup_epochs=10
total_epochs=100
seed=2025
save_epoch_interval=10

# Other settings
connector="_"
merge_type="uniform"

# Parse the options. (e.g., ./run.sh --stage 1)
# Note that the options should be defined as shell variables before parsing
. local/scripts/parse_options.sh || exit 1

# Read contrast groups from file and process each line
while IFS= read -r contrast_pair; do
    # Skip empty lines
    [ -z "$contrast_pair" ] && continue

    # Clean the contrast pair string
    contrast_pair=$(echo "${contrast_pair}" | tr -d '\n\r')

    # Create result directory name from the contrast pair
    pair_dir=$(echo "${contrast_pair}" | tr ' ' '_')

    # Check if result directory exists
    full_result_dir="${result_dir}/${pair_dir}"
    if [ -d "$full_result_dir" ]; then
        echo "Warning: Directory ${full_result_dir} already exists. Skipping this contrast pair."
        continue
    fi

    # Run Python script with parameters
    python local/fl_ce_contrast.py \
        --syllable_fl \
        --pretrained_model "${pretrained_model}" \
        --contrast_groups "${contrast_pair}" \
        --train_corpus "${train_corpus}" \
        --dev_corpus "${dev_corpus}" \
        --mtest_corpus "${mtest_corpus}" \
        --label2id_file "${label2id_file}" \
        --num_epochs "${num_epochs}" \
        --result_dir "${result_dir}/${pair_dir}" \
        --hidden_size "${hidden_size}" \
        --num_layers "${num_layers}" \
        --num_heads "${num_heads}" \
        --batch_size "${batch_size}" \
        --max_seq_length "${max_seq_length}" \
        --dropout "${dropout}" \
        --base_lr "${base_lr}" \
        --weight_decay "${weight_decay}" \
        --warmup_epochs "${warmup_epochs}" \
        --total_epochs "${total_epochs}" \
        --seed "${seed}" \
	--gpu "$gpu" \
        --save_epoch_interval "${save_epoch_interval}" \
        --connector "${connector}" \
        --merge_type "${merge_type}" \
	--exit
done < "${contrast_group_file}"

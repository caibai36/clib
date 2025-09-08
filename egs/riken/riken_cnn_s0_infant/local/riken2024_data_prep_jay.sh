#!/bin/bash

# Set bash to 'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands'
set -euo pipefail

# General configuration
stage=0 # Start from 0 if you need to start from data preparation
data_name=riken2024_jay_family

# Data
riken2024=/home/bin-wu/share/data/riken/riken2024

# Parse the options. (e.g., ./run.sh --stage 1)
# Note that the options should be defined as shell variables before parsing
. local/scripts/parse_options.sh || exit 1

if [ ${stage} -le 1 ]; then
    echo "Preparing wav files and audacity segments..."
    mkdir -p data/$data_name/
    rm -rf data/$data_name/wav.scp data/$data_name/seg.scp data/$data_name/seg2d.scp

    # Wav files
    for file in "$riken2024"/jay_family/seg/*.txt; do
        id=$(basename "$file" .txt)
        dir=$(dirname "$(dirname "$file")")
        echo "$id $dir/wav/$id.wav"
    done >> data/$data_name/wav.scp

    # Paths to audacity 1D segments
    for file in "$riken2024"/jay_family/seg/*.txt; do
        id=$(basename "$file" .txt)
        dir=$(dirname "$(dirname "$file")")
        echo "$id $dir/seg/$id.txt"
    done >> data/$data_name/seg.scp

    # Audacity 2D segments
    for file in "$riken2024"/jay_family/seg/*.txt; do
        id=$(basename "$file" .txt)
        dir=$(dirname "$(dirname "$file")")
        echo "$id $dir/seg2d/$id.txt"
    done >> data/$data_name/seg2d.scp

    python local/scripts/data2info.py \
        --data data/$data_name \
        --scps seg:data/$data_name/seg.scp seg2d:data/$data_name/seg2d.scp \
        > data/$data_name/info.json
fi

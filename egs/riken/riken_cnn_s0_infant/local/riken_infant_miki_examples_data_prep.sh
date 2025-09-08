#!/bin/bash

# Set bash to 'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands'
set -euo pipefail

# General configuration
stage=0 # Start from 0 if you need to start from data preparation
data_name=infant_miki_examples

# Data
infant_miki_examples=/data/share/bin-wu/data/human/speech/examples/infant_miki_examples/HomeRecording01

# Parse the options. (e.g., ./run.sh --stage 1)
# Note that the options should be defined as shell variables before parsing
. local/scripts/parse_options.sh || exit 1

if [ ${stage} -le 1 ]; then
    echo "Preparing wav files and audacity segments..."
    mkdir -p data/$data_name/
    rm -rf data/$data_name/wav.scp data/$data_name/seg.scp

    # Wav files
    for file in "$infant_miki_examples"/*.wav; do
        id=$(basename "$file" .wav)
        dir=$(dirname "$file")
        echo "$id $dir/$id.wav"
    done >> data/$data_name/wav.scp

    # Paths to audacity 1D segments
    for file in "$infant_miki_examples"/*.wav; do
        id=$(basename "$file" .wav)
        dir=$(dirname "$file")
        echo "$id $dir/$id.txt"
    done >> data/$data_name/seg.scp
    
    python local/scripts/data2info.py \
        --data data/$data_name \
        --scps seg:data/$data_name/seg.scp \
        > data/$data_name/info.json
fi

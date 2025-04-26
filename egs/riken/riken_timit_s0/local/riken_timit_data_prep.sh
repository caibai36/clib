#!/bin/bash

# Set bash to 'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands'
set -euo pipefail

# General configuration
stage=0 # Start from 0 if you need to start from data preparation
data_name=timit

# Data
processed_timit=/data/share/bin-wu/data/human/speech/timit/processsing/timit

# Parse the options. (e.g., ./run.sh --stage 1)
# Note that the options should be defined as shell variables before parsing
. local/scripts/parse_options.sh || exit 1

if [ ${stage} -le 1 ]; then
    echo "Preparing TIMIT data files (phone, word, text, wav)..."
    mkdir -p data/$data_name/

    # Clean up existing files if they exist
    rm -f data/$data_name/wav.scp data/$data_name/phone.scp data/$data_name/word.scp data/$data_name/text.scp data/$data_name/seg.scp

    # Create wav.scp file with paths to wav files
    echo "Creating wav.scp..."
    for file in "$processed_timit"/wav/*.wav; do
        id=$(basename "$file" .wav)
        echo "$id $file"
    done >> data/$data_name/wav.scp

    # Create phone.scp file with paths to phone transcription files
    echo "Creating phone.scp..."
    for file in "$processed_timit"/phone/*.PHN; do
        id=$(basename "$file" .PHN)
        echo "$id $file"
    done >> data/$data_name/phone.scp

    # Create word.scp file with paths to word transcription files
    echo "Creating word.scp..."
    for file in "$processed_timit"/word/*.WRD; do
        id=$(basename "$file" .WRD)
        echo "$id $file"
    done >> data/$data_name/word.scp

    # Create text.scp file with text content from TXT files
    echo "Creating text.scp with actual text content..."
    for file in "$processed_timit"/text/*.TXT; do
        id=$(basename "$file" .TXT)
        text=$(cat "$file")
        echo "$id $text"
    done >> data/$data_name/text.scp

    # Copy phone.scp to seg.scp
    echo "Creating seg.scp (copy of phone.scp)..."
    cp data/$data_name/phone.scp data/$data_name/seg.scp

    mkdir -p conf/data
    cp "$processed_timit"/division_timit.yaml conf/data/

    echo "TIMIT data preparation completed successfully"
fi

if [ ${stage} -le 2 ]; then
    # Generate info.json with the script
    echo "Generating info.json..."

    python local/scripts/data2info.py \
        --data data/$data_name \
        --scps seg:data/$data_name/seg.scp phone:data/$data_name/phone.scp word:data/$data_name/word.scp text:data/$data_name/text.scp \
        > data/$data_name/info.json
fi

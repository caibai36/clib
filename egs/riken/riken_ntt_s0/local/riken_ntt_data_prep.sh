#!/bin/bash

# Set bash to 'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands'
set -euo pipefail

# General configuration
stage=0 # Start from 0 if you need to start from data preparation
data_name=ntt_infant

# Data
# ntt_infant=data/local/ntt_infant_data
# ntt_infant=/data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/processed/ntt_infant_data
ntt_infant=/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_ntt_s0/data/local/ntt_infant_data

# Parse the options. (e.g., ./run.sh --stage 1)
# Note that the options should be defined as shell variables before parsing
. local/scripts/parse_options.sh || exit 1

if [ ${stage} -le -1 ]; then
   ./local/riken_ntt_add_kana_comment_token.sh
   python local/riken_ntt_create_dataset.py --output_dir data/local/ntt_infant_data
fi

if [ ${stage} -le 1 ]; then
    echo "Preparing ntt_infant dataset files..."
    mkdir -p data/$data_name/
    rm -rf data/$data_name/wav.scp data/$data_name/text.scp data/$data_name/comment.scp \
           data/$data_name/kana.scp data/$data_name/kana_token.scp data/$data_name/kana_comment_token.scp \
           data/$data_name/subset.scp data/$data_name/token.scp

    # Wav files
    for dir in "$ntt_infant"/*; do
        subject=$(basename "$dir")
        for file in "$dir"/wav/*.wav; do
            id=$(basename "$file" .wav)
            echo "$id $file"
        done
    done >> data/$data_name/wav.scp

    # Paths to text files
    for dir in "$ntt_infant"/*; do
        subject=$(basename "$dir")
        for file in "$dir"/text/*.txt; do
            id=$(basename "$file" .txt)
            echo "$id $file"
        done
    done >> data/$data_name/text.scp

    # Paths to comment files
    for dir in "$ntt_infant"/*; do
        subject=$(basename "$dir")
        for file in "$dir"/comment/*.txt; do
            id=$(basename "$file" .txt)
            echo "$id $file"
        done
    done >> data/$data_name/comment.scp

    # Paths to kana files
    for dir in "$ntt_infant"/*; do
        subject=$(basename "$dir")
        for file in "$dir"/kana/*.txt; do
            id=$(basename "$file" .txt)
            echo "$id $file"
        done
    done >> data/$data_name/kana.scp

    # Paths to kana_token files
    for dir in "$ntt_infant"/*; do
        subject=$(basename "$dir")
        for file in "$dir"/kana_token/*.txt; do
            id=$(basename "$file" .txt)
            echo "$id $file"
        done
    done >> data/$data_name/kana_token.scp

    # Paths to kana_comment_token files
    for dir in "$ntt_infant"/*; do
        subject=$(basename "$dir")
        for file in "$dir"/kana_comment_token/*.txt; do
            id=$(basename "$file" .txt)
            echo "$id $file"
        done
    done >> data/$data_name/kana_comment_token.scp

    # Subsets of the datasets (subject info)
    for dir in "$ntt_infant"/*; do
        subject=$(basename "$dir")
        for file in "$dir"/wav/*.wav; do
            id=$(basename "$file" .wav)
            echo "$id $subject"
        done
    done >> data/$data_name/subset.scp

    cp data/$data_name/kana_comment_token.scp data/$data_name/token.scp
    python local/scripts/data2info.py \
        --data data/$data_name \
        --scps text:data/$data_name/text.scp comment:data/$data_name/comment.scp \
              kana:data/$data_name/kana.scp kana_token:data/$data_name/kana_token.scp \
              kana_comment_token:data/$data_name/kana_comment_token.scp \
	      token:data/$data_name/token.scp \
              subset:data/$data_name/subset.scp \
        > data/$data_name/info.json
fi

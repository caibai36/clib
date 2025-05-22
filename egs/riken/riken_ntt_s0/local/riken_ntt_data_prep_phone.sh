#!/bin/bash

# Set bash to 'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands'
set -euo pipefail

# General configuration
stage=0 # Start from 0 if you need to start from data preparation
data_name=ntt_infant_phone

# Data
# ntt_infant_phone=/data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/processed/ntt_infant_data_force_alignment_phone
ntt_infant_phone=/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_ntt_s0/exp/out/all/fixed

# Parse the options. (e.g., ./run.sh --stage 1)
# Note that the options should be defined as shell variables before parsing
. local/scripts/parse_options.sh || exit 1

if [ ${stage} -le -1 ]; then
    # data/ntt_infant/ comes from riken_ntt_data_prep.sh
    cp -r data/ntt_infant/ data/ntt_infant_phone
fi

if [ ${stage} -le 1 ]; then
    echo "Preparing ntt_infant dataset files..."
    mkdir -p data/$data_name/
    rm -rf data/$data_name/seg.scp

    # Seg file (Kaldi force aligned phone)
    for file in $ntt_infant_phone/*.txt; do
	echo "$(basename "$file" .txt) $file"
    done | sort -u >> data/$data_name/seg.scp

    cp data/$data_name/kana_comment_token.scp data/$data_name/token.scp
    python local/scripts/data2info.py \
        --data data/$data_name \
        --scps text:data/$data_name/text.scp comment:data/$data_name/comment.scp \
              kana:data/$data_name/kana.scp kana_token:data/$data_name/kana_token.scp \
              kana_comment_token:data/$data_name/kana_comment_token.scp \
	      token:data/$data_name/token.scp \
              subset:data/$data_name/subset.scp \
	      seg:data/$data_name/seg.scp \
        > data/$data_name/info.json
fi

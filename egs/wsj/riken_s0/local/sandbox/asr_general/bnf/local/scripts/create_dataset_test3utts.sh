#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euox pipefail

# Prepare some basic config files of kaldi.
bash local/kaldi_conf.sh
# Note: cmd.sh, path.sh are created by kaldi_conf.sh
. cmd.sh
. path.sh

stage=1
dataset=test3utts
# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

if [ $stage -le 1 ]; then
    date
    echo "Data preparation..."
    mkdir -p data/$dataset
    # head -3 /project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/zr19/s2/data/test_en/wav.scp > data/$dataset/wav.scp
    # head -3 /project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/zr19/s2/data/test_en/utt2spk > data/$dataset/utt2spk
    echo "S002_0006942614 /project/nakamura-lab08/Work/bin-wu/share/data/zr19/db/english/test/S002_0006942614.wav" > data/$dataset/wav.scp
    echo "S002_0026904189 /project/nakamura-lab08/Work/bin-wu/share/data/zr19/db/english/test/S002_0026904189.wav" >> data/$dataset/wav.scp
    echo "S002_0028048675 /project/nakamura-lab08/Work/bin-wu/share/data/zr19/db/english/test/S002_0028048675.wav" >> data/$dataset/wav.scp
    echo "S002_0006942614 S002" > data/$dataset/utt2spk
    echo "S002_0026904189 S002" >> data/$dataset/utt2spk
    echo "S002_0028048675 S002" >> data/$dataset/utt2spk
    utils/utt2spk_to_spk2utt.pl data/${dataset}/utt2spk > data/${dataset}/spk2utt
    date
fi

if [ $stage -le 2 ]; then
    date
    echo "Feature extraction..."
    ./local/scripts/feat_extract.sh --feat_nj 1 --dataset $dataset --cmvn true --vtln false --delta_order 0 --mfcc_conf conf/mfcc.conf
    date
fi

if [ $stage -le 3 ]; then
    date
    echo "Make json files for $dataset..."
    local/scripts/data2json.sh --feat data/$dataset/feats.scp --output-utts-json data/$dataset/utts.json data/$dataset
    local/scripts/data2json.sh --feat data/$dataset/raw.scp --output-utts-json data/$dataset/utts_raw.json data/$dataset
    date
fi

#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euox pipefail

# Prepare some basic config files of kaldi.
bash local/kaldi_conf.sh
# Note: cmd.sh, path.sh are created by kaldi_conf.sh
. cmd.sh
. path.sh

stage=3
# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

# Database for zerospeech 2019
db=/project/nakamura-lab08/Work/bin-wu/share/data/zr19/db # CHECKME downloaded by local/download.sh

if [ $stage -le 1 ]; then
    date
    echo "Data preparation..."
    ./local/zr19_data_prep.sh --dataset test_en --audio_dir $db/english/test --vad_file $db/english/vads.txt
    ./local/zr19_data_prep.sh --dataset train_en --audio_dir $db/english/train/unit --vad_file $db/english/vads.txt
    date
fi

if [ $stage -le 2 ]; then
    date
    echo "Feature extraction..."
    [ ! -d data/test_en_hires]  && cp -r data/test_en data/test_en_hires
    [ ! -d data/train_en_hires ] && cp -r data/train_en data/train_en_hires
    [ ! -d data/test_en_hires80 ] && cp -r data/test_en data/test_en_hires80
    [ ! -d data/train_en_hires80 ] && cp -r data/train_en data/train_en_hires80

    ./local/script/feat_extract.sh --dataset test_en --cmvn true --vtln true --delta_order 2 --mfcc_conf conf/mfcc.conf --min_segment_length 0.001 # 3 hours
    ./local/script/feat_extract.sh --dataset train_en --cmvn true --vtln true --delta_order 2 --mfcc_conf conf/mfcc.conf --min_segment_length 0.001 # 2 hours
    ./local/script/feat_extract.sh --dataset test_en_hires --cmvn true --vtln false --delta_order 0 --mfcc_conf conf/mfcc_hires.conf --min_segment_length 0.001 # 4 min
    ./local/script/feat_extract.sh --dataset train_en_hires --cmvn true --vtln false --delta_order 0 --mfcc_conf conf/mfcc_hires.conf --min_segment_length 0.001 # 5 min
    ./local/script/feat_extract.sh --dataset test_en_hires80 --cmvn true --vtln false --delta_order 0 --mfcc_conf conf/mfcc_hires80.conf --min_segment_length 0.001 # 4 min
    ./local/script/feat_extract.sh --dataset train_en_hires80 --cmvn true --vtln false --delta_order 0 --mfcc_conf conf/mfcc_hires80.conf --min_segment_length 0.001 # 5 min
    date
fi

if [ $stage -le 3 ]; then
    date
    for dataset in $(ls data/); do
	echo "Make json files for $dataset..."
	local/script/data2json.sh --feat data/$dataset/feats.scp --output-utts-json data/$dataset/utts.json data/$dataset
	local/script/data2json.sh --feat data/$dataset/raw.scp --output-utts-json data/$dataset/utts_raw.json data/$dataset
    done
    date
fi

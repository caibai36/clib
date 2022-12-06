#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail

# Prepare some basic config files of kaldi.
./local/kaldi_conf.sh # CHECKME
# Note: cmd.sh, path.sh are created by kaldi_conf.sh
. cmd.sh
. path.sh

# general configuration
stage=1  # start from 0 if you need to start from data preparation

mfcc_config=conf/mfcc.conf # 13 dimensional mfcc feature of kaldi default setting for timit
mfcc_hires=conf/mfcc_hires.conf # 40 dimensional mfcc feature of kaldi default setting for wsj
mel_config=clib/conf/feat/taco_mel_f80.json # 80 dimensional mel feature of tacotron tts

mfcc39=true
mfcc40=true
mel80=true

cmvn=true

source_data_dir=data # directory containing the dataset without any feature extraction (without feature postfix e.g., 'train' (not 'train_mfcc39').) 

datasets=
# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

if [ -z "$datasets" ]; then
    echo "./local/yonden_feat_extraction.sh --datasets \"\$datasets\" [--source_data_dir \$source_data_dir (default data)] [--mfcc39 true|false] [--mfcc40 true|false] [--mel80 true|false] [--cmvn true]"
    echo "e.g., ./local/yonden_feat_extraction.sh --datasets \"\$datasets\" --source_data_dir \$source_data_dir --mfcc39 true --mfcc40 true --mel80 true"
    exit 1
fi

if [ $mfcc39 ]; then
    date
    echo "Making 39-dimensional mfcc feature (13dim+delta+deltadelta)..."

    for dataset in $datasets; do
    	x=${dataset}_mfcc39
	mkdir -p data/$x; cp -r ${source_data_dir}/${dataset}/* data/${x} # Fail to overwrite with command `cp -r data/${dataset} data/${x}'
	./local/scripts/feat_extract.sh --dataset ${x} --cmvn $cmvn --mfcc_conf $mfcc_config --delta_order 2 # 39-dim mfcc
    done

    date
fi

if [ $mfcc40 ]; then
    date
    echo "Making 40-dimensional mfcc feature..."

    for dataset in $datasets; do
    	x=${dataset}_mfcc40
	mkdir -p data/$x; cp -r ${source_data_dir}/${dataset}/* data/${x} # Fail to overwrite with command `cp -r data/${dataset} data/${x}'
	./local/scripts/feat_extract.sh --dataset ${x} --cmvn $cmvn --mfcc_conf $mfcc_hires # 40-dim mfcc
    done
    date
fi

if [ $mel80 ]; then
    date
    echo "Making 80-dimensional mel feature..."

    for dataset in $datasets; do
    	x=${dataset}_mel80
	mkdir -p data/$x; cp -r ${source_data_dir}/${dataset}/* data/${x} # Fail to overwrite with command `cp -r data/${dataset} data/${x}'
    	./local/scripts/feat_extract_taco.sh --dataset ${x} --cmvn $cmvn --mel_conf $mel_config  # 80-dim mel
    done
    date
fi

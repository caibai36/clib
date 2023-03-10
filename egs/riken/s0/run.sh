#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -uo pipefail

# # scripts
# mkdir -p local
# cp -r ../../wsj/s0/local/{kaldi_conf.sh,scripts} local
# cp -r ../../wsj/s0/clib .

# Prepare some basic config files of kaldi.
./local/kaldi_conf.sh # CHECKME
# Note: cmd.sh, path.sh are created by kaldi_conf.sh
. cmd.sh
. path.sh

# general configuration
stage=0  # start from 0 if you need to start from data preparation
sed -i '/high-freq/ d' conf/mfcc_hires.conf # remove the high frequency cut for marmoset data
sed -i '/high-freq/ d' conf/mfcc_hires80.conf # remove the high frequency cut for marmoset data
echo -e "--channel=1\n--sample-frequency=48000" >> conf/mfcc.conf
echo -e "--channel=1\n--sample-frequency=48000" >> conf/mfcc_hires.conf
echo -e "--channel=1\n--sample-frequency=48000" >> conf/mfcc_hires80.conf
sed -i 's/use-energy=false/use-energy=true/g' conf/mfcc.conf
sed -i 's/use-energy=false/use-energy=true/g' conf/mfcc_hires.conf
sed -i 's/use-energy=false/use-energy=true/g' conf/mfcc_hires80.conf
mfcc_config=conf/mfcc_hires.conf  # use the high-resolution mfcc for training the neural network;
                             # 40 dimensional mfcc feature used by Google and kaldi wsj network training.

# increase num_freq to 2048 because sampling rate increases. (see: http://librosa.org/doc/main/generated/librosa.stft.html)  
sed -e 's/"sample_rate":16000/"sample_rate":48000/g' -e 's/"num_freq":1025/"num_freq":2048/g' clib/conf/feat/taco_mel_f80.json > clib/conf/feat/taco_mel_f80_sr48000.json
sed -e 's/"sample_rate":16000/"sample_rate":48000/g' -e 's/"num_freq":1025/"num_freq":2048/g' -e 's/"num_mels":80/"num_mels":13/g' clib/conf/feat/taco_mel_f80.json > clib/conf/feat/taco_mel_f13_sr48000.json
mel_config=clib/conf/feat/taco_mel_f80_sr48000.json # 80 dimensional mel feature of tacotron tts

# Data and model options
feat=mel80 # or mfcc40

# Data
dataset=riken_sample
riken_sample=/project/nakamura-lab08/Work/bin-wu/workspace/datasets/riken/samples

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

if [ ${stage} -le 1 ]; then
    date
    echo "Data preparation..."
    ./local/riken_sample_data_prep.sh --dataset $dataset --corpus $riken_sample
    date
fi

if [ ${stage} -le 2 ]; then
    date
    echo "Making 40-dimensional mfcc feature..."

    for dataset in riken_sample; do
	./local/scripts/feat_extract.sh --dataset ${dataset} --cmvn true --feat_nj 1 --mfcc_conf $mfcc_config --delta_order 0 
    done
    date
fi

if [ ${stage} -le 2 ]; then
    date
    echo "Making 13-dimensional mfcc feature..."

    for dataset in riken_sample; do
	x=${dataset}_mfcc13
	mkdir -p data/${x}
    	cp -r data/${dataset}/* data/${x} # Fail to overwrite with command `cp -r data/${dataset} data/${x}'
	./local/scripts/feat_extract.sh --dataset ${x} --cmvn true --feat_nj 1 --mfcc_conf conf/mfcc.conf # 13-dim mfcc
    done
    date
fi

if [ ${stage} -le 2 ]; then
    date
    echo "Making 39-dimensional mfcc feature..."
    for dataset in riken_sample; do
	x=${dataset}_mfcc39
	mkdir -p data/${x}
	cp -r data/${dataset}/* data/${x} # Fail to overwrite with command `cp -r data/${dataset} data/${x}'
	./local/scripts/feat_extract.sh --dataset ${x} --cmvn true --feat_nj 1 --mfcc_conf conf/mfcc.conf --delta_order 2 # 39-dim mfcc
    done
    date
fi

if [ ${stage} -le 2 ]; then
    date
    echo "Making 40-dimensional mfcc feature..."
    for dataset in riken_sample; do
	x=${dataset}_mfcc40
	mkdir -p data/${x}
	cp -r data/${dataset}/* data/${x} # Fail to overwrite with command `cp -r data/${dataset} data/${x}'
	./local/scripts/feat_extract.sh --dataset ${x} --cmvn true --feat_nj 1 --mfcc_conf conf/mfcc_hires.conf --delta_order 0
    done
    date
fi

if [ ${stage} -le 2 ]; then
    date
    echo "Making 80-dimensional mfcc feature..."
    for dataset in riken_sample; do
	x=${dataset}_mfcc80
	mkdir -p data/${x}
	cp -r data/${dataset}/* data/${x} # Fail to overwrite with command `cp -r data/${dataset} data/${x}'
	./local/scripts/feat_extract.sh --dataset ${x} --cmvn true --feat_nj 1 --mfcc_conf conf/mfcc_hires80.conf --delta_order 2 # 39-dim mfcc
    done
    date
fi

if [ ${stage} -le 3 ]; then
    date
    echo "Making 13-dimensional mel feature..."

    for dataset in riken_sample; do
	x=${dataset}_mel13
	mkdir -p data/${x}
    	cp -r data/${dataset}/* data/${x} # Fail to overwrite with command `cp -r data/${dataset} data/${x}'
	./local/scripts/feat_extract_taco.sh --dataset ${x} --cmvn true --feat_nj 1 --mel_conf clib/conf/feat/taco_mel_f13_sr48000.json  # 13-dim mel
    done
    date
fi

if [ ${stage} -le 3 ]; then
    date
    echo "Making 39-dimensional mel feature..."
    for dataset in riken_sample; do
	x=${dataset}_mel39
	mkdir -p data/${x}
	cp -r data/${dataset}/* data/${x} # Fail to overwrite with command `cp -r data/${dataset} data/${x}'
	./local/scripts/feat_extract_taco.sh --dataset ${x} --cmvn true --feat_nj 1 --mel_conf clib/conf/feat/taco_mel_f13_sr48000.json --delta_order 2 # 39-dim mel
    done
    date
fi

if [ ${stage} -le 3 ]; then
    date
    echo "Making 80-dimensional mel feature..."

    for dataset in riken_sample; do
    	x=${dataset}_mel80
	mkdir -p data/${x}
    	cp -r data/${dataset}/* data/${x} # Fail to overwrite with command `cp -r data/${dataset} data/${x}'
    	./local/scripts/feat_extract_taco.sh --dataset ${x} --cmvn true --feat_nj 1 --mel_conf $mel_config  # 80-dim mel 
    done
    date
fi

if [ ${stage} -le 3 ]; then
    date
    echo "Making 80+delta dimensional mel feature..."

    for dataset in riken_sample; do
    	x=${dataset}_mel80_delta1
	mkdir -p data/${x}
    	cp -r data/${dataset}/* data/${x} # Fail to overwrite with command `cp -r data/${dataset} data/${x}'
    	./local/scripts/feat_extract_taco.sh --dataset ${x} --cmvn true --feat_nj 1 --mel_conf $mel_config  --delta_order 1 # 80-dim mel 
    done
    date
fi

if [ ${stage} -le 3 ]; then
    date
    echo "Making 80+delta+delta dimensional mel feature..."

    for dataset in riken_sample; do
    	x=${dataset}_mel80_delta2
	mkdir -p data/${x}
    	cp -r data/${dataset}/* data/${x} # Fail to overwrite with command `cp -r data/${dataset} data/${x}'
    	./local/scripts/feat_extract_taco.sh --dataset ${x} --cmvn true --feat_nj 1 --mel_conf $mel_config  --delta_order 2 # 80-dim mel 
    done
    date
fi

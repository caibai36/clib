#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail

stage=1
dataset=riken_sample
corpus=/project/nakamura-lab08/Work/bin-wu/workspace/datasets/riken/samples
vad_file=

. utils/parse_options.sh || exit 1 # eg. ./run.sh --stage 1

if [ -z $dataset ] || [ -z $corpus ]; then
    echo "./local/riken_sample_data_prep.sh --dataset \$dataset --corpus \$corpus"
    exit 1
fi

if [ $stage -le 1 ]; then
    echo "Data preparation for dataset ${dataset}..."
    mkdir -p data/${dataset}

    echo "Preparing wav.scp..."
    # echo "1100F_0124_2017 sox -v 0.99 $corpus/1100F_0124_2017_sox_uncompressed.wav -t wav -r 48000 -b 16 - |" > data/${dataset}/wav.scp
    # sox -v 0.99 1100F_0124_2017.wav -t wav -r 48000 -b 16 - > 1100F_0124_2017_sox_uncompressed.wav
    echo "1100F_0124_2017 $corpus/1100F_0124_2017_sox_uncompressed.wav" > data/${dataset}/wav.scp
    echo "1100F_0124_2017_0s_10s $corpus/1100F_0124_2017_0s_10s_sox_uncompressed.wav" >> data/${dataset}/wav.scp
    echo "1100F_0124_2017_0s_30s $corpus/1100F_0124_2017_0s_30s_sox_uncompressed.wav" >> data/${dataset}/wav.scp
    # echo "pair10_animal1_together_hear_1h10min_1065s_1075s $corpus/pair10_animal1_together_hear_1h10min_1065s_1075s.wav" >> data/${dataset}/wav.scp

    
    echo "Preparing utt2spk and spk2utt..."
    echo "1100F_0124_2017 1100F_0124_2017" > data/${dataset}/utt2spk
    echo "1100F_0124_2017_0s_10s 1100F_0124_2017_0s_10s" >> data/${dataset}/utt2spk
    echo "1100F_0124_2017_0s_30s 1100F_0124_2017_0s_30s" >> data/${dataset}/utt2spk
    # echo "pair10_animal1_together_hear_1h10min_1065s_1075s pair10_animal1_together_hear_1h10min_1065s_1075s" >> data/${dataset}/utt2spk
    utils/utt2spk_to_spk2utt.pl data/${dataset}/utt2spk > data/${dataset}/spk2utt

    echo "Preparing segment..."

    echo "Preparing text..."
fi

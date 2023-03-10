#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -uo pipefail

# Prepare some basic config files of kaldi.
./local/kaldi_conf.sh # CHECKME
# Note: cmd.sh, path.sh are created by kaldi_conf.sh
. cmd.sh
. path.sh

# general configuration
stage=8  # start from 0 if you need to start from data preparation
mfcc_config=conf/mfcc_hires.conf  # use the high-resolution mfcc for training the neurnal network;
                             # 40 dimensional mfcc feature used by Google and kaldi wsj network training.
mel_config=clib/conf/feat/taco_mel_f80.json # 80 dimensional mel feature of tacotron tts

# The official dataset of 'landman2020close - Close-range vocal interaction in the common marmoset (Callithrix jacchus)' from
# https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0227392
marmoset_mit=/project/nakamura-lab08/Work/bin-wu/share/data/marmoset_mit

if [ ${stage} -le 1 ]; then
    date
    echo "Data preparation..."

    # local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?
    # local/marmoset_mit_data_prep.sh $marmoset_mit || exit 1;
    date
fi

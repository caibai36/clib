#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail

# Prepare some basic config files of kaldi.
sh local/kaldi_conf.sh
# Note: cmd.sh, path.sh are created by kaldi_conf.sh
. cmd.sh
. path.sh

# general configuration
stage=2  # start from 0 if you need to start from data preparation
mfcc_config=conf/mfcc_hires.conf  # use the high-resolution mfcc for training the neurnal network;
                             # 40 dimensional mfcc feature used by Google and kaldi wsj network training.

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

# data
# wsj0=/project/nakamura-lab01/Share/Corpora/Speech/en/WSJ/wsj0
# wsj1=/project/nakamura-lab01/Share/Corpora/Speech/en/WSJ/wsj1
wsj=/project/nakamura-lab01/Share/Corpora/Speech/en/WSJ/

if [ $stage -le 0 ]; then
    date
    echo "Stage 0: Data Preparation"
    # local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?
    local/cstr_wsj_data_prep.sh $wsj || exit 1;
    local/wsj_format_data.sh || exit 1;
    date
fi

# Directory contains mfcc features and cmvn statistics.
mfcc_dir=mfcc
if [ $stage -le 1 ]; then
    date
    echo "Stage 1: Feature Extraction"
    for x in test_eval92 test_eval93 test_dev93 train_si284; do
	# Do Mel-frequency cepstral coefficients (mfcc) feature extraction.
	# We use the 40-dim mfcc by kaldi neural network training default.
	steps/make_mfcc.sh --cmd "$train_cmd" --nj 20 \
			   --mfcc-config $mfcc_config --write_utt2num_frames true \
			   data/$x exp/make_mfcc/$x $mfcc_dir/$x || exit 1
	# Compute Cepstral mean and variance normalization (cmvn) per speaker instead of globally.
	# (See discussion at https://groups.google.com/forum/#!msg/kaldi-help/2Cw_6mZlquQ/HeTJPcv5CAAJ)
	steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfcc_dir/$x || exit 1;
	# Fixing data format.
	utils/fix_data_dir.sh data/$x || exit 1 # remove segments with problems
    done

    utils/subset_data_dir.sh --first data/train_si284 7138 data/train_si84 || exit 1

    # Now make subset with the shortest 2k utterances from si-84.
    utils/subset_data_dir.sh --shortest data/train_si84 2000 data/train_si84_2kshort || exit 1;
    date
fi

if [ $stage -le 2 ]; then
    date
    echo "Dump Features after CMVN"
    for x in test_eval92 test_eval93 test_dev93 train_si284 train_si84 train_si84_2kshort; do
	if [ -f data/$x/raw.scp ]; then
	    cp data/$x/raw.scp data/$x/feats.scp # make sure feats.scp store paths of the raw mfcc ark.
	else
	    cp data/$x/feats.scp data/$x/raw.scp # store the script file of the raw mfcc before CMVN;
	fi

	# Dump the features after CMVN.
	cutils/make_cmvn.sh data/$x mfcc/$x
    done
    date
fi

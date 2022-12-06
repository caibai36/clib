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
stage=1  # start from 0 if you need to start from data preparation

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

# data
yonden=/localwork/asrwork/yonden/data
segmented_recordings=/project/nakamura-lab08/Work/bin-wu/workspace/datasets/yonden/20220906 # directory that outputs segmented recordings according to wav.scp and segments

if [ ${stage} -le 1 ]; then
    date
    echo "Data preparation..."

    # Start from the stage 2, skipping the stage 1 of the segmentation of long recordings.
    local/yonden_data_prep_all_3to28.sh --stage 2 --corpus $yonden --segmented_recordings $segmented_recordings --dir data/local/all_3to28
    date
fi

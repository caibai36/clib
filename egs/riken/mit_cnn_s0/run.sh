#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -uo pipefail

# general configuration
stage=0  # start from 0 if you need to start from data preparation

# Data
dataset_name=mit_sample # see https://marmosetbehavior.mit.edu
mit_sample=/data/share/bin-wu/data/marmoset/vocalization/marmoset_mit_cnn/original/Wave_files/

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. local/scripts/parse_options.sh || exit 1

if [ ${stage} -le 1 ]; then
    date
    echo "Data preparation..."
    ./local/mit_sample_data_prep.sh --mit_sample $mit_sample
    date
fi

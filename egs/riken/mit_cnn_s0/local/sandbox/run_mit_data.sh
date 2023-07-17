#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -uo pipefail

# general configuration
stage=2  # start from 0 if you need to start from data preparation

# Data
dataset_name=mit_data # see dataset shared by the paper of 'Close range vocal interaction in the common marmoset (Callithrix Jacchus)'
mit_data=/data/share/bin-wu/data/marmoset/vocalization/marmoset_mit

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. local/scripts/parse_options.sh || exit 1

if [ ${stage} -le 1 ]; then
    date
    echo "Data preparation..."
    ./local/mit_data_data_prep.sh --mit_data $mit_data --stage 0
    date
fi

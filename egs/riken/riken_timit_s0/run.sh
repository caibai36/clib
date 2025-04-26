###################################################################################
# File: run.sh
####################################################################################

#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -uo pipefail

# Prepare some basic config files of kaldi.
./local/kaldi_conf.sh # CHECKME
# Note: cmd.sh, path.sh are created by kaldi_conf.sh
. cmd.sh
. path.sh
export PATH=/home/bin-wu/.local/miniconda3/envs/mlp/bin:$PATH

# general configuration
stage=8  # start from 0 if you need to start from data preparation
. local/scripts/parse_options.sh || exit 1

# the directory of timit
# timit=/project/nakamura-lab01/Share/Corpora/Speech/en/TIMIT/TIMIT # CHECKME
timit=/data/share/bin-wu/data/human/speech/timit/TIMIT/TIMIT
processed_timit=/data/share/bin-wu/data/human/speech/timit/processsing/timit

if [ ${stage} -le 1 ]; then
    date
    echo "Data preparation..."

    # local/riken_timit_kaldi_data_prep.sh $timit || exit 1
    ./local/riken_create_timit_dataset.sh $timit $processed_timit || exit 1
    ./local/riken_timit_data_prep.sh --data_name timit --processed_timit $processed_timit || exit 1
    ./local/riken_timit_create_dict.sh
    date
fi

# Use local/sandbox/run_riken_timit.sh for training
# use local/sandbox/riken_cnn_audio2seg_highres.py for evaluation

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
stage=0        # start from 0 if you need to start from data preparation

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

# data
# wsj0=/project/nakamura-lab01/Share/Corpora/Speech/en/WSJ/wsj0
# wsj1=/project/nakamura-lab01/Share/Corpora/Speech/en/WSJ/wsj1
wsj=/project/nakamura-lab01/Share/Corpora/Speech/en/WSJ/

if [ ${stage} -le 0 ]; then
    date
    echo "stage 0: Data preparation"
    # local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?
    local/cstr_wsj_data_prep.sh $wsj
    local/wsj_format_data.sh
    date
fi

#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -uo pipefail

# general configuration
stage=0  # start from 0 if you need to start from data preparation

# Data
mit_sample=/data/share/bin-wu/data/marmoset/vocalization/marmoset_mit_cnn/original/Wave_files

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. local/scripts/parse_options.sh || exit 1

if [ ${stage} -le 1 ]; then
    echo "Preparing wav.scp of wav files and aud.scp of audacity labels..."
    mkdir -p data/mit_sample/
    rm data/mit_sample/wav.scp data/mit_sample/aud.scp

    for file in $mit_sample/*/*wav; do
	echo $(basename $file _aligned.wav | sed -e s'/20150814_//g' -e  s'/_20161219//g') $file >> data/mit_sample/wav.scp
    done

    for file in $mit_sample/*/*{Cricket,Enid,Sailor,Setta,Athos,Porthos}*txt; do
	echo $(basename $file .txt | sed 's/_20161219//g') $file >> data/mit_sample/aud.scp
    done

    python local/scripts/data2info.py --data data/mit_sample --scps aud:data/mit_sample/aud.scp > data/mit_sample/info.json
fi

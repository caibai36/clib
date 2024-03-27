#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -uo pipefail

# general conf
stage=0  # start from 0 if you need to start from data preparation

# data conf
dataset_name="mit_data"

# dataset
mit_data=/data/share/bin-wu/data/marmoset/vocalization/marmoset_mit

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. local/scripts/parse_options.sh || exit 1

if [ ${stage} -le 1 ]; then
    echo "Preparing wav.scp of wav files and aud.scp of audacity labels..."
    mkdir -p data/$dataset_name/
    rm -rf data/$dataset_name/wav.scp data/$dataset_name/aud.scp

    for file in $mit_data/data/pair*/*wav; do
	echo $(basename $file .wav | sed -e 's/animal/a/g' -e 's/together/toget/g' | sed -r 's/^pair([0-9]+)_/p\1/g') $file >> data/$dataset_name/wav.scp
    done

    for file in $mit_data/processed/audacity/audacity_labels/*.txt; do
	echo $(basename $file .txt) $file >> data/$dataset_name/aud.scp
    done

    python local/scripts/data2info.py --data data/$dataset_name --scps aud:data/$dataset_name/aud.scp > data/$dataset_name/info.json
fi

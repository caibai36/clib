#!/bin/bash

# Set bash to 'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands'
set -uo pipefail

# General configuration
stage=0 # Start from 0 if you need to start from data preparation
data_name="mit_sample"

# "train_dev" is all uttids for training and development sets. Uttids are a sequence of animal pairs.
# e.g., "A B C D", where "A" and "B" are the first pair; "C" and "D" are the second pair.
# "dev_pair_ind" is the index of the pair in the train_dev array taken as the development set.
# Taking value 1 means take the second pair as the development set.
# e.g., ["A", "B", "C", "D", "E", "F"] would take ["C", "D"] as the development set
# and the remaining pairs of ["A", "B", "E", "F"] as the training set.
train_dev="Cricket Enid Setta Sailor"
dev_pair_ind=1

test_wav_uttids="Athos Porthos" # A sequence of pairs

# Data
mit_sample=/data/share/bin-wu/data/marmoset/vocalization/marmoset_mit_cnn/original/Wave_files

# Parse the options. (e.g., ./run.sh --stage 1)
# Note that the options should be defined as shell variables before parsing
. local/scripts/parse_options.sh || exit 1

if [ ${stage} -le 1 ]; then
    echo "Preparing wav.scp of wav files and aud.scp of audacity labels..."
    mkdir -p data/$data_name/
    rm -rf data/$data_name/wav.scp data/$data_name/aud.scp

    find "$mit_sample" -name "*wav" | while read -r file; do
        basename=$(basename "$file" _aligned.wav | sed -e 's/20150814_//g' -e 's/_20161219//g')
        echo "$basename $file" >> data/$data_name/wav.scp
    done

    for file in $mit_sample/*/*{Cricket,Enid,Sailor,Setta,Athos,Porthos}*txt; do
	echo $file
        basename=$(basename "$file" .txt | sed 's/_20161219//g')
        echo "$basename $file" >> data/$data_name/aud.scp
    done

    python local/scripts/data2info.py --data data/$data_name --scps seg:data/$data_name/aud.scp > data/$data_name/info.json
fi

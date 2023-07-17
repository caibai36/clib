#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -uo pipefail

# general configuration
stage=0  # start from 0 if you need to start from data preparation

dataset_name="mit_data"

# "train_dev" is all uttids for training and development sets. Uttids are a sequence of animal pairs.
# e.g., "A B C D", where "A" and "B" are the first pair; "C" and "D" are the second pair.
# "dev_pair_ind" is the indices of pairs in train_dev array taken as the development set. Taking value 1 means take the second pair as the development set.
# e.g., ["A", "B", "C", "D", "E", "F"] would take ["C", "D"] as the development set and the remaining pairs of ["A", "B", "E", "F"] as the training set.
train_dev="p2a1_toget p2a2_toget p3a1_toget p3a2_toget p4a1_toget p4a2_toget p5a1_toget p5a2_toget p6a1_toget p6a2_toget p7a1_toget p7a2_toget p8a1_toget p8a2_toget p9a1_toget p9a2_toget p10a1_toget p10a2_toget"
dev_pair_ind=0
test="p1a1_toget p1a2_toget"

# Data
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

if [ ${stage} -le 2 ]; then
    date
    echo "Preparing spectra and labels for training and development sets..."
    python local/mit_cnn_data_prep.py --info_json data/$dataset_name/info.json \
	   --train_dev  $train_dev \
	   --dev_pair_ind $dev_pair_ind \
	   --out_dir exp/data/$dataset_name
    date
fi

if [ ${stage} -le 3 ]; then
    date
    echo "Converting an onehot target into two onehot targets of an animal pair for training and development sets..."
    python local/mit_cnn_target_norm_single2.py --train_target_multi exp/data/$dataset_name/train_target_multi \
	   --dev_target_multi exp/data/$dataset_name/dev_target_multi \
	   --train_target_single1 exp/data/$dataset_name/train_target_single1 \
	   --train_target_single2 exp/data/$dataset_name/train_target_single2 \
	   --dev_target_single1 exp/data/$dataset_name/dev_target_single1 \
	   --dev_target_single2 exp/data/$dataset_name/dev_target_single2
    date
fi

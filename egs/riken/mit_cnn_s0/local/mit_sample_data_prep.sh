#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -uo pipefail

# general configuration
stage=0  # start from 0 if you need to start from data preparation

dataset_name="mit_sample"

# "train_dev" is all uttids for training and development sets. Uttids are a sequence of animal pairs.
# e.g., "A B C D", where "A" and "B" are the first pair; "C" and "D" are the second pair.
# "dev_pair_ind" is the indices of pairs in train_dev array taken as the development set. Taking value 1 means take the second pair as the development set.
# e.g., ["A", "B", "C", "D", "E", "F"] would take ["C", "D"] as the development set and the remaining pairs of ["A", "B", "E", "F"] as the training set.
train_dev="Cricket Enid Setta Sailor"
dev_pair_ind=1
test_wav_uttids="Athos Porthos" # a sequence of pairs

# Data
mit_sample=/data/share/bin-wu/data/marmoset/vocalization/marmoset_mit_cnn/original/Wave_files

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. local/scripts/parse_options.sh || exit 1

if [ ${stage} -le 1 ]; then
    echo "Preparing wav.scp of wav files and aud.scp of audacity labels..."
    mkdir -p data/$dataset_name/
    rm -rf data/$dataset_name/wav.scp data/$dataset_name/aud.scp

    for file in $mit_sample/*/*wav; do
	echo $(basename $file _aligned.wav | sed -e s'/20150814_//g' -e  s'/_20161219//g') $file >> data/$dataset_name/wav.scp
    done

    for file in $mit_sample/*/*{Cricket,Enid,Sailor,Setta,Athos,Porthos}*txt; do
	echo $(basename $file .txt | sed 's/_20161219//g') $file >> data/$dataset_name/aud.scp
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

if [ ${stage} -le 4 ]; then
    date
    echo "Create 2500ms spectral segments to prepare the inputs of test sets..."
    python local/mit_cnn_wav_into_test_2500_raw.py --info_json data/$dataset_name/info.json \
	   --test_wav_uttids  $test_wav_uttids \
	   --out_dir exp/data/$dataset_name
    date
fi

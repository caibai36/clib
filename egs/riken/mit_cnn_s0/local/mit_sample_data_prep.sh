#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -uo pipefail

# general configuration
stage=0  # start from 0 if you need to start from data preparation

dataset_name="mit_sample"
train_dev="Cricket Enid Setta Sailor"
dev_pair_ind=1

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
	   --train_dev  $train_dev\
	   --dev_pair_ind $dev_pair_ind \
	   --out_dir exp/data/$dataset_name
    date
fi

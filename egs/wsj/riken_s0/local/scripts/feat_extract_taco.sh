#!/bin/bash

# Implemented by bin-wu at 23:02 on 3 April 2020

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail

. path.sh
. cmd.sh

stage=1
dataset= # name of dataset
mel_conf=conf/feat/taco_mel_f80.json
cmvn=true
delta_order=0 # if feat+delta+delta then delta_order=2

# min_segment_length=0.1 # Minimum segment length in seconds (reject shorter segments) (float, default = 0.1)
feat_nj=4 # num_jobs for feature extraction

echo "$0 $@"  # Print the command line for logging

. utils/parse_options.sh || exit 1 # eg. ./run.sh --stage 1

if [[ -z $dataset ]]; then
    echo -e "$0"
    echo
    echo -e "Feature extraction e.g. wav->[raw_mel]->[cmvn]->[add_delta]"
    echo -e "Usage: $0 --dataset \$dataset"
    echo
    echo "Options: "
    echo -e "\t--dataset\tname of dataset, dataset located at data/\$dataset (required)"
    echo -e "\t--mel_conf\te.g., conf/feat/taco_mel_f80.json (default)"
    echo -e "\t--cmvn\tdo mean and variance normalizaiton (default true)"
    echo -e "\t--delta_order\te.g., if mel+delta+delta, then delta_order=2 (default 0)"
    echo -e "\t--feat_nj\tthe number of jobs for feature extraction (default 4)"
    echo
    echo -e "feats.scp contains final normalized features; raw.scp contains raw features"
    echo
    exit 1
fi

# Get dimension of mel from the configuration file
num_ceps=$(awk '/num_mels/ {print gensub(/^.*:([0-9]+).*$/, "\\1", "g", $0)}' $mel_conf)
num_ceps_default=80;
mel_dim=$(if [ ! -z  $num_ceps ]; then echo $num_ceps; else echo $num_ceps_default; fi)
echo "mel_dim: $mel_dim"
echo "mel_delta_order: $delta_order"

# Directory contains mel features and cmvn statistics.
mel_dir=feat/${dataset}_mel${mel_dim}_delta${delta_order}
mkdir -p $mel_dir

if [ $stage -le 1 ]; then
    echo "---------------------------------------------------"
    echo "Mel feature extraction and compute CMVN of data set"
    echo "---------------------------------------------------"

    rm -rf data/${dataset}/{feats,raw}.scp # avoid confusion of scp files in make_cmvn.sh

    # Do Mel-frequency cepstral coefficients (mel) feature extraction.
    python3 local/scripts/make_mel_taco.py \
	   --feat_config $mel_conf \
	   --write_utt2num_frames true \
	   --data_dir data/${dataset} \
	   --feat_dir $mel_dir
    steps/compute_cmvn_stats.sh data/${dataset} exp/make_mel/${dataset} $mel_dir
    utils/fix_data_dir.sh data/${dataset} || exit 1 # Fix the data format and remove segments with problems
fi

if [ $stage -le 2 ]; then
    if $cmvn; then
	echo "---------------------------------------------------"
	echo "Dump the features after CMVN."
	echo "---------------------------------------------------"
	# Dump the features after CMVN.
	local/scripts/make_cmvn.sh --nj $feat_nj --delta_order $delta_order data/${dataset} $mel_dir
    fi
fi

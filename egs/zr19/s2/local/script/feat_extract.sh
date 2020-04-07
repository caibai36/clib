#!/bin/bash

# Implemented by bin-wu at 23:02 on 3 April 2020

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail

. path.sh
. cmd.sh

stage=1
dataset= # name of dataset
mfcc_conf=conf/mfcc.conf # conf of mfcc; default frame_shift=10 frame_length=25 # default 25
cmvn=true
vtln=false
delta_order=0 # if mfcc+delta+delta then delta_order=2

min_segment_length=0.1 # Minimum segment length in seconds (reject shorter segments) (float, default = 0.1)
num_gauss_ubm=400 # for an ubm model when training vtln (timit defaultnumGaussUBM=400)
feat_nj=10 # num_jobs for feature extraction

echo "$0 $@"  # Print the command line for logging

. utils/parse_options.sh || exit 1 # eg. ./run.sh --stage 1

if [[ -z $dataset ]]; then
    echo -e "$0"
    echo
    echo -e "Feature extraction e.g. wav->[raw_mfcc]->[vltn]->[cmvn]->[add_delta]"
    echo -e "Usage: $0 --dataset \$dataset"
    echo
    echo "Options: "
    echo -e "\t--dataset\tname of dataset, dataset located at data/\$dataset (required)"
    echo -e "\t--mfcc_conf\te.g., conf/mfcc.conf or conf/mfcc_hires.conf (default conf/mfcc.conf)"
    echo -e "\t--vtln\tdo vocal track length normalization (default false)"
    echo -e "\t--cmvn\tdo mean and variance normalizaiton (default true)"
    echo -e "\t--delta_order\te.g., if mfcc+delta+delta, then delta_order=2 (default 0)"
    echo
    echo -e "feats.scp contains final normalized features; raw.scp contains raw features"
    echo
    exit 1
fi

# Get dimension of mfcc from the configuration file
num_ceps=$(awk '/--num-ceps/ {print gensub(/--num-ceps=([0-9]+) .+$/, "\\1", "g", $0)}' $mfcc_conf);
num_ceps_default=13;
mfcc_dim=$(if [ ! -z  $num_ceps ]; then echo $num_ceps; else echo $num_ceps_default; fi)
echo "mfcc_dim: $mfcc_dim "
echo "mfcc_delta_order: $delta_order"

# Directory contains mfcc features and cmvn statistics.
mfcc_dir=feat/${dataset}_mfcc${mfcc_dim}_delta${delta_order}
# Directory contains mfcc features with lVTLN and its cmvn statistics.
mfcc_vtln_dir=feat/${dataset}_mfcc${mfcc_dim}_vtln_delta${delta_order}
mkdir -p $mfcc_dir
if $vtln; then mkdir -p $mfcc_vtln_dir; fi


if [ $stage -le 1 ]; then
    echo "---------------------------------------------------"
    echo "MFCC feature extration and compute CMVN of data set"
    echo "---------------------------------------------------"

    # Do Mel-frequency cepstral coefficients (mfcc) feature extraction.
    # Differ from step/make_mfcc.sh  (extract-segments => set --min-segment-length=0.001)
    local/script/make_mfcc.sh --mfcc-config $mfcc_conf \
			      --nj $feat_nj \
			      --write-utt2num-frames true \
			      --min_segment_length $min_segment_length \
			      data/${dataset} exp/make_mfcc/${dataset} $mfcc_dir
    steps/compute_cmvn_stats.sh data/${dataset} exp/make_mfcc/${dataset} $mfcc_dir
    utils/fix_data_dir.sh data/${dataset} || exit 1 # Fix the data format and remove segments with problems
fi

if [ $stage -le 2 ]; then
    if $vtln; then
	echo "---------------------------------------------------"
	echo "lvtln model training"
	echo "---------------------------------------------------"
	# Train the linear Vocal Tract Length Normalization (lVTLN) to get the warping factors.
	# This script does not require to start with a trained model.
	# There is another script (steps/train_lvtln.sh) that requires an initial model.
	./local/script/train_lvtln.sh $num_gauss_ubm data/${dataset} exp/${dataset}_vtln

	# Prepare the data files for vtln. We copy it because the following scripts may change the data files.
	mkdir -p data/${dataset}_vtln
	cp -r data/${dataset}/* data/${dataset}_vtln || exit 1
	cp exp/${dataset}_vtln/warps/utt2warp data/${dataset}_vtln || exit 1

	echo "---------------------------------------------------"
	echo "MFCC feature extraction with lVTLN of dataset"
	echo "---------------------------------------------------"

	# Do Mel-frequency cepstral coefficients (mfcc) feature extraction. (utt2num-frames already copied)
	local/script/make_mfcc.sh --mfcc-config $mfcc_conf \
				  --nj $feat_nj \
				  --min_segment_length $min_segment_length \
				  data/${dataset}_vtln exp/make_mfcc/${dataset}_vtln $mfcc_vtln_dir
	steps/compute_cmvn_stats.sh data/${dataset}_vtln exp/make_mfcc/${dataset}_vtln $mfcc_vtln_dir
	utils/fix_data_dir.sh data/${dataset}_vtln || exit 1 # remove segments with problems
    fi
fi

if [ $stage -le 3 ]; then
    if $cmvn; then
	echo "---------------------------------------------------"
	echo "Dump the features after CMVN."
	echo "---------------------------------------------------"
	# Dump the features after CMVN.
	local/script/make_cmvn.sh --nj $feat_nj --delta_order $delta_order data/${dataset} $mfcc_dir
	if $vtln; then
	    local/script/make_cmvn.sh --nj $feat_nj --delta_order $delta_order data/${dataset}_vtln $mfcc_vtln_dir
	fi
    fi
fi

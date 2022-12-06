#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail

# Prepare some basic config files of kaldi.
./local/kaldi_conf.sh # CHECKME
# Note: cmd.sh, path.sh are created by kaldi_conf.sh
. cmd.sh
. path.sh

# general configuration
stage=1  # start from 0 if you need to start from data preparation

dict=
token_scp_type="" # e.g. kana; the line of data/$dataset/$token_scp can be 'uttid token1 token2...' (e.g., uttid シ キ <space> シャ <space> ヒ ラ オ カ <space> <space> <period>)
non_ling_syms="" # dummy variable when sepcifying token_scp_type

# Dataset directory of different features
mfcc39=true
mfcc40=true
mel80=true

datasets=
# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

if [ -z "$datasets" ] || [ -z $dict ]; then
    echo "./local/yonden_data2json.sh --datasets \"\$datasets\" [--token_scp_type \$token_scp_type ] [--non_ling_syms \$non_ling_syms] [--source_data_dir \$source_data_dir (default data)] [--mfcc39 true|false] [--mfcc40 true|false] [--mel80 true|false] [--cmvn true]"
    echo
    echo "e.g., ./local/yonden_data2json.sh --datasets \"\$datasets\" --dict \$dict --token_scp_type kana --mfcc39 true --mfcc40 true --mel80 true"

    echo
    echo "--non_ling_syms is not needed if --token_scp_type is given."
    exit 1
fi

if [ $mfcc39 ]; then
    echo "Making json files"
    # get scp files (each line as 'uttid scp_content'), then merge them into utts.json.
    #     (eg. num_frames.scp, feat_dim.scp, num_tokens.scp, tokenid.scp, vocab_size.scp, feat.scp, token.scp and etc.)
    # If you want to add more information, just create more scp files in data2json.sh
    for dataset in $datasets; do
	x=${dataset}_mfcc39
	local/scripts/data2json.sh --feat data/$x/feats.scp \
				   --token_scp data/$x/$token_scp_type \
    				   --non-ling-syms "${non_ling_syms}" \
    				   --output-utts-json data/$x/utts.json \
    				   --output-dir-of-scps data/$x/scps \
    				   data/$x ${dict}
    done
fi

if [ $mfcc40 ]; then
    echo "Making json files"
    # get scp files (each line as 'uttid scp_content'), then merge them into utts.json.
    #     (eg. num_frames.scp, feat_dim.scp, num_tokens.scp, tokenid.scp, vocab_size.scp, feat.scp, token.scp and etc.)
    # If you want to add more information, just create more scp files in data2json.sh
    for dataset in $datasets; do
	x=${dataset}_mfcc40
	local/scripts/data2json.sh --feat data/$x/feats.scp \
				   --token_scp data/$x/$token_scp_type \
    				   --non-ling-syms "${non_ling_syms}" \
    				   --output-utts-json data/$x/utts.json \
    				   --output-dir-of-scps data/$x/scps \
    				   data/$x ${dict}
    done
fi

if [ $mel80 ]; then
    echo "Making json files"
    # get scp files (each line as 'uttid scp_content'), then merge them into utts.json.
    #     (eg. num_frames.scp, feat_dim.scp, num_tokens.scp, tokenid.scp, vocab_size.scp, feat.scp, token.scp and etc.)
    # If you want to add more information, just create more scp files in data2json.sh
    for dataset in $datasets; do
	x=${dataset}_mel80
	local/scripts/data2json.sh --feat data/$x/feats.scp \
				   --token_scp data/$x/$token_scp_type \
    				   --non-ling-syms "${non_ling_syms}" \
    				   --output-utts-json data/$x/utts.json \
    				   --output-dir-of-scps data/$x/scps \
    				   data/$x ${dict}

    done
fi

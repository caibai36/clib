#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -uo pipefail

# general conf
stage=8  # start from 0 if you need to start from data preparation
run=run0
dataset_name=mit_sample # see https://marmosetbehavior.mit.edu
data_name=default # training:20150814_Cricket_Enid; eval/dev:20150903_Setta_Sailor;test/prediction:20161219_Athos_Porthos
# dataset_name=mit_data # see dataset shared by the paper of 'Close range vocal interaction in the common marmoset (Callithrix Jacchus)'
# data_name=default # training:pair3-10;dev:pair2;test:pair1

# data conf
data_div_yaml="conf/data/division_sample.yaml"
# data_div_yaml="conf/data/division.yaml"
feat_yaml="conf/feat/feat_sample.yaml"
# feat_yaml="conf/feat/feat.yaml"
token2id_yaml="conf/dict/token2id_marmoset.yaml"
train_chunk_size_ms=500
train_chunk_shift_ms=150
dev_chunk_size_ms=500
dev_chunk_shift_ms=400
test_chunk_size_ms=500
test_chunk_shift_ms=400
feat_dir="feat/sample"
# feat_dir="feat"

# model conf
model_name=att
exp_dir=exp/att

# dataset
mit_sample=/data/share/bin-wu/data/marmoset/vocalization/marmoset_mit_cnn/original/Wave_files
# mit_data=/data/share/bin-wu/data/marmoset/vocalization/marmoset_mit

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. local/scripts/parse_options.sh || exit 1

if [ ${stage} -le 1 ]; then
    date
    echo "Data preparation..."
    ./local/mit_sample_data_prep.sh --mit_sample $mit_sample --dataset_name $dataset_name --stage 0 # Get info.json
    # ./local/mit_data_data_prep.sh --mit_data $mit_data --dataset_name $dataset_name --stage 0 # Get info.json
    date
fi

chunk_json_dir="data/${dataset_name}/chunk_info/traincsize${train_chunk_size_ms}cshift${train_chunk_shift_ms}_devcsize${dev_chunk_size_ms}cshift${dev_chunk_shift_ms}_testcsize${test_chunk_size_ms}cshift${test_chunk_shift_ms}"
info_json="data/${dataset_name}/info.json"
if [ ${stage} -le 2 ]; then
    date
    echo "Chunk preparation..."
    echo "Breaking long audios and segments into short chunks..."
    python local/mit_seg_prep.py --data_div_yaml $data_div_yaml \
	   --info_json $info_json \
	   --feat_yaml $feat_yaml \
	   --token2id_yaml $token2id_yaml \
	   --train_chunk_size_ms $train_chunk_size_ms \
	   --train_chunk_shift_ms $train_chunk_shift_ms \
	   --dev_chunk_size_ms $dev_chunk_size_ms \
	   --dev_chunk_shift_ms $dev_chunk_shift_ms \
	   --test_chunk_size_ms $test_chunk_size_ms \
	   --test_chunk_shift_ms $test_chunk_shift_ms \
	   --feat_dir $feat_dir \
	   --chunk_json_dir $chunk_json_dir
    date
fi

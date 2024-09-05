#!/bin/bash

# Set bash to 'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands'
set -euo pipefail

# General configuration
stage=8 # Start from 0 if you need to start from data preparation
seed=2020
gpu=auto

# Data and model options
run=run0
dataset_name=riken2024
data_dir_name=riken2024_jay_half
model_name=cnn
exp_dir=exp/sys

# Options for data
data_div_yaml="conf/data/division_jay_half.yaml"  # YAML file containing data division by IDs for train, dev, and test sets
label2id_yaml="conf/dict/label2id.yaml"  # YAML file containing label-to-labelID mapping
# When the middle part of a sliding window overlaps with an interval of a label from an given Audacity segment file, assign the label to the window.
middle_part=0.05  # Proportion of the middle part of the sliding window in seconds for label assignment.
window_size=0.5  # DO NOT CHANGE; Size of the sliding window in seconds for label assignment. Be careful to modify window_size due to consistency to 2500ms chunks.
window_shift=0.05  # Shift of the sliding window in seconds for label assignment
noise_preserve_steps=5  # Number of steps to skip between preserved all-noise-no-label chunks # 1 means keeping all noise segments
eval_model="model.ckpt" # Set model to evaluate: "model_e20.ckpt" (epoch 20), "model.ckpt" (latest epoch), and "model_best_dev.ckpt" (epoch with best dev score)

test_id=230807_001_ch1 # One test id in $data_div_yaml file

# Options for training and evaluation
num_epochs=25
save_epoch_interval=20 # save model every x epochs
batch_size=2048 # 25
lr=0.0003
lr_decay_interval=1 # Decay the learning rate every x epochs by lr *= 0.97
avg_pred_win=5 # collect predicted probabilities by averaging that across x consecutive predictions
cutoffs="0 0.9 0.8"

# Data
# mit_sample=P:/riken/share/data/marmoset_mit_cnn/original/Wave_files # Windows Git shell
riken2024=/data/share/bin-wu/data/marmoset/vocalization/riken2024 # Linux shell

# Parse the options. (e.g., ./run.sh --stage 1)
# Note that the options should be defined as shell variables before parsing
. local/scripts/parse_options.sh || exit 1

if [ ${stage} -le -1 ]; then
    date
    echo "Data preparation..."
    ./local/riken2024_data_prep.sh --riken2024 "$riken2024" --stage 0
    date
fi

# info_json="data/${data_dir_name}/info.json"  # JSON file mapping IDs to audio and label file paths
info_json="data/riken2024/info.json"  # JSON file mapping IDs to audio and label file paths
data_name=$(basename $data_div_yaml .yaml)_winmid${middle_part}size${window_size}shift${window_shift}_noisekeep${noise_preserve_steps}
if [ ${stage} -le 2 ]; then
    date
    echo "Preparing spectra and labels for datasets..."

    python local/riken_cnn_data_prep_torchaudio.py \
	   --info_json $info_json \
	   --data_div_yaml $data_div_yaml \
	   --label2id_yaml $label2id_yaml \
	   --middle_part $middle_part \
	   --window_size $window_size \
	   --window_shift $window_shift \
	   --noise_preserve_steps $noise_preserve_steps \
	   --out_dir exp/data/$data_name
     date
fi

result_dir=$exp_dir/$dataset_name/$data_name/${model_name}-${run}/bs${batch_size}lr${lr}lrdecay${lr_decay_interval}avgpredwin${avg_pred_win}/train
mkdir -p $result_dir
if [ ${stage} -le 3 ]; then
    date
    echo "Train riken cnn..."
    python local/riken_cnn_train_v1.py \
        --train_input exp/data/$data_name/train_input.npy \
        --train_target exp/data/$data_name/train_target.npy \
        --dev_input exp/data/$data_name/dev_input.npy \
        --dev_target exp/data/$data_name/dev_target.npy \
	--mtest_input exp/data/$data_name/test_input.npy \
	--mtest_target exp/data/$data_name/test_target.npy \
	--batch_size $batch_size \
	--label2id_yaml $label2id_yaml \
	--dropout_rate 0.4 \
	--lr $lr \
	--lr_decay_interval $lr_decay_interval \
	--epsilon 0.001 \
	--num_epochs $num_epochs \
	--save_epoch_interval $save_epoch_interval \
	--avg_pred_win $avg_pred_win \
	--seed $seed \
	--gpu $gpu \
	--result $result_dir \
	--overwrite \
	--exit
    date
fi

eval_dir=$exp_dir/$dataset_name/$data_name/${model_name}-${run}/bs${batch_size}lr${lr}lrdecay${lr_decay_interval}avgpredwin${avg_pred_win}/eval
mkdir -p "$eval_dir"
if [ ${stage} -le 4 ]; then
    date
    echo "Test riken cnn..."
    python local/riken_cnn_test_v1.py \
        --test_input exp/data/$data_name/test_input_${test_id}.npy \
        --test_pred $eval_dir/test_pred_${test_id}.npy \
        --batch_size $batch_size \
	--label2id_yaml $label2id_yaml \
        --dropout_rate 0.4 \
        --lr $lr \
        --epsilon 0.001 \
        --avg_pred_win $avg_pred_win \
	--seed $seed \
	--gpu $gpu \
        --eval_model "${result_dir}/${eval_model}"
    date
fi

if [ ${stage} -le 5 ]; then
    date
    echo "Get segments from riken cnn..."
    python local/riken_cnn_predictor.py \
	   --label2id_yaml $label2id_yaml \
           --pred_files $eval_dir/test_pred_${test_id}.npy \
           --out_dir "$eval_dir"
    date
fi

if [ ${stage} -le 6 ]; then
    date
    echo "Evaluate segments from riken cnn..."
    rm -rf "$eval_dir/results.txt"

    python local/riken_cnn_eval_acc_confmat.py \
           --info_json data/riken2024/info.json \
	   --label2id_yaml $label2id_yaml \
           --hypo_files $eval_dir/test_pred_${test_id}.txt \
           --ref_uttids $test_id | tee -a "$eval_dir/results.txt"
    echo | tee -a "$eval_dir/results.txt"
    date
fi

#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -uo pipefail

# general configuration
stage=2  # start from 0 if you need to start from data preparation

# Data and model options
# model: mit_cnn_72
# 72: 2 stream with 9+9 output layers, where
# 72 is model index from website https://marmosetbehavior.mit.edu/
# 9 types include "noise" and "trill, twitter, phee, triphee, tsik, ek, chirp, and chatter"
run=run0
dataset_name=mit_sample # see https://marmosetbehavior.mit.edu
data_name=mit_sample0 # training:20150814_Cricket_Enid; eval/dev:20150903_Setta_Sailor;test/prediction:20161219_Athos_Porthos
model_name=mit_cnn_72
exp_dir=exp/sys

# options for training classificaton
# cutoffs: a list of cutoffs. e.g., cutoffs="0 0.3 0.4 0.5 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95"
# # predicted as a "noise" type when the confidence, the maximum prob. of all target types, is less than cutoff.
#
# Model setup on the paper of [oikarinen, 2019]
# # Iterations: 74000 # default is 2601; full dataset was trained for 74001
# # Batch size: 25
# # Learning rate: 3e-4
# # Exponentially decreasing with an epsilon of 1e-3 (the original author mistakes the epsilon of adam optimizer about numerical stability for the exponetial decay of learning rate)
# # Cutoff of 0.8 has proved to be the most accurate at replicating human labels but lower levels can help recognize more calls. If not provided a cutoff of 0.7 will be used
# # for the big dataset we used "if i%2000==0:" to evaluate accuracies less often
batch_size=25
lr=0.0003
# dropout=0.4 # for dropout layer
# eps=0.001 # for adam optimizer
cutoffs="0.7 0.8"
num_iter=2601 # 74001
eval_interval=200 # 2000 # evaluate every x iterations and lr = lr * 0.97
avg_pred_win=5 # collect predicted probabilities by averaging across x consecutive predictions with step size of 1

# Data
mit_sample=/data/share/bin-wu/data/marmoset/vocalization/marmoset_mit_cnn/original/Wave_files

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. local/scripts/parse_options.sh || exit 1

if [ ${stage} -le 1 ]; then
    date
    echo "Data preparation..."
    ./local/mit_sample_data_prep.sh --mit_sample $mit_sample --stage 0
    date
fi

result_dir=$exp_dir/$dataset_name/$data_name/${model_name}-${run}/bs${batch_size}lr${lr}evalinterval${eval_interval}avgpredwin${avg_pred_win}/train
if [ ${stage} -le 2 ]; then
    date
    echo "Train mit cnn 72..."
    python local/mit_cnn_train_72.py --train_input1 exp/data/$dataset_name/train_input1 \
	   --train_input2 exp/data/$dataset_name/train_input2 \
	   --train_target_single1 exp/data/$dataset_name/train_target_single1 \
	   --train_target_single2 exp/data/$dataset_name/train_target_single2 \
	   --dev_input1 exp/data/$dataset_name/dev_input1 \
	   --dev_input2 exp/data/$dataset_name/dev_input2 \
	   --dev_target_single1 exp/data/$dataset_name/dev_target_single1 \
	   --dev_target_single2 exp/data/$dataset_name/dev_target_single2 \
	   --batch_size $batch_size \
	   --dropout_rate 0.4 \
	   --lr $lr \
	   --epsilon 0.001 \
	   --num_iter $num_iter \
	   --eval_interval $eval_interval \
	   --avg_pred_win $avg_pred_win \
	   --result $result_dir
    date
fi

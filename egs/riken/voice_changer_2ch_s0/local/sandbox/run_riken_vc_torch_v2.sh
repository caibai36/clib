#!/bin/bash
# Implemented by bin-wu on 2024/11/04
# Note that phone set would be updated to "cr, cp, ek, ph, ts, tr, pp, tw" for all mit scripts.

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -uo pipefail

# general configuration
stage=8  # start from 0 if you need to start from data preparation

# data and model options
# model: mit_cnn_72
# 72: 2 stream with 9+9 output layers, where
# 72 is model index from website https://marmosetbehavior.mit.edu/
# 9 types include "noise" and "trill, twitter, phee, triphee, tsik, ek, chirp, and chatter"
run=run0
# dataset_name=mit_sample # see https://marmosetbehavior.mit.edu
# data_name=mit_sample0 # training:20150814_Cricket_Enid; eval/dev:20150903_Setta_Sailor;test/prediction:20161219_Athos_Porthos
dataset_name=riken_vc # see dataset of voice changer from Jay
data_name=riken_vc0 # training:pair3-10;eval/dev:pair2;test/prediction:pair1
model_name=mit_cnn_72
exp_dir=exp/run

# options for training classificaton
batch_size=25
lr=0.0003
cutoffs="0" # Deprecated; always keep the default value '0' (cutoff is tied to 'trph').
num_epochs=20
eval_epoch_interval=1 # evaluate every x iterations and lr = lr * 0.97
avg_pred_win=5 # collect predicted probabilities by averaging across x consecutive predictions with step size of 1

# Data
riken_vc_data=/data01/share/bin-wu/data/marmoset/vocalization/voice_changer_2ch
first="p1a1_240711-0985" # first uttid of a test pair
sec="p1a2_240711-0985" # second uttid of a test pair

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. local/scripts/parse_options.sh || exit 1

if [ ${stage} -le -1 ]; then
    date
    echo "Data preparation..."
    ./local/riken_vc_prep.sh --data $riken_vc_data --stage 0
    date
fi

result_dir=$exp_dir/$dataset_name/$data_name/${model_name}-${run}/bs${batch_size}lr${lr}evalinterval${eval_epoch_interval}avgpredwin${avg_pred_win}/train
mkdir -p $result_dir
if [ ${stage} -le 2 ]; then
    date
    echo "Train mit cnn 72..."
    python local/mit_cnn_train_72_torch_v2.py --train_input1 exp/data/$dataset_name/train_input1 \
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
	   --num_epochs $num_epochs \
	   --eval_epoch_interval $eval_epoch_interval \
	   --avg_pred_win $avg_pred_win \
	   --result $result_dir \
	   --overwrite \
	   --exit
    date
fi

eval_dir=$exp_dir/$dataset_name/$data_name/${model_name}-${run}/bs${batch_size}lr${lr}evalinterval${eval_epoch_interval}avgpredwin${avg_pred_win}/eval
mkdir -p $eval_dir
if [ ${stage} -le 3 ]; then
    date
    echo "Test mit cnn 72..."
    python local/mit_cnn_test_72_torch_v2.py --test_input1 exp/data/$dataset_name/test_input1_$first \
	   --test_input2 exp/data/$dataset_name/test_input2_$sec \
	   --test_pred1 $eval_dir/test_pred1_$first \
	   --test_pred2 $eval_dir/test_pred2_$sec \
	   --batch_size $batch_size \
	   --dropout_rate 0.4 \
	   --lr $lr \
	   --epsilon 0.001 \
	   --avg_pred_win $avg_pred_win \
	   --eval_model $result_dir/model.ckpt
    date
fi

if [ ${stage} -le 4 ]; then
    date
    echo "Cut off mit cnn 72..."
    python local/mit_cnn_cutoff_predictor_single.py --cutoffs $cutoffs \
	   --pred_files $eval_dir/test_pred1_$first.npy $eval_dir/test_pred2_$sec.npy \
	   --out_dir $eval_dir
    date
fi

if [ ${stage} -le 5 ]; then
    date
    echo "Evalute mit cnn 72..."
    rm $eval_dir/results.txt

    for cutoff in $cutoffs; do
	echo Cutoff: $cutoff | tee -a $eval_dir/results.txt
	python local/mit_cnn_eval_acc.py --info_json data/$dataset_name/info.json \
	       --hypo_files $eval_dir/test_pred1_${first}_cutoff$cutoff.txt $eval_dir/test_pred2_${sec}_cutoff$cutoff.txt \
	       --ref_uttids $first $sec | tee -a $eval_dir/results.txt
	echo | tee -a $eval_dir/results.txt
    done
    date
fi

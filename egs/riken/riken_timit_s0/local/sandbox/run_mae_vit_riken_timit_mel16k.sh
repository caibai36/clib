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
dataset_name=timit
data_dir_name=timit
model_name=mae_vit  # Changed from cnn to mae_vit
exp_dir=exp/sys_mae_vit  # Changed from sys to sys_mae_vit

# Options for data
data_div_yaml="conf/data/division_timit.yaml"  # YAML file containing data division by IDs for train, dev, and test sets
label2id_yaml="conf/dict/timit_label2id.yaml"  # YAML file containing label-to-labelID mapping
# When the middle part of a sliding window overlaps with an interval of a label from an given Audacity segment file, assign the label to the window.
middle_part=0.05  # Proportion of the middle part of the sliding window in seconds for label assignment.
window_size=0.5  # DO NOT CHANGE; Size of the sliding window in seconds for label assignment. Be careful to modify window_size due to consistency to 2500ms chunks.
window_shift=0.05  # Shift of the sliding window in seconds for label assignment
noise_preserve_steps=1 # Number of steps to skip between preserved all-noise-no-label chunks # 1 means keeping all noise segments
eval_model="model.ckpt" # Set model to evaluate: "model_e20.ckpt" (epoch 20), "model.ckpt" (latest epoch), and "model_best_dev.ckpt" (epoch with best dev score)

test_id=FDHC0_SI1559 # One test id in $data_div_yaml file

# Options for training and evaluation
num_epochs=25
save_epoch_interval=5 # save model every x epochs
batch_size=256  # Increased from 2048 to 256 for ViT
base_lr=0.0001  # Base learning rate
weight_decay=0.3  # Weight decay for AdamW optimizer
warmup_epochs=5  # Number of warmup epochs
total_epochs=25  # Total number of epochs for cosine annealing (including linear warmup)
avg_pred_win=5 # collect predicted probabilities by averaging that across x consecutive predictions

# ViT model parameters - Updated to mae_vit_base
image_size=256        # Image size (height, width) (not used in MAE; MAE uses 224x224)
patch_size=16         # Patch size (height, width)
dim=768               # Embedding dimension
depth=12              # Number of transformer layers
heads=12              # Number of attention heads
mlp_dim=3072         # Dimension of the MLP layer
pool="cls"           # Pooling type (cls or mean)
channels=1           # Number of input channels
dim_head=64          # Dimension of each attention head (not used in MAE)
dropout=0.1          # Dropout rate
emb_dropout=0.1      # Embedding dropout rate (not used in MAE)
mae_pretrained="base"       # "base" for facebook/vit-mae-base, path to pretrained model, or empty string for no pretraining
mae_train_layers=6  # Train all layers (-1) or specify number of layers to train

# Data
# mit_sample=P:/riken/share/data/marmoset_mit_cnn/original/Wave_files # Windows Git shell
# infant_miki_examples=/data/share/bin-wu/data/human/speech/examples/infant_miki_examples/HomeRecording01

# Parse the options. (e.g., ./run.sh --stage 1)
# Note that the options should be defined as shell variables before parsing
. local/scripts/parse_options.sh || exit 1

# if [ ${stage} -le 1 ]; then
#     date
#     echo "Data preparation..."
#     ./local/riken_infant_miki_examples_data_prep.sh --infant_miki_examples "$infant_miki_examples" --data_name "$data_dir_name" --stage 0
#     date
# fi

# info_json="data/${data_dir_name}/info.json"  # JSON file mapping IDs to audio and label file paths
info_json="data/timit/info.json"  # JSON file mapping IDs to audio and label file paths
data_name=$(basename $data_div_yaml .yaml)_winmid${middle_part}size${window_size}shift${window_shift}_noisekeep${noise_preserve_steps}_mel16k
if [ ${stage} -le 2 ]; then
    date
    echo "Preparing spectra and labels for datasets..."

    python local/riken_cnn_data_prep_torchaudio_mel16k.py \
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

vit_params="dim${dim}depth${depth}heads${heads}mlpdim${mlp_dim}patch${patch_size}dropout${dropout}maetrain${mae_train_layers}maepretrained${mae_pretrained:+$(basename ${mae_pretrained} .ckpt)}"
result_dir=$exp_dir/$dataset_name/$data_name/${model_name}-${run}/bs${batch_size}baselr${base_lr}wd${weight_decay}warmup${warmup_epochs}total${total_epochs}avgpredwin${avg_pred_win}_${vit_params}/train
mkdir -p $result_dir
if [ ${stage} -le 3 ]; then
    date
    echo "Train riken mae vit..."
    python local/riken_mae_vit_train_v1.py \
        --train_input exp/data/$data_name/train_input.npy \
        --train_target exp/data/$data_name/train_target.npy \
        --dev_input exp/data/$data_name/dev_input.npy \
        --dev_target exp/data/$data_name/dev_target.npy \
	--mtest_input exp/data/$data_name/test_input.npy \
	--mtest_target exp/data/$data_name/test_target.npy \
	--batch_size $batch_size \
	--label2id_yaml $label2id_yaml \
        --image_size $image_size \
        --patch_size $patch_size \
        --dim $dim \
        --depth $depth \
        --heads $heads \
        --mlp_dim $mlp_dim \
        --pool $pool \
        --channels $channels \
        --dim_head $dim_head \
        --dropout $dropout \
        --emb_dropout $emb_dropout \
        --base_lr $base_lr \
        --weight_decay $weight_decay \
        --warmup_epochs $warmup_epochs \
        --total_epochs $total_epochs \
	--num_epochs $num_epochs \
	--save_epoch_interval $save_epoch_interval \
	--avg_pred_win $avg_pred_win \
        --mae_pretrained "${mae_pretrained}" \
        --mae_train_layers $mae_train_layers \
        --seed $seed \
        --gpu $gpu \
	--result $result_dir \
	--overwrite \
	--exit
    date
fi

# eval_dir=$exp_dir/$dataset_name/$data_name/${model_name}-${run}/bs${batch_size}baselr${base_lr}wd${weight_decay}warmup${warmup_epochs}total${total_epochs}avgpredwin${avg_pred_win}_${vit_params}/eval
# mkdir -p "$eval_dir"
# if [ ${stage} -le 4 ]; then
#     date
#     echo "Test riken mae vit..."
#     python local/riken_mae_vit_test_v1.py \
#         --test_input exp/data/$data_name/test_input_${test_id}.npy \
#         --test_pred $eval_dir/test_pred_${test_id}.npy \
#         --batch_size $batch_size \
# 	--label2id_yaml $label2id_yaml \
#         --image_size $image_size \
#         --patch_size $patch_size \
#         --dim $dim \
#         --depth $depth \
#         --heads $heads \
#         --mlp_dim $mlp_dim \
#         --pool $pool \
#         --channels $channels \
#         --dim_head $dim_head \
#         --dropout $dropout \
#         --emb_dropout $emb_dropout \
#         --base_lr $base_lr \
#         --weight_decay $weight_decay \
#         --warmup_epochs $warmup_epochs \
#         --total_epochs $total_epochs \
#         --avg_pred_win $avg_pred_win \
#         --mae_pretrained "${mae_pretrained}" \
#         --mae_train_layers $mae_train_layers \
#         --seed $seed \
#         --gpu $gpu \
#         --eval_model "${result_dir}/${eval_model}"
#     date
# fi

# if [ ${stage} -le 5 ]; then
#     date
#     echo "Get segments from riken mae vit..."
#     python local/riken_cnn_predictor.py \
# 	   --label2id_yaml $label2id_yaml \
#            --pred_files $eval_dir/test_pred_${test_id}.npy \
#            --out_dir "$eval_dir"
#     date
# fi

# if [ ${stage} -le 6 ]; then
#     date
#     echo "Evaluate segments from riken mae vit..."
#     rm -rf "$eval_dir/results.txt"

#     python local/riken_cnn_eval_acc_confmat.py \
#            --info_json data/timit/info.json \
# 	   --label2id_yaml $label2id_yaml \
#            --hypo_files $eval_dir/test_pred_${test_id}.txt \
#            --ref_uttids $test_id | tee -a "$eval_dir/results.txt"
#     echo | tee -a "$eval_dir/results.txt"
#     date
# fi

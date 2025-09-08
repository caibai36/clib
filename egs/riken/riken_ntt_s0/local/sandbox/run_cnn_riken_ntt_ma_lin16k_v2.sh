#!/bin/bash

# Set bash to 'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands'
set -euo pipefail

# General configuration
stage=8 # Start from 0 if you need to start from data preparation

# Data and model options
run=run0
dataset_name=ntt_infant
data_dir_name=ntt_infant
model_name=cnn
exp_dir=exp/sys_ma

# Options for data (If missing noise key, then manually add a key of 'noise' to conf/dict/ntt_label2id.yaml)
data_div_yaml="conf/data/division_ntt_riken_model_ma.yaml"  # YAML file containing data division by IDs for train, dev, and test sets
label2id_yaml="conf/dict/ntt_label2id.yaml"  # YAML file containing label-to-labelID mapping
# When the middle part of a sliding window overlaps with an interval of a label from an given Audacity segment file, assign the label to the window.
middle_part=0.05  # Proportion of the middle part of the sliding window in seconds for label assignment.
window_size=0.5  # DO NOT CHANGE; Size of the sliding window in seconds for label assignment. Be careful to modify window_size due to consistency to 2500ms chunks.
window_shift=0.05 # 0.1 # 0.05  # Shift of the sliding window in seconds for label assignment
noise_preserve_steps=5 # 10 # Number of steps to skip between preserved all-noise-no-label chunks # 1 means keeping all noise segments

# Options for evaluation
pred_resolution=0.01  # Evaluation resolution: 0.05 or 0.01 (0.05 for training and evaluation; but prediction can be 0.05 or 0.01)
ref_key_in_info_json="seg"  # Key in info.json for reference files

# Options for training and evaluation
num_epochs=25
save_epoch_interval=20 # save model every x epochs
batch_size=2048 # 256 # 2048 # 25
lr=0.0003
lr_decay_interval=1 # Decay the learning rate every x epochs by lr *= 0.97
avg_pred_win=5 # collect predicted probabilities by averaging that across x consecutive predictions
cutoffs="0 0.9 0.8"
gpu="auto"

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
info_json="data/ntt_infant_phone/info.json"  # JSON file mapping IDs to audio and label file paths
data_name=$(basename $data_div_yaml .yaml)_winmid${middle_part}size${window_size}shift${window_shift}_noisekeep${noise_preserve_steps}_lin16k
if [ ${stage} -le 2 ]; then
    date
    echo "Preparing spectra and labels for datasets using linear spectrograms (FFT=512)..."

    python local/riken_cnn_data_prep_torchaudio_lin16k_less_cpu_memory.py \
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
    echo "Train riken cnn with linear spectrograms (FFT=512)..."
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
	--result $result_dir \
	--gpu $gpu \
	--overwrite \
	--exit
    date
fi

eval_dir=$exp_dir/$dataset_name/$data_name/${model_name}-${run}/bs${batch_size}lr${lr}lrdecay${lr_decay_interval}avgpredwin${avg_pred_win}/eval
mkdir -p "$eval_dir"
if [ ${stage} -le 4 ]; then
    date
    echo "Test riken cnn with linear spectrograms (FFT=512) on all test files..."
    python local/sandbox/riken_cnn_audio2seg_highres_padding_lin16k_yaml.py \
        --eval_model "${result_dir}/model.ckpt" \
        --eval_dir "$eval_dir" \
        --eval_data_division_yaml "$data_div_yaml" \
        --info_json "$info_json" \
        --ref_key_in_info_json "$ref_key_in_info_json" \
        --batch_size 25 \
	--label2id_yaml $label2id_yaml \
        --avg_pred_win $avg_pred_win \
        --pred_resolution $pred_resolution \
        --fixed_factor 0 \
        --gpu auto \
        --padding
    date
fi

if [ ${stage} -le 5 ]; then
    date
    echo "Evaluate segments from riken cnn..."
    rm -rf "$eval_dir/results.txt"

    # Get all hypothesis files (create arrays, not strings)
    hypo_files=($(find "$eval_dir/hyp" -name "*.txt" 2>/dev/null | sort))
    ref_files=($(find "$eval_dir/ref" -name "*.txt" 2>/dev/null | sort))

    if [ ${#hypo_files[@]} -eq 0 ]; then
        echo "No hypothesis files found in $eval_dir/hyp"
        exit 1
    fi

    if [ ${#ref_files[@]} -eq 0 ] && [ -n "$ref_key_in_info_json" ]; then
        echo "No reference files found in $eval_dir/ref"
        exit 1
    fi

    # Run evaluation
    echo; python -c "import yaml; data=yaml.safe_load(open('$data_div_yaml')); [print(f'{k}: {len(v) if v is not None else 0} files') for k,v in data.items()]" | grep "test:" | sed 's/test:/Test set in data divsion:/g'
    python local/riken_cnn_eval_acc_confmat_v2.py \
           --label2id_yaml $label2id_yaml \
           --hypo_files "${hypo_files[@]}" \
           --ref_files "${ref_files[@]}" | tee -a "$eval_dir/results.txt"

    echo | tee -a "$eval_dir/results.txt"
    date
fi

#!/bin/bash

# Set reference file
ref_file="/home/bin-wu/share/data/riken/riken2024/jay_family/seg/230807_001_ch1.txt"
tolerance=0.1

# Array of all hypothesis files
hypo_files=(
    "/data/home/bin-wu/workspace/projects/clib/egs/riken/riken_das_s0/data/jay_half_test/230807_001_ch1_annotations.csv.sorted.txt"
    "/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/exp/sys/riken2024/division_jay_half_winmid0.05size0.5shift0.05_noisekeep5/cnn-run0/bs2048lr0.0003lrdecay1avgpredwin5/eval/test_pred_230807_001_ch1.txt"
    "exp/sys_vit/riken2024/division_jay_half_winmid0.05size0.5shift0.05_noisekeep5/vit-run0/bs256baselr0.002wd0.3warmup10total200avgpredwin5_dim384depth6heads6mlpdim1536patch16dropout0.1/eval/test_pred_230807_001_ch1.txt"
    "/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/exp/sys_mae_vit/riken2024/division_jay_half_winmid0.05size0.5shift0.05_noisekeep5/mae_vit-run0/bs256baselr0.0001wd0.3warmup10total100avgpredwin5_dim768depth12heads12mlpdim3072patch16dropout0.1maetrain6maepretrainedmodel_epoch_390_scratch_6days.pt/eval/test_pred_230807_001_ch1.txt"
    "/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/exp/sys_mae_vit/riken2024/division_jay_half_winmid0.05size0.5shift0.05_noisekeep5/mae_vit-run0/bs256baselr0.0001wd0.3warmup10total100avgpredwin5_dim768depth12heads12mlpdim3072patch16dropout0.1maetrain6maepretrainedmodel_epoch_390_base_6days.pt/eval/test_pred_230807_001_ch1.txt"
    "/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/exp/sys_mae_vit/riken2024/division_jay_half_winmid0.05size0.5shift0.05_noisekeep5/mae_vit-run0/bs256baselr0.0001wd0.3warmup10total100avgpredwin5_dim768depth12heads12mlpdim3072patch16dropout0.1maetrain6maepretrainedmodel_epoch_400_scratch_12days.pt/eval/test_pred_230807_001_ch1.txt"
    "/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/exp/sys_mae_vit/riken2024/division_jay_half_winmid0.05size0.5shift0.05_noisekeep5/mae_vit-run0/bs256baselr0.0001wd0.3warmup10total100avgpredwin5_dim768depth12heads12mlpdim3072patch16dropout0.1maetrain6maepretrainedmodel_epoch_400_base_12days.pt/eval/test_pred_230807_001_ch1.txt"
    "/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/exp/sys_mae_vit/riken2024/division_jay_half_winmid0.05size0.5shift0.05_noisekeep5/mae_vit-run0/bs256baselr0.0001wd0.3warmup10total100avgpredwin5_dim768depth12heads12mlpdim3072patch16dropout0.1maetrain6maepretrainedmodel_epoch_400_scratch_24days.pt/eval/test_pred_230807_001_ch1.txt"
    "/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/exp/sys_mae_vit/riken2024/division_jay_half_winmid0.05size0.5shift0.05_noisekeep5/mae_vit-run0/bs256baselr0.0001wd0.3warmup10total100avgpredwin5_dim768depth12heads12mlpdim3072patch16dropout0.1maetrain6maepretrainedmodel_epoch_400_base_24days.pt/eval/test_pred_230807_001_ch1.txt"
    "/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/exp/sys_mae_vit/riken2024/division_jay_half_winmid0.05size0.5shift0.05_noisekeep5/mae_vit-run0/bs256baselr0.0001wd0.3warmup10total100avgpredwin5_dim768depth12heads12mlpdim3072patch16dropout0.1maetrain6maepretrainedmodel_epoch_400_scratch_48days.pt/eval/test_pred_230807_001_ch1.txt"
    "/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/exp/sys_mae_vit/riken2024/division_jay_half_winmid0.05size0.5shift0.05_noisekeep5/mae_vit-run0/bs256baselr0.0001wd0.3warmup10total100avgpredwin5_dim768depth12heads12mlpdim3072patch16dropout0.1maetrain6maepretrainedmodel_epoch_400_base_48days.pt/eval/test_pred_230807_001_ch1.txt"
    "/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/exp/sys_mae_vit/riken2024/division_jay_half_winmid0.05size0.5shift0.05_noisekeep5/mae_vit-run0/bs256baselr0.0001wd0.3warmup10total100avgpredwin5_dim768depth12heads12mlpdim3072patch16dropout0.1maetrain6maepretrainedmodel_epoch_400_scratch_155days.pt/eval/test_pred_230807_001_ch1.txt"
    "/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/exp/sys_mae_vit/riken2024/division_jay_half_winmid0.05size0.5shift0.05_noisekeep5/mae_vit-run0/bs256baselr0.0001wd0.3warmup10total100avgpredwin5_dim768depth12heads12mlpdim3072patch16dropout0.1maetrain6maepretrainedmodel_epoch_400_base_155days.pt/eval/test_pred_230807_001_ch1.txt"
)

# Function to extract model name from path
get_model_name() {
    local filepath=$1
    if [[ $filepath =~ scratch_([0-9]+days) ]]; then
        echo "MAE-ViT Scratch ${BASH_REMATCH[1]}"
    elif [[ $filepath =~ base_([0-9]+days) ]]; then
        echo "MAE-ViT Base ${BASH_REMATCH[1]}"
    elif [[ $filepath =~ cnn-run ]]; then
        echo "CNN"
    elif [[ $filepath =~ vit-run ]]; then
        echo "ViT"
    elif [[ $filepath =~ annotations ]]; then
        echo "Baseline"
    else
        echo "Unknown model"
    fi
}

# Evaluate each hypothesis file
for hypo_file in "${hypo_files[@]}"; do
    echo "==============================================="
    echo "Evaluating: $(get_model_name "$hypo_file")"
    echo "File: $hypo_file"
    python ../riken_cnn_s0/local/sandbox/riken_cnn_eval_acc_boundaries.py \
        --ref_files "$ref_file" \
        --hypo_files "$hypo_file" \
        --tolerance "$tolerance"
    echo
done

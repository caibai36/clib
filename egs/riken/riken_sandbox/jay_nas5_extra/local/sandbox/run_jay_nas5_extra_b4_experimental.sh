#!/bin/bash

# run_mae_analysis.sh
# MAE Feature Extraction and Visualization Pipeline for Marmoset Developmental Analysis
#
# Set bash to 'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands'
set -euo pipefail

# General configuration
stage=8  # Start from 0 if you need to start from data preparation
gpu=3

# Dataset name
dataset=b4_1372F_1168M_3269F  # Dataset identifier for logging and output naming

# Data paths - CONFIGURE THESE
wav_dir=/data04/home/bin-wu/workspace/projects/virtualbox/share/data/jay_das_annotation/b4_1372F_1168M_3269F/wav
seg_dir=/data04/home/bin-wu/workspace/projects/virtualbox/share/data/jay_das_annotation/b4_1372F_1168M_3269F/seg
info_csv=/data04/home/bin-wu/workspace/projects/virtualbox/share/data/jay_das_annotation/b4_1372F_1168M_3269F/b4_2025-12-19outlineAGEid.csv

# Output directories - CONFIGURE THESE
data_dir=data/b4_1372F_1168M_3269F
exp_dir=/data04/home/bin-wu/workspace/projects/virtualbox/share/data/jay_das_annotation/exp/sandbox/b4_1372F_1168M_3269F
fig_dir=/data04/home/bin-wu/workspace/projects/virtualbox/share/data/jay_das_annotation/exp/sandbox/b4_1372F_1168M_3269F/figures

# Model path
mae_model=conf/model/mae_pretrained_model_epoch_400_base_48days.pt

# CSV extraction options
age_days_col=Age
audio_id_col=id
max_duration=3.0

# MAE feature extraction options
batch_size=32

# Dimensionality reduction options
pca_components=50
tsne_perplexity=30.0
tsne_iterations=1000
random_seed=42

# Visualization options
label_option=2        # 1: keep all, 2: merge u-X, 3: remove u-X
attribute=age_days
n_cols=14
topk=5
n_stages=3
sort_stage=avg_age    # Options: avg_age, min_age, max_age (b4 uses min_age)
feature_mode=count_ratio+mae_tsne  # Options: count_ratio, mae_tsne, count_ratio+mae_tsne
max_clusters=10

# Parse the options. (e.g., ./run_mae_analysis.sh --stage 1 --dataset b4_1372F_1168M_3269F)
# Note that the options should be defined as shell variables before parsing
. local/scripts/parse_options.sh || exit 1

# Create necessary directories
mkdir -p ${data_dir}
mkdir -p ${exp_dir}
mkdir -p ${fig_dir}
mkdir -p logs

if [ ${stage} -le 0 ]; then
    date
    echo "Stage 0: Extracting CSV information for ${dataset}..."

    python -u local/nas5_extra_extract_csv_info.py \
        --wav_dir ${wav_dir} \
        --seg_dir ${seg_dir} \
        --info_csv ${info_csv} \
        --age_days_col ${age_days_col} \
        --audio_id_col ${audio_id_col} \
        --dataid ${dataset} \
        --output_csv ${data_dir}/${dataset}.csv \
        --max_duration ${max_duration} \
        |& tee logs/stage0_extract_csv_info_${dataset}.log
    date
fi

if [ ${stage} -le 1 ]; then
    date
    echo "Stage 1: Extracting MAE features for ${dataset}..."

    python local/extract_mae_feature_from_csv.py \
        --csv_path ${data_dir}/${dataset}.csv \
        --wav_dir ${wav_dir} \
        --output_dir ${exp_dir} \
        --mae_model ${mae_model} \
        --batch_size ${batch_size} \
        --gpu ${gpu} \
        |& tee logs/stage1_extract_mae_features_${dataset}.log
    date
fi

if [ ${stage} -le 2 ]; then
    date
    echo "Stage 2: Performing dimensionality reduction (PCA + t-SNE)..."

    python local/mae_dim_reduction.py \
        --info_csv ${exp_dir}/info.csv \
        --mae_features ${exp_dir}/mae_raw.npy \
        --output_dir ${exp_dir} \
	--pca_components ${pca_components} \
	--tsne_perplexity ${tsne_perplexity} \
	--tsne_iterations ${tsne_iterations} \
	--random_seed ${random_seed} \
	|& tee logs/stage2_mae_dim_reduction_${dataset}.log
    date
fi

if [ ${stage} -le 3 ]; then
    date
    echo "Stage 3: Generating comprehensive visualizations..."

    python local/sandbox/vis.py \
        --info_mae_csv ${exp_dir}/info_mae.csv \
        --output_dir ${fig_dir} \
        --data_name ${dataset} \
        --label_option ${label_option} \
        --attribute ${attribute} \
        --n_cols ${n_cols} \
        --topk ${topk} \
        --n_stages ${n_stages} \
        --sort_stage ${sort_stage} \
        --feature_mode ${feature_mode} \
        --max_clusters ${max_clusters} \
        |& tee logs/stage3_visualize_comprehensive_${dataset}.log
    date
fi

if [ ${stage} -le 4 ]; then
    date
    echo "Stage 4: Generating summary report..."

    report_file=${exp_dir}/analysis_report.txt

    echo "MAE FEATURE ANALYSIS REPORT" > ${report_file}
    echo "================================================================================" >> ${report_file}
    echo "Dataset: ${dataset}" >> ${report_file}
    echo "Date: $(date)" >> ${report_file}
    echo "" >> ${report_file}
    echo "Configuration:" >> ${report_file}
    echo "  MAE Model: ${mae_model}" >> ${report_file}
    echo "  Batch Size: ${batch_size}" >> ${report_file}
    echo "  PCA Components: ${pca_components}" >> ${report_file}
    echo "  t-SNE Perplexity: ${tsne_perplexity}" >> ${report_file}
    echo "  t-SNE Iterations: ${tsne_iterations}" >> ${report_file}
    echo "  Label Option: ${label_option}" >> ${report_file}
    echo "  Feature Mode: ${feature_mode}" >> ${report_file}
    echo "  Number of Stages: ${n_stages}" >> ${report_file}
    echo "  Sort Stage by: ${sort_stage}" >> ${report_file}
    echo "" >> ${report_file}
    echo "Input Paths:" >> ${report_file}
    echo "  WAV Directory: ${wav_dir}" >> ${report_file}
    echo "  SEG Directory: ${seg_dir}" >> ${report_file}
    echo "  Info CSV: ${info_csv}" >> ${report_file}
    echo "" >> ${report_file}
    echo "Output Files:" >> ${report_file}
    echo "  CSV Data: ${data_dir}/${dataset}.csv" >> ${report_file}
    echo "  MAE Features: ${exp_dir}/mae_raw.npy" >> ${report_file}
    echo "  Spectrograms: ${exp_dir}/spec_raw.npy" >> ${report_file}
    echo "  Info with t-SNE: ${exp_dir}/info_mae.csv" >> ${report_file}
    echo "  t-SNE Coordinates: ${exp_dir}/mae_tsne.npy" >> ${report_file}
    echo "  Figures: ${fig_dir}/" >> ${report_file}
    echo "" >> ${report_file}
    echo "File Sizes:" >> ${report_file}
    du -sh ${data_dir}/${dataset}.csv 2>/dev/null >> ${report_file} || echo "  CSV: Not found" >> ${report_file}
    du -sh ${exp_dir}/mae_raw.npy 2>/dev/null >> ${report_file} || echo "  MAE: Not found" >> ${report_file}
    du -sh ${exp_dir}/spec_raw.npy 2>/dev/null >> ${report_file} || echo "  Spec: Not found" >> ${report_file}
    du -sh ${exp_dir}/info_mae.csv 2>/dev/null >> ${report_file} || echo "  Info: Not found" >> ${report_file}
    echo "" >> ${report_file}
    echo "Generated Figures:" >> ${report_file}
    ls -lh ${fig_dir}/*.png 2>/dev/null >> ${report_file} || echo "  No figures found" >> ${report_file}

    cat ${report_file}
    date
fi

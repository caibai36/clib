#!/bin/bash

# run_jay_nas5_b2_f1.sh
# MAE Feature Extraction and Visualization Pipeline for Marmoset Developmental Analysis
# with Age Filtering (15 weeks = 105 days)
#
# Implemented by bin-wu at 10:06 on 20160106
#
# Set bash to 'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands'
set -euo pipefail

# General configuration
stage=8  # Starting from stage 1 (MAE feature extraction)
gpu=2

# Dataset name
dataset=b2_f1_mae_vit  # Dataset identifier for logging and output naming

# Data paths - CONFIGURED FOR NAS5 B2 F1
wav_dir=/data01/share/bin-wu/data/marmoset/vocalization/riken_long/nas5/b2_1305F_759M_3162F
# seg_dir=  # Empty from stage 1
info_csv=/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/data/nas5_b2_f1/b2_f1_mae_vit.csv

# Output directories - CONFIGURED FOR NAS5 B2 F1
exp_dir=/data02/share/bin-wu/exp/sandbox/nas5/b2_f1
dim_reduction_dir=/data02/share/bin-wu/exp/sandbox/nas5/b2_f1/dim_reduction_15w
fig_dir=/data02/share/bin-wu/exp/sandbox/nas5/b2_f1/dim_reduction_15w/figures

# Model path
mae_model=conf/model/mae_pretrained_model_epoch_400_base_48days.pt

# MAE feature extraction options
batch_size=32

# Age filtering options
age_col=age_days
begin_age=0
end_age=105  # 15 weeks = 105 days

# Output filenames for filtered results
mae_tsne_name=mae_tsne.npy
info_mae_name=info_mae.csv

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
sort_stage=avg_age    # Options: avg_age, min_age, max_age
feature_mode=count_ratio+mae_tsne  # Options: count_ratio, mae_tsne, count_ratio+mae_tsne
max_clusters=10

# Parse the options. (e.g., ./run_nas5_b2_f1_15w.sh --stage 1 --gpu 2)
# Note that the options should be defined as shell variables before parsing
. local/scripts/parse_options.sh || exit 1

# Create necessary directories
mkdir -p ${exp_dir}
mkdir -p ${dim_reduction_dir}
mkdir -p ${fig_dir}
mkdir -p logs

# Stage -1 is skipped since CSV already exists
if [ ${stage} -le -1 ]; then
    date
    echo "Stage -1: Extracting CSV information for ${dataset}..."
    echo "INFO: Skipping this stage - CSV already exists at ${info_csv}"
    date
fi

if [ ${stage} -le 1 ]; then
    date
    echo "Stage 1: Extracting MAE features for ${dataset}..."
    echo "INFO: WAV directory: ${wav_dir}"
    echo "INFO: The script will recursively search for WAV files in subdirectories"
    echo "INFO: Expected WAV structure: ${wav_dir}/20240309_/mic_L/240309_001_ch1.wav"

    # python local/extract_mae_feature_from_csv.py \
    python local/extract_mae_feature_from_csv_compatible_v0.py \
        --csv_path ${info_csv} \
        --wav_dir ${wav_dir} \
        --output_dir ${exp_dir} \
        --mae_model ${mae_model} \
        --batch_size ${batch_size} \
        --gpu ${gpu} \
        |& tee logs/stage1_extract_mae_features_${dataset}.log

    echo "INFO: MAE features saved to ${exp_dir}/mae_raw.npy"
    echo "INFO: Spectrograms saved to ${exp_dir}/spec_raw.npy"
    echo "INFO: Info CSV saved to ${exp_dir}/info.csv"
    date
fi

if [ ${stage} -le 2 ]; then
    date
    echo "Stage 2: Performing dimensionality reduction (PCA + t-SNE) with age filtering..."
    echo "INFO: Age filtering: ${begin_age} to ${end_age} days"
    echo "INFO: Output directory: ${dim_reduction_dir}"

    python local/mae_dim_reduction_begin_end_ages.py \
        --info_csv ${exp_dir}/info.csv \
        --mae_features ${exp_dir}/mae_raw.npy \
        --output_dir ${dim_reduction_dir} \
        --age_col ${age_col} \
        --begin_age ${begin_age} \
        --end_age ${end_age} \
        --mae_tsne ${mae_tsne_name} \
        --info_mae ${info_mae_name} \
        --pca_components ${pca_components} \
        --tsne_perplexity ${tsne_perplexity} \
        --tsne_iterations ${tsne_iterations} \
        --random_seed ${random_seed} \
        |& tee logs/stage2_mae_dim_reduction_${dataset}.log

    echo "INFO: Filtered MAE features saved to ${dim_reduction_dir}/mae_filtered.npy"
    echo "INFO: t-SNE results saved to ${dim_reduction_dir}/${mae_tsne_name}"
    echo "INFO: Info with t-SNE saved to ${dim_reduction_dir}/${info_mae_name}"
    date
fi

if [ ${stage} -le 3 ]; then
    date
    echo "Stage 3: Generating comprehensive visualizations..."
    echo "INFO: Using filtered data from ${dim_reduction_dir}/${info_mae_name}"

    python local/sandbox/vis.py \
        --info_mae_csv ${dim_reduction_dir}/${info_mae_name} \
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

    report_file=${dim_reduction_dir}/analysis_report.txt

    echo "MAE FEATURE ANALYSIS REPORT" > ${report_file}
    echo "================================================================================" >> ${report_file}
    echo "Dataset: ${dataset}" >> ${report_file}
    echo "Age Filter: ${begin_age} to ${end_age} days" >> ${report_file}
    echo "Date: $(date)" >> ${report_file}
    echo "" >> ${report_file}
    echo "Configuration:" >> ${report_file}
    echo "  MAE Model: ${mae_model}" >> ${report_file}
    echo "  Batch Size: ${batch_size}" >> ${report_file}
    echo "  GPU: ${gpu}" >> ${report_file}
    echo "  Age Column: ${age_col}" >> ${report_file}
    echo "  Age Range: ${begin_age} to ${end_age} days" >> ${report_file}
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
    echo "  Info CSV: ${info_csv}" >> ${report_file}
    echo "" >> ${report_file}
    echo "Output Files:" >> ${report_file}
    echo "  MAE Features (original): ${exp_dir}/mae_raw.npy" >> ${report_file}
    echo "  MAE Features (filtered): ${dim_reduction_dir}/mae_filtered.npy" >> ${report_file}
    echo "  Spectrograms: ${exp_dir}/spec_raw.npy" >> ${report_file}
    echo "  Info CSV (original): ${exp_dir}/info.csv" >> ${report_file}
    echo "  Info CSV (filtered): ${dim_reduction_dir}/info_filtered.csv" >> ${report_file}
    echo "  Info with t-SNE: ${dim_reduction_dir}/${info_mae_name}" >> ${report_file}
    echo "  t-SNE Coordinates: ${dim_reduction_dir}/${mae_tsne_name}" >> ${report_file}
    echo "  Age Mask: ${dim_reduction_dir}/age_mask.txt" >> ${report_file}
    echo "  Figures: ${fig_dir}/" >> ${report_file}
    echo "" >> ${report_file}
    echo "File Sizes:" >> ${report_file}
    du -sh ${info_csv} 2>/dev/null >> ${report_file} || echo "  Input CSV: Not found" >> ${report_file}
    du -sh ${exp_dir}/mae_raw.npy 2>/dev/null >> ${report_file} || echo "  MAE (original): Not found" >> ${report_file}
    du -sh ${dim_reduction_dir}/mae_filtered.npy 2>/dev/null >> ${report_file} || echo "  MAE (filtered): Not found" >> ${report_file}
    du -sh ${exp_dir}/spec_raw.npy 2>/dev/null >> ${report_file} || echo "  Spec: Not found" >> ${report_file}
    du -sh ${exp_dir}/info.csv 2>/dev/null >> ${report_file} || echo "  Info (original): Not found" >> ${report_file}
    du -sh ${dim_reduction_dir}/${info_mae_name} 2>/dev/null >> ${report_file} || echo "  Info MAE: Not found" >> ${report_file}
    echo "" >> ${report_file}
    echo "Sample Statistics:" >> ${report_file}
    if [ -f ${exp_dir}/info.csv ]; then
        echo "  Total samples (original): $(tail -n +2 ${exp_dir}/info.csv | wc -l)" >> ${report_file}
    fi
    if [ -f ${dim_reduction_dir}/${info_mae_name} ]; then
        echo "  Total samples (filtered): $(tail -n +2 ${dim_reduction_dir}/${info_mae_name} | wc -l)" >> ${report_file}
    fi
    echo "" >> ${report_file}
    echo "Generated Figures:" >> ${report_file}
    ls -lh ${fig_dir}/*.png 2>/dev/null >> ${report_file} || echo "  No figures found" >> ${report_file}

    cat ${report_file}
    date
fi

echo "================================================================================"
echo "Pipeline completed successfully!"
echo "Original features: ${exp_dir}"
echo "Filtered results: ${dim_reduction_dir}"
echo "Figures: ${fig_dir}"
echo "================================================================================"

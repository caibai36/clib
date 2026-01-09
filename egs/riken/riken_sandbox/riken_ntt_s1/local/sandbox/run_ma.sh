#!/bin/bash

# run_ntt_ma.sh
# MAE Feature Extraction and Clustering Analysis Pipeline for NTT Infant Data
# with Age Filtering
#
# Implemented by bin-wu
#
# Set bash to 'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands'
set -euo pipefail

# General configuration
stage=8  # Starting from stage 0
gpu=1

# Dataset name
dataset=ntt_ma  # Dataset identifier for logging and output naming

# Data paths - CONFIGURED FOR NTT MA
audio_root=/data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/processed/ntt_infant_data
info_csv=/data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/processed/metas_speaker_force_alignment_phone/meta_ma_speaker.csv

# Output directories - CONFIGURED FOR NTT MA
exp_dir=/data02/share/bin-wu/exp/sandbox/ntt/ma/filter_begin0_end-2
fig_dir=/data02/share/bin-wu/exp/sandbox/ntt/ma/filter_begin0_end-2/figures

# Model path
mae_model=conf/models/mae_pretrained_model_best_sa.pt

# Spectrogram extraction options
batch_size=32
whole_begin=0      # Start from month 0
whole_end=-2       # The last two months of each family are reserved for development and test

# Age column
age_col=age_months  # Column name for age

# Output filenames for results
mae_tsne_name=mae_tsne.npy
info_mae_name=info_mae.csv

# Dimensionality reduction options
pca_components=50
tsne_perplexity=30.0
tsne_iterations=1000
random_seed=42

# Clustering options
n_stages=3
feature_mode=count_ratio+mae_tsne  # Options: count_ratio, mae_tsne, count_ratio+mae_tsne
max_clusters=10
sort_stage=avg_age    # Options: avg_age, min_age, max_age
label_col=label       # Column name for labels (auto-detect if empty)

# Parse the options. (e.g., ./run_ntt_ma.sh --stage 1 --gpu 2)
# Note that the options should be defined as shell variables before parsing
. local/scripts/parse_options.sh || exit 1

# Create necessary directories
mkdir -p ${exp_dir}
mkdir -p ${fig_dir}
mkdir -p logs

if [ ${stage} -le 0 ]; then
    date
    echo "Stage 0: Extracting linear spectrograms for ${dataset}..."
    echo "INFO: Input CSV: ${info_csv}"
    echo "INFO: Audio root: ${audio_root}"
    echo "INFO: Age filter: months ${whole_begin} to ${whole_end}"
    echo "INFO: Output directory: ${exp_dir}"

    python local/ntt_extract_spec_batch_lin16k.py \
        --whole \
        --whole_begin ${whole_begin} \
        --whole_end ${whole_end} \
        --csv_path ${info_csv} \
        --audio_root ${audio_root} \
        --output_dir ${exp_dir} \
        --batch_size ${batch_size} \
        |& tee logs/stage0_extract_spec_${dataset}.log

    echo "INFO: Spectrograms saved to ${exp_dir}/spec_raw.npy"
    echo "INFO: Info CSV saved to ${exp_dir}/info.csv"
    date
fi

if [ ${stage} -le 1 ]; then
    date
    echo "Stage 1: Extracting MAE features for ${dataset}..."
    echo "INFO: Input spectrograms: ${exp_dir}/spec_raw.npy"
    echo "INFO: MAE model: ${mae_model}"
    echo "INFO: Using GPU: ${gpu}"

    python local/riken_mae_pretrained_feature_extraction.py \
        --gpu ${gpu} \
        --input_file ${exp_dir}/spec_raw.npy \
        --pretrained_path ${mae_model} \
        --output_file ${exp_dir}/mae_raw.npy \
        |& tee logs/stage1_extract_mae_features_${dataset}.log

    echo "INFO: MAE features saved to ${exp_dir}/mae_raw.npy"
    date
fi

if [ ${stage} -le 2 ]; then
    date
    echo "Stage 2: Performing dimensionality reduction (PCA + t-SNE)..."
    echo "INFO: Input features: ${exp_dir}/mae_raw.npy"
    echo "INFO: Input info CSV: ${exp_dir}/info.csv"
    echo "INFO: Output directory: ${exp_dir}"

    python local/ntt_mae_dim_reduction_begin_end_ages_filter_none_signal.py \
        --info_csv ${exp_dir}/info.csv \
        --mae_features ${exp_dir}/mae_raw.npy \
        --output_dir ${exp_dir} \
        --mae_tsne ${mae_tsne_name} \
        --info_mae ${info_mae_name} \
        --pca_components ${pca_components} \
        --tsne_perplexity ${tsne_perplexity} \
        --tsne_iterations ${tsne_iterations} \
        --random_seed ${random_seed} \
        |& tee logs/stage2_mae_dim_reduction_${dataset}.log

    echo "INFO: Filtered MAE features saved to ${exp_dir}/mae_filtered.npy"
    echo "INFO: t-SNE results saved to ${exp_dir}/${mae_tsne_name}"
    echo "INFO: Info with t-SNE saved to ${exp_dir}/${info_mae_name}"
    date
fi

if [ ${stage} -le 3 ]; then
    date
    echo "Stage 3: Performing hierarchical clustering analysis..."
    echo "INFO: Using data from ${exp_dir}/${info_mae_name}"
    echo "INFO: Number of stages: ${n_stages}"
    echo "INFO: Feature mode: ${feature_mode}"

    # Build command with optional label column
    cmd="python local/sandbox/vis.py \
        --info_mae_csv ${exp_dir}/${info_mae_name} \
        --output_dir ${fig_dir} \
        --data_name ${dataset} \
        --n_stages ${n_stages} \
        --feature_mode ${feature_mode} \
        --max_clusters ${max_clusters} \
        --age_unit ${age_col} \
        --sort_stage ${sort_stage}"

    # Add label column if specified
    if [ -n "${label_col}" ]; then
        cmd="${cmd} --label_col ${label_col}"
    fi

    eval ${cmd} |& tee logs/stage3_clustering_analysis_${dataset}.log

    echo "INFO: Clustering results saved to ${fig_dir}/"
    date
fi

if [ ${stage} -le 4 ]; then
    date
    echo "Stage 4: Generating summary report..."

    report_file=${exp_dir}/analysis_report.txt

    echo "NTT INFANT MAE FEATURE ANALYSIS REPORT" > ${report_file}
    echo "================================================================================" >> ${report_file}
    echo "Dataset: ${dataset}" >> ${report_file}
    echo "Age Filter: months ${whole_begin} to ${whole_end}" >> ${report_file}
    echo "Date: $(date)" >> ${report_file}
    echo "" >> ${report_file}
    echo "Configuration:" >> ${report_file}
    echo "  MAE Model: ${mae_model}" >> ${report_file}
    echo "  Batch Size: ${batch_size}" >> ${report_file}
    echo "  GPU: ${gpu}" >> ${report_file}
    echo "  Age Column: ${age_col}" >> ${report_file}
    echo "  Whole Begin: ${whole_begin}" >> ${report_file}
    echo "  Whole End: ${whole_end}" >> ${report_file}
    echo "  PCA Components: ${pca_components}" >> ${report_file}
    echo "  t-SNE Perplexity: ${tsne_perplexity}" >> ${report_file}
    echo "  t-SNE Iterations: ${tsne_iterations}" >> ${report_file}
    echo "  Feature Mode: ${feature_mode}" >> ${report_file}
    echo "  Number of Stages: ${n_stages}" >> ${report_file}
    echo "  Sort Stage by: ${sort_stage}" >> ${report_file}
    echo "" >> ${report_file}
    echo "Input Paths:" >> ${report_file}
    echo "  Audio Root: ${audio_root}" >> ${report_file}
    echo "  Info CSV: ${info_csv}" >> ${report_file}
    echo "" >> ${report_file}
    echo "Output Files:" >> ${report_file}
    echo "  Spectrograms: ${exp_dir}/spec_raw.npy" >> ${report_file}
    echo "  MAE Features (raw): ${exp_dir}/mae_raw.npy" >> ${report_file}
    echo "  MAE Features (filtered): ${exp_dir}/mae_filtered.npy" >> ${report_file}
    echo "  Info CSV (original): ${exp_dir}/info.csv" >> ${report_file}
    echo "  Info CSV (filtered): ${exp_dir}/info_filtered.csv" >> ${report_file}
    echo "  Info with t-SNE: ${exp_dir}/${info_mae_name}" >> ${report_file}
    echo "  t-SNE Coordinates: ${exp_dir}/${mae_tsne_name}" >> ${report_file}
    echo "  NaN Mask: ${exp_dir}/nan_mask.txt" >> ${report_file}
    echo "  Clustering Results: ${fig_dir}/" >> ${report_file}
    echo "" >> ${report_file}
    echo "File Sizes:" >> ${report_file}
    du -sh ${info_csv} 2>/dev/null >> ${report_file} || echo "  Input CSV: Not found" >> ${report_file}
    du -sh ${exp_dir}/spec_raw.npy 2>/dev/null >> ${report_file} || echo "  Spectrograms: Not found" >> ${report_file}
    du -sh ${exp_dir}/mae_raw.npy 2>/dev/null >> ${report_file} || echo "  MAE (raw): Not found" >> ${report_file}
    du -sh ${exp_dir}/mae_filtered.npy 2>/dev/null >> ${report_file} || echo "  MAE (filtered): Not found" >> ${report_file}
    du -sh ${exp_dir}/info.csv 2>/dev/null >> ${report_file} || echo "  Info (original): Not found" >> ${report_file}
    du -sh ${exp_dir}/${info_mae_name} 2>/dev/null >> ${report_file} || echo "  Info MAE: Not found" >> ${report_file}
    echo "" >> ${report_file}
    echo "Sample Statistics:" >> ${report_file}
    if [ -f ${exp_dir}/info.csv ]; then
        echo "  Total samples (original): $(tail -n +2 ${exp_dir}/info.csv | wc -l)" >> ${report_file}
    fi
    if [ -f ${exp_dir}/${info_mae_name} ]; then
        echo "  Total samples (after filtering): $(tail -n +2 ${exp_dir}/${info_mae_name} | wc -l)" >> ${report_file}
    fi
    echo "" >> ${report_file}
    echo "Generated Figures:" >> ${report_file}
    ls -lh ${fig_dir}/*.png 2>/dev/null >> ${report_file} || echo "  No figures found" >> ${report_file}
    echo "" >> ${report_file}
    echo "Clustering Results:" >> ${report_file}
    ls -lh ${fig_dir}/*.csv 2>/dev/null >> ${report_file} || echo "  No CSV files found" >> ${report_file}

    cat ${report_file}
    date
fi

echo "================================================================================"
echo "Pipeline completed successfully!"
echo "Results directory: ${exp_dir}"
echo "Clustering directory: ${fig_dir}"
echo "================================================================================"

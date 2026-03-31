#!/bin/bash

# Set bash to 'debug' mode
set -euo pipefail

# General configuration
stage=8

# Data paths
id2nii=conf/mri/id2nii/new_england_id2nii_age_le_50.yaml
id2age=data_mri/new_england/t1w/id2age_rounded.yaml
subjects_dir=/data02/share/bin-wu/data/human/brain/harvard_mri/processed/infant_freesurfer/age_le_50_months/new_england/

# Configuration files
log_dir=logs/infant_freesurfer

# Parse options
. local/scripts/parse_options.sh || exit 1

# Create directories
mkdir -p ${log_dir}

# ===============================================
# Stage 1: Data Preparation
# ===============================================
if [ ${stage} -le -1 ]; then
    echo "============================================="
    echo "Stage 1: Data Preparation"
    echo "  Prepare New England and Calgary MRI datasets"
    echo "  Generate id2nii and id2age mapping files"
    echo "============================================="
    date

    # Run preparation scripts for both datasets
    ./local/mri/local/prepare_all_new_england_calgary.sh

    date
    echo "Stage 1 completed"
    echo ""
fi

# ===============================================
# Stage 2: Infant FreeSurfer Processing
# ===============================================
if [ ${stage} -le 2 ]; then
    echo "============================================="
    echo "Stage 2: Infant FreeSurfer Processing"
    echo "  Run Infant FreeSurfer on all subjects"
    echo "  Process T1w images for brain segmentation"
    echo "============================================="
    date

    # Read id2nii YAML file and process each subject
    while IFS=': ' read -r subjid nii_path; do
        # Skip empty lines and comments
        [[ -z "${subjid}" || "${subjid}" =~ ^#.*$ ]] && continue

        # Trim whitespace
        subjid=$(echo ${subjid} | xargs)
        nii_path=$(echo ${nii_path} | xargs)

        # Get age from id2age YAML
        age=$(grep "^${subjid}:" ${id2age} | cut -d':' -f2 | xargs)

        # Skip if age not found
        if [ -z "${age}" ]; then
            echo "Warning: Could not find age for ${subjid}, skipping..."
            continue
        fi

        # Check if T1w input exists
        if [ ! -f "${nii_path}" ]; then
            echo "Warning: T1w file not found for ${subjid}: ${nii_path}"
            continue
        fi

        echo "Processing ${subjid}: age=${age} months"
        echo "  Input: ${nii_path}"

        # Run Infant FreeSurfer
        ./local/mri/local/run_infant_freesurfer.sh \
            --t1w_input ${nii_path} \
            --subjid ${subjid} \
            --age ${age} \
            --subjects_dir ${subjects_dir} \
            |& tee ${log_dir}/run_infant_freesurfer_${subjid}_${age}.log

        echo "Completed ${subjid}"
        echo "----------------------------------------"

    done < ${id2nii}

    date
    echo "Stage 2 completed"
    echo ""
fi

# # ===============================================
# # Stage 3: Vocal Area Extraction
# # ===============================================
# if [ ${stage} -le 3 ]; then
#     echo "============================================="
#     echo "Stage 3: Vocal Area Extraction"
#     echo "  Extract affective vocalization areas"
#     echo "  Based on FreeSurfer segmentation results"
#     echo "============================================="
#     date

#     # Read id2nii YAML file and extract vocal areas for each subject
#     while IFS=': ' read -r subjid nii_path; do
#         # Skip empty lines and comments
#         [[ -z "${subjid}" || "${subjid}" =~ ^#.*$ ]] && continue

#         # Trim whitespace
#         subjid=$(echo ${subjid} | xargs)
#         nii_path=$(echo ${nii_path} | xargs)

#         # Get age from id2age YAML
#         age=$(grep "^${subjid}:" ${id2age} | cut -d':' -f2 | xargs)

#         # Skip if age not found
#         if [ -z "${age}" ]; then
#             echo "Warning: Could not find age for ${subjid}, skipping..."
#             continue
#         fi

#         # Check if T1w input exists
#         if [ ! -f "${nii_path}" ]; then
#             echo "Warning: T1w file not found for ${subjid}: ${nii_path}"
#             continue
#         fi

#         # Check if FreeSurfer output exists
#         if [ ! -d "${subjects_dir}/${subjid}" ]; then
#             echo "Warning: FreeSurfer output not found for ${subjid}, skipping..."
#             continue
#         fi

#         echo "Extracting vocal areas for ${subjid}: age=${age} months"
#         echo "  Input: ${nii_path}"
#         echo "  FreeSurfer dir: ${subjects_dir}/${subjid}"

#         # Run vocal area extraction
# #        local/mri/local/sandbox/extract_affective_vocalization_areas.ver2_detail.sh \
# 	local/mri/local/sandbox/extract_affective_vocalization_areas.ver3_detail.sh \
#             --t1w_input ${nii_path} \
#             --subjid ${subjid} \
#             --subjects_dir ${subjects_dir} \
#             |& tee ${log_dir}/extract_vocal_areas_${subjid}_${age}.log

#         echo "Completed ${subjid}"
#         echo "----------------------------------------"

#     done < ${id2nii}

#     date
#     echo "Stage 3 completed"
#     echo ""
# fi

echo "All stages completed!"

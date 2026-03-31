#!/bin/bash

# Infant Brain Morphometry Pipeline: iBEATv2 + infant FreeSurfer + FreeSurfer
# This pipeline assumes that iBEATv2 tissue segmentations have already been generated
# Reference: https://github.com/TeddyTuresky/Longitudinal-Trajectories-Early-Brain-Development-Language/blob/main/pre-peer-review/1.Structure/reFS.sh

set -euo pipefail

# General configuration
stage=8

# Data paths
id2nii=conf/mri/id2nii/new_england_id2nii_age_le_50.yaml
id2age=data_mri/new_england/t1w/id2age_rounded.yaml
id2ibeat_tissue=conf/mri/id2ibeat_tissue/new_england_id2ibeat_tissue_age_le_50.yaml

# Directory paths
fs_subjects_dir=/data02/share/bin-wu/data/human/brain/harvard_mri/processed/ibeat_infant_freesurfer_freesurfer/age_le_50_months/new_england
script_dir=local/mri/local/structure
matlab_bin_dir=/usr/local/MATLAB/R2025b/bin

# Configuration files
log_dir=logs/ibeat_infant_freesurfer_freesurfer

# Model options
num_jobs=10 # Number of parallel jobs

# Parse options
. local/scripts/parse_options.sh || exit 1

# Create directories
mkdir -p ${log_dir} ${fs_subjects_dir}

export SUBJECTS_DIR=${fs_subjects_dir}

# Setup directories (after parser)
ifs_parent_dir="${ifs_parent_dir:-$(dirname ${fs_subjects_dir})/ifs_outputs}"

# Find matlab functions directory
matlab_func_dir="$(cd ${script_dir} && pwd)"

echo "Using FreeSurfer: $FREESURFER_HOME"
echo "SUBJECTS_DIR: ${fs_subjects_dir}"
echo "Infant FreeSurfer parent dir: ${ifs_parent_dir}"

# ===============================================
# Stage -1: Data Preparation
# ===============================================
if [ ${stage} -le -1 ]; then
    echo "============================================="
    echo "Stage -1: Data Preparation"
    echo "  Prepare New England and Calgary MRI datasets"
    echo "  Generate id2nii and id2age mapping files"
    echo "  Generate id2ibeat_tissue mapping file"
    echo "============================================="
    date

    # Run preparation scripts for both datasets
    ./local/mri/local/prepare_all_new_england_calgary.sh

    date
    echo "Stage -1 completed"
    echo ""
fi

# ===============================================
# Stage 0-1: Import and FreeSurfer recon-all
# ===============================================
if [ ${stage} -le 1 ]; then
    echo "============================================="
    echo "Stage 0-1: Import and FreeSurfer recon-all"
    echo "  Import T1w NIfTI files into FreeSurfer format"
    echo "  Run recon-all -all -nofill"
    echo "  Stop at step 15 (before filled.mgz)"
    echo "============================================="
    date

    # Read id2nii YAML file and process each subject
    while IFS=': ' read -r subjid nii_path; do
        # Skip empty lines and comments
        [[ -z "${subjid}" || "${subjid}" =~ ^#.*$ ]] && continue

        # Trim whitespace
        subjid=$(echo ${subjid} | xargs)
        nii_path=$(echo ${nii_path} | xargs)

        # Check if T1w input exists
        if [ ! -f "${nii_path}" ]; then
            echo "Warning: T1w file not found for ${subjid}: ${nii_path}"
            continue
        fi

        # Setup directories for this subject
        fs_dir="${fs_subjects_dir}/${subjid}"

        # Check if subject already exists
        if [ -d "${fs_dir}" ]; then
            echo "Subject ${subjid} already exists, running recon-all"
        else
            echo "Importing and processing ${subjid}"
            echo "  Input: ${nii_path}"
        fi

        # Import (if needed) and run recon-all -all -nofill
        # -nofill stops recon-all at step 15 (https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all)
        #  -nofill skips step 15 (no filled.mgz), but recon-all still tries step 16+
        #  if you requested them (e.g., with -all), which then crash.
        #  error: mri_pretess: could not open ../mri/filled.mgz
        set +e  # Temporarily allow commands to fail
        recon-all -i ${nii_path} -all -nofill -subjid ${subjid} -openmp ${num_jobs} \
            |& tee ${log_dir}/stage01_recon_all_${subjid}.log
        set -e  # Re-enable exit on error
        # Alternative without NU intensity correction:
        # recon-all -i ${nii_path} -all -subjid ${subjid} -nonuintensitycor -openmp ${num_jobs}

        echo "Completed ${subjid}"
        echo "----------------------------------------"

    done < ${id2nii}

    date
    echo "Stage 1 completed"
    echo ""
fi

# # ===============================================
# # Stage 2: Remove files based on initial FS run
# # ===============================================
# if [ ${stage} -le 2 ]; then
#     echo "============================================="
#     echo "Stage 2: Remove files based on initial FS run"
#     echo "  Clean up transforms and intensity correction files"
#     echo "============================================="
#     date

#     # Read id2nii YAML file and clean each subject
#     while IFS=': ' read -r subjid nii_path; do
#         # Skip empty lines and comments
#         [[ -z "${subjid}" || "${subjid}" =~ ^#.*$ ]] && continue

#         # Trim whitespace
#         subjid=$(echo ${subjid} | xargs)

#         # Setup directories for this subject
#         fs_dir="${fs_subjects_dir}/${subjid}"

#         # Check if subject directory exists
#         if [ ! -d "${fs_dir}" ]; then
#             echo "Warning: Subject directory not found for ${subjid}, skipping..."
#             continue
#         fi

#         echo "Cleaning ${subjid}"

#         rm -rf ${fs_dir}/mri/transforms/* \
#                ${fs_dir}/mri/orig_nu.mgz \
#                ${fs_dir}/mri/mri_nu_correct.mni.log

#         echo "Completed ${subjid}"
#         echo "----------------------------------------"

#     done < ${id2nii}

#     date
#     echo "Stage 2 completed"
#     echo ""
# fi

# # ===============================================
# # Stage 3: Setting up for infant FreeSurfer
# # ===============================================
# if [ ${stage} -le 3 ]; then
#     echo "============================================="
#     echo "Stage 3: Setting up for infant FreeSurfer"
#     echo "  Convert orig.mgz to mprage.nii.gz"
#     echo "============================================="
#     date

#     # Read id2nii YAML file and setup each subject
#     while IFS=': ' read -r subjid nii_path; do
#         # Skip empty lines and comments
#         [[ -z "${subjid}" || "${subjid}" =~ ^#.*$ ]] && continue

#         # Trim whitespace
#         subjid=$(echo ${subjid} | xargs)

#         # Setup directories for this subject
#         fs_dir="${fs_subjects_dir}/${subjid}"
#         ifs_dir="${ifs_parent_dir}/${subjid}"

#         # Check if subject directory exists
#         if [ ! -d "${fs_dir}" ]; then
#             echo "Warning: Subject directory not found for ${subjid}, skipping..."
#             continue
#         fi

#         echo "Setting up ${subjid} for infant FreeSurfer"

#         mkdir -p ${ifs_dir}
#         mri_convert -i ${fs_dir}/mri/orig.mgz -o ${ifs_dir}/mprage.nii.gz

#         echo "Completed ${subjid}"
#         echo "----------------------------------------"

#     done < ${id2nii}

#     date
#     echo "Stage 3 completed"
#     echo ""
# fi

# # ===============================================
# # Stage 4: Running infant FreeSurfer
# # ===============================================
# if [ ${stage} -le 4 ]; then
#     echo "============================================="
#     echo "Stage 4: Running infant FreeSurfer"
#     echo "  Process infant brain segmentation"
#     echo "============================================="
#     date

#     # Read id2nii YAML file and process each subject
#     while IFS=': ' read -r subjid nii_path; do
#         # Skip empty lines and comments
#         [[ -z "${subjid}" || "${subjid}" =~ ^#.*$ ]] && continue

#         # Trim whitespace
#         subjid=$(echo ${subjid} | xargs)

#         # Get age from id2age YAML
#         age=$(grep "^${subjid}:" ${id2age} | cut -d':' -f2 | xargs)

#         # Skip if age not found
#         if [ -z "${age}" ]; then
#             echo "Warning: Could not find age for ${subjid}, skipping..."
#             continue
#         fi

#         # Setup directories for this subject
#         ifs_dir="${ifs_parent_dir}/${subjid}"

#         # Check if infant FreeSurfer directory exists
#         if [ ! -d "${ifs_dir}" ]; then
#             echo "Warning: Infant FreeSurfer directory not found for ${subjid}, skipping..."
#             continue
#         fi

#         echo "Running infant FreeSurfer for ${subjid}: age=${age} months"

#         ${script_dir}/iFS_wrap.sh ${ifs_parent_dir} ${subjid} ${age} \
#             |& tee ${log_dir}/stage4_infant_freesurfer_${subjid}_${age}.log

#         echo "Using FreeSurfer: $FREESURFER_HOME"
#         echo "Completed ${subjid}"
#         echo "----------------------------------------"

#     done < ${id2nii}

#     date
#     echo "Stage 4 completed"
#     echo ""
# fi

# # ===============================================
# # Stage 5: Merge iBEAT + infant FS segmentations
# # ===============================================
# if [ ${stage} -le 5 ]; then
#     echo "============================================="
#     echo "Stage 5: Merge iBEAT + infant FS segmentations"
#     echo "  ibeat2aseg: merge iBEAT tissue with infant FS aseg"
#     echo "  aseg2wm: generate white-matter masks for FS"
#     echo "============================================="
#     date

#     # ibeat2aseg: merges iBEATv2 tissue segmentation (${m_ibeat}) with infant FreeSurfer aseg data (${m_ifs} and ${m_fs_mri}).
#     #     converts iBEAT segmentations (gray/white/CSF) into FreeSurfer's label format (aseg.mgz).
#     # aseg2wm(...): Generates white-matter masks (and possibly aseg.presurf.mgz and wm.mgz)
#     #     required by FreeSurfer's surface reconstruction stages.

#     # Read id2nii YAML file and process each subject
#     while IFS=': ' read -r subjid nii_path; do
#         # Skip empty lines and comments
#         [[ -z "${subjid}" || "${subjid}" =~ ^#.*$ ]] && continue

#         # Trim whitespace
#         subjid=$(echo ${subjid} | xargs)

#         # Get iBEAT tissue segmentation path from id2ibeat_tissue YAML
#         ibeat_tissue=$(grep "^${subjid}:" ${id2ibeat_tissue} | cut -d':' -f2 | xargs)

#         # Skip if iBEAT tissue not found
#         if [ -z "${ibeat_tissue}" ]; then
#             echo "Warning: Could not find iBEAT tissue for ${subjid}, skipping..."
#             continue
#         fi

#         # Check if iBEAT tissue file exists
#         if [ ! -f "${ibeat_tissue}" ]; then
#             echo "Warning: iBEAT tissue file not found for ${subjid}: ${ibeat_tissue}"
#             continue
#         fi

#         # Setup directories for this subject
#         fs_dir="${fs_subjects_dir}/${subjid}"
#         ifs_dir="${ifs_parent_dir}/${subjid}"

#         # Check if directories exist
#         if [ ! -d "${fs_dir}" ] || [ ! -d "${ifs_dir}" ]; then
#             echo "Warning: FreeSurfer or infant FreeSurfer directory not found for ${subjid}, skipping..."
#             continue
#         fi

#         echo "Merging segmentations for ${subjid}"
#         echo "  iBEAT tissue: ${ibeat_tissue}"

#         # Matlab paths (quoted for Matlab)
#         m_fs_mri="'${fs_dir}/mri'" # output location for ibeat2aseg.m
#         m_ibeat="'${ibeat_tissue}'"
#         m_ifs="'${ifs_dir}'"
#         m_aseg="'${fs_dir}/mri/aseg.presurf.nii'" # output from ibeat2aseg.m and input to aseg2wm.m
#         m_func="'${matlab_func_dir}'"

#         ${matlab_bin_dir}/matlab -nodesktop -nosplash -r \
#             "addpath(${m_func}); \
#              ibeat2aseg(${m_ibeat}, ${m_ifs}, ${m_fs_mri}); \
#              aseg2wm(${m_aseg}); \
#              exit;" \
#             |& tee ${log_dir}/stage5_merge_segmentations_${subjid}.log

#         echo "Completed ${subjid}"
#         echo "----------------------------------------"

#     done < ${id2nii}

#     date
#     echo "Stage 5 completed"
#     echo ""
# fi

# # ===============================================
# # Stage 6: Finish FS recon-all -autorecon2-wm
# # ===============================================
# if [ ${stage} -le 6 ]; then
#     echo "============================================="
#     echo "Stage 6: Finish FS recon-all -autorecon2-wm"
#     echo "  Copy Talairach transforms from infant FS"
#     echo "  Complete autorecon2-wm pipeline"
#     echo "============================================="
#     date

#     # Read id2nii YAML file and process each subject
#     while IFS=': ' read -r subjid nii_path; do
#         # Skip empty lines and comments
#         [[ -z "${subjid}" || "${subjid}" =~ ^#.*$ ]] && continue

#         # Trim whitespace
#         subjid=$(echo ${subjid} | xargs)

#         # Setup directories for this subject
#         fs_dir="${fs_subjects_dir}/${subjid}"
#         ifs_dir="${ifs_parent_dir}/${subjid}"

#         # Check if directories exist
#         if [ ! -d "${fs_dir}" ] || [ ! -d "${ifs_dir}" ]; then
#             echo "Warning: FreeSurfer or infant FreeSurfer directory not found for ${subjid}, skipping..."
#             continue
#         fi

#         echo "Finishing autorecon2-wm for ${subjid}"

#         cp ${ifs_dir}/mri/transforms/talairach*.xfm ${fs_dir}/mri/transforms/
#         ${script_dir}/fs_autorecon2_end.sh ${fs_dir} ${ifs_dir} ${subjid} \
#             |& tee ${log_dir}/stage6_autorecon2_${subjid}.log

#         echo "Completed ${subjid}"
#         echo "----------------------------------------"

#     done < ${id2nii}

#     date
#     echo "Stage 6 completed"
#     echo ""
# fi

# # ===============================================
# # Stage 7: Run FS recon-all -autorecon3
# # ===============================================
# if [ ${stage} -le 7 ]; then
#     echo "============================================="
#     echo "Stage 7: Run FS recon-all -autorecon3"
#     echo "  Complete surface reconstruction pipeline"
#     echo "  Apply expert.opts adjustments"
#     echo "============================================="
#     date

#     # Read id2nii YAML file and process each subject
#     while IFS=': ' read -r subjid nii_path; do
#         # Skip empty lines and comments
#         [[ -z "${subjid}" || "${subjid}" =~ ^#.*$ ]] && continue

#         # Trim whitespace
#         subjid=$(echo ${subjid} | xargs)

#         # Setup directories for this subject
#         fs_dir="${fs_subjects_dir}/${subjid}"

#         # Check if subject directory exists
#         if [ ! -d "${fs_dir}" ]; then
#             echo "Warning: FreeSurfer directory not found for ${subjid}, skipping..."
#             continue
#         fi

#         echo "Running autorecon3 for ${subjid}"

#         # make sure that the expert.opts is in the script directory (matlab_func_dir)
#         ${script_dir}/fs_autorecon3_wrap.sh ${fs_dir} ${matlab_func_dir} ${subjid} ${num_jobs} \
#             |& tee ${log_dir}/stage7_autorecon3_${subjid}.log

#         echo "Completed ${subjid}"
#         echo "----------------------------------------"

#     done < ${id2nii}

#     date
#     echo "Stage 7 completed"
#     echo ""
# fi

# echo "All stages completed!"

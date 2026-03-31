#!/bin/bash

# FreeSurfer Pipeline for subjects over 50 months (using standard recon-all)
# This pipeline runs recon-all for complete structural processing of older children/adults
# Reference: https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all

# set -euo pipefail

# General configuration
stage=1
num_jobs=25

# Data options
t1w_input="/data02/share/bin-wu/data/human/brain/harvard_mri/raw/new_england/ds006169-1.0.3/sub-01/ses-03/anat/sub-01_ses-03_T1w.nii.gz"
subjid="id2"
age=74
# Processed results at $subjects_dir/$subjid; use absolute path
subjects_dir="$PWD/exp/mri/test_pure_fs/fs_outputs"

. ./local/scripts/parse_options.sh || exit 1

# Export FreeSurfer configuration
export FS_LICENSE=/work02/home/bin-wu/.local/freesurfer/license.txt
export FREESURFER_HOME=/work02/home/bin-wu/.local/freesurfer/v8.1.0/8.1.0 && \
    export SUBJECTS_DIR=/work02/home/bin-wu/.local/freesurfer/v8.1.0/subjects && \
    source $FREESURFER_HOME/SetUpFreeSurfer.sh && \
    echo "Using FreeSurfer 8.1.0 (Subjects: $SUBJECTS_DIR)"

export SUBJECTS_DIR=$subjects_dir
mkdir -p $SUBJECTS_DIR

echo "Using SUBJECTS_DIR: $SUBJECTS_DIR"
echo "Subject: ${subjid}, Age: ${age} months"
echo "Using ${num_jobs} parallel jobs"

# Stage 1: Run recon-all (complete pipeline)
if [ ${stage} -le 1 ]; then
    echo "Stage 1: Running recon-all (complete pipeline with -all)"
    date

    recon-all \
        -s ${subjid} \
        -i ${t1w_input} \
        -all \
        -parallel \
        -openmp ${num_jobs}

    date
fi

echo "Pipeline completed: ${subjid}"
echo "Results saved in: ${SUBJECTS_DIR}/${subjid}"

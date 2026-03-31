#!/bin/bash

# Infant FreeSurfer Pipeline for subjects under 25 months (0-5 years with v8.1.0)
# This pipeline runs infant_recon_all for complete structural processing
# Reference: https://surfer.nmr.mgh.harvard.edu/fswiki/infantFS

# set -euo pipefail

# General configuration
stage=1

# Data options
t1w_input="/data02/share/bin-wu/data/human/brain/harvard_mri/raw/new_england/ds006169-1.0.3/sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz"
subjid="id1"
age=7
# Processed results at $fs_subject_dir/$subjid; use absolute path required by infant_all
subjects_dir="$PWD/exp/mri/test_pure_ifs/ifs_outputs" 

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

# Stage 1: Run infant_recon_all (complete pipeline)
if [ ${stage} -le 1 ]; then
    echo "Stage 1: Running infant_recon_all (complete pipeline)"
    date
    
    if [ ${age} -eq 0 ]; then
        # v8.1.0 does not interpret "--age 0" properly - MUST use --newborn
        infant_recon_all --s ${subjid} --inputfile ${t1w_input} --newborn
        echo "Used --newborn flag (age 0 requires this flag in v8.1.0)"
    else
        infant_recon_all --s ${subjid} --inputfile ${t1w_input} --age ${age}
    fi
    
    date
fi

echo "Pipeline completed: ${subjid}"
echo "Results saved in: ${SUBJECTS_DIR}/${subjid}"

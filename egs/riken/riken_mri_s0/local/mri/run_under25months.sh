#!/bin/bash

# Run the script with freesurfer v7
# Reference: https://github.com/TeddyTuresky/Longitudinal-Trajectories-Early-Brain-Development-Language/blob/main/pre-peer-review/1.Structure/reFS.sh

# Infant Brain Morphometry Pipeline: iBEATv2 + infant FreeSurfer + FreeSurfer
# This pipeline assumes that iBEATv2 tissue segmentations have already been generated
# and that the T1w image has already been imported into FreeSurfer SUBJECTS_DIR.

set -euo pipefail

# # How to import nii file into SUBJECTS_DIR.
# (mlp) [bin-wu@s186 riken_ntt_s0]$(master) SUBJECTS_DIR="exp/mri/test/fs_processed" # output directory for processed files
# (mlp) [bin-wu@s186 riken_ntt_s0]$(master) mkdir -p $SUBJECTS_DIR
# (mlp) [bin-wu@s186 riken_ntt_s0]$(master) recon-all -i /data02/share/bin-wu/data/human/brain/harvard_mri/raw/new_england/ds006169-1.0.3/sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz -subjid id1 # From dataset to processed directory
# (mlp) [bin-wu@s186 riken_ntt_s0]$(master) ls exp/mri/test/fs_processed/id1/
# label  mri  scripts  stats  surf  tmp  touch  trash

# General configuration
stage=8
# Use a absolute path for SUBJECTS_DIR to make infant_recon_all work
fs_subjects_dir="$PWD/exp/mri/test/fs_processed" # output directory for processed files ($SUBJECTS_DIR in freesurfer convention)
script_dir="local/mri/local/structure"
matlab_bin_dir="/usr/local/MATLAB/R2025b/bin"

# Data option
ibeat_tissue="/work02/home/bin-wu/workspace/projects/tests/test_ibeatv2/new_england/out/sub-01_ses-01/T1-skullstripped-rmcere-tissue.nii.gz"
subjid="id1"
age=6

# Model option
num_jobs=30 # Number of parallel jobs

. ./local/scripts/parse_options.sh || exit 1

export SUBJECTS_DIR=$fs_subjects_dir

# # Validate inputs
# if [ $# -ne 3 ]; then
#     echo "Usage: $0 [options] <subject-id> <ibeat-tissue-seg> <age-months>"
#     echo ""
#     echo "This script assumes that iBEATv2 tissue segmentations have already been generated."
#     echo ""
#     echo "Options:"
#     echo "  --stage <N>                Start from stage N (default: 1)"
#     echo "  --fs-subjects-dir <dir>    FreeSurfer SUBJECTS_DIR"
#     echo "  --ifs-parent-dir <dir>     Parent directory for infant FS"
#     echo "  --script-dir <dir>         Directory containing helper scripts"
#     echo "  --use-nuintensitycor <bool> Use NU intensity correction (default: false)"
#     exit 1
# fi

# subjid=$1
# ibeat_tissue=$2
# age=$3

# # Check inputs
# if [[ ! -f "${ibeat_tissue}" ]]; then
#     echo "ERROR: iBEATv2 tissue segmentation not specified or not found: ${ibeat_tissue}"
#     exit 1
# fi

# if [ $# -ne 3 ]; then
#     echo "ERROR: Three arguments not specified. Participant ID and/or age may be missing."
#     exit 1
# fi

# Setup directories
fs_dir="${fs_subjects_dir}/${subjid}"
ifs_parent_dir="${ifs_parent_dir:-$(dirname ${fs_subjects_dir})/ifs_outputs}"
ifs_dir="${ifs_parent_dir}/${subjid}"

# Find matlab functions directory
matlab_func_dir="$(cd ${script_dir} && pwd)"

# Matlab paths (quoted for Matlab)
m_fs_mri="'${fs_dir}/mri'" # output location for ibeat2aseg.m
m_ibeat="'${ibeat_tissue}'"
m_ifs="'${ifs_dir}'"
m_aseg="'${fs_dir}/mri/aseg.presurf.nii'" # output from ibeat2aseg.m and input to aseg2wm.m
m_func="'${matlab_func_dir}'"

echo "Using FreeSurfer: $FREESURFER_HOME"
echo "Subject: ${subjid}, Age: ${age} months, Starting from stage: ${stage}"

# Stage 1: Run FreeSurfer recon-all (first steps)
if [ ${stage} -le 1 ]; then
    echo "Stage 1: Running first steps of recon-all"
    date
    # -nofill stops recon-all at step 15 (https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all)
    #  -nofill skips step 15 (no filled.mgz), but recon-all still tries step 16+
    #  if you requested them (e.g., with -all), which then crash.
    #  error: mri_pretess: could not open ../mri/filled.mgz
    set +e  # Temporarily allow commands to fail
    recon-all -all -nofill -subjid ${subjid} -openmp ${num_jobs}
    set -e  # Re-enable exit on error
    # recon-all -all -subjid ${subjid} -nonuintensitycor -openmp ${num_jobs}
    date
fi

# Stage 2: Remove files based on initial FS run
if [ ${stage} -le 2 ]; then
    echo "Stage 2: Removing files based on initial FS run"
    date
    rm -rf ${fs_dir}/mri/transforms/* \
          ${fs_dir}/mri/orig_nu.mgz \
          ${fs_dir}/mri/mri_nu_correct.mni.log
    date
fi

# Stage 3: Setting up for infant FS
if [ ${stage} -le 3 ]; then
    echo "Stage 3: Setting up for infant FS"
    date
    mkdir -p ${ifs_dir}
    mri_convert -i ${fs_dir}/mri/orig.mgz -o ${ifs_dir}/mprage.nii.gz
    date
fi

# Stage 4: Running infant FS
if [ ${stage} -le 4 ]; then
    echo "Stage 4: Running infant FS"
    date
    ${script_dir}/iFS_wrap.sh ${ifs_parent_dir} ${subjid} ${age}
    echo "Using FreeSurfer: $FREESURFER_HOME"
    date
fi

# Stage 5: Merging iBEATv2 tissue segmentation with infant FS aseg
if [ ${stage} -le 5 ]; then
    echo "Stage 5: Merging iBEATv2 tissue segmentation with infant FS aseg and generating aseg.presurf and wm files for FS"
    # ibeat2aseg: merges iBEATv2 tissue segmentation (${m_ibeat}) with infant FreeSurfer aseg data (${m_ifs} and ${m_fs_mri}).
    #     converts iBEAT segmentations (gray/white/CSF) into FreeSurfer’s label format (aseg.mgz).
    # aseg2wm(...): Generates white-matter masks (and possibly aseg.presurf.mgz and wm.mgz)
    #     required by FreeSurfer’s surface reconstruction stages.
    date
    $matlab_bin_dir/matlab -nodesktop -nosplash -r \
        "addpath(${m_func}); \
         ibeat2aseg(${m_ibeat}, ${m_ifs}, ${m_fs_mri}); \
         aseg2wm(${m_aseg}); \
         exit;"
    date
fi

# Stage 6: Finishing FS recon-all -autorecon2-wm pipeline
if [ ${stage} -le 6 ]; then
    echo "Stage 6: Finishing FS recon-all -autorecon2-wm pipeline, including going back and performing some -autorecon1 steps using iFS files"
    date
    cp ${ifs_dir}/mri/transforms/talairach*.xfm ${fs_dir}/mri/transforms/
    ${script_dir}/fs_autorecon2_end.sh ${fs_dir} ${ifs_dir} ${subjid}
    date
fi

# Stage 7: Running FS recon-all -autorecon3 pipeline with adjustments
if [ ${stage} -le 7 ]; then
    echo "Stage 7: Running FS recon-all -autorecon3 pipeline with adjustments"
    date
    # make sure that the expert.opts is in the script directory (matlab_func_dir)
    ${script_dir}/fs_autorecon3_wrap.sh ${fs_dir} ${matlab_func_dir} ${subjid} ${num_jobs}
    date
fi

echo "Pipeline completed: ${subjid}"

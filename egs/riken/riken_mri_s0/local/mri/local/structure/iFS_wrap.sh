#!/bin/bash


# change path parameters for iFS (and not FS)

# module load ncf fsl/6.0.4-ncf
# export FREESURFER_HOME=/n/gaab_mri_l3/Lab/DMC-Gaab2/tools/tkt_tools/infant_freesurfer/freesurfer
# source $FREESURFER_HOME/SetUpFreeSurfer.sh
# export FS_LICENSE=/n/gaab_mri_l3/Lab/DMC-Gaab2/tools/tkt_tools/infant_freesurfer/freesurfer/license.txt
# export SUBJECTS_DIR=${1}


# Run infant FS
# Switch to v8 for infant_recon_all
export FREESURFER_HOME=/work02/home/bin-wu/.local/freesurfer/v8.1.0/8.1.0 && \
    export SUBJECTS_DIR=/work02/home/bin-wu/.local/freesurfer/v8.1.0/subjects && \
    source $FREESURFER_HOME/SetUpFreeSurfer.sh && \
    echo "Using FreeSurfer 8.1.0 (Subjects: $SUBJECTS_DIR)"

echo "export SUBJECTS_DIR=${1}"
echo "infant_recon_all --s ${2} --age ${3}"
export SUBJECTS_DIR=${1}
infant_recon_all --s ${2} --age ${3}

# Switch back to v7
export FREESURFER_HOME=/work02/home/bin-wu/.local/freesurfer/v7.3.2/7.3.2 && \
           export SUBJECTS_DIR=/work02/home/bin-wu/.local/freesurfer/v7.3.2/subjects && \
           source $FREESURFER_HOME/SetUpFreeSurfer.sh && \
           echo "Using FreeSurfer 7.3.2 (Subjects: $SUBJECTS_DIR)"

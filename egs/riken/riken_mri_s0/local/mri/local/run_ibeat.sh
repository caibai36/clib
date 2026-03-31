#!/bin/bash

# Set bash to 'debug' mode
set -euo pipefail

# General configuration
stage=8
gpu_device=2  # GPU device ID

# Data paths
id2nii=conf/mri/id2nii/new_england_id2nii_age_le_50.yaml
id2age=data_mri/new_england/t1w/id2age_rounded.yaml
mount_dir=/data02/share/bin-wu/data/human/brain/harvard_mri
processed_data_dir=/data02/share/bin-wu/data/human/brain/harvard_mri/processed/ibeat/age_le_50_months/new_england/
license_dir=/work02/home/bin-wu/.local/ibeat/License

# Configuration files
log_dir=logs/ibeat

# iBEAT parameters
docker_image=ibeatgroup/ibeat_v2:release210

# Parse options
. local/scripts/parse_options.sh || exit 1

# Create directories
mkdir -p ${log_dir} ${processed_data_dir}

# ============================
# Stage 0: Process all subjects
# ============================
if [ ${stage} -le 0 ]; then
    date
    echo "Stage 0: Processing MRI data with iBEAT v2..."
    
    # Read id2nii YAML file
    while IFS=': ' read -r id nii_path; do
        # Skip empty lines and comments
        [[ -z "${id}" || "${id}" =~ ^#.*$ ]] && continue
        
        # Trim whitespace
        id=$(echo ${id} | xargs)
        nii_path=$(echo ${nii_path} | xargs)
        
        # Get age from id2age YAML
        age_rounded=$(grep "^${id}:" ${id2age} | cut -d':' -f2 | xargs)
        
        if [ -z "${age_rounded}" ]; then
            echo "Warning: Could not find age for ${id}, skipping..."
            continue
        fi
        
        # Define paths
        t1w_input=${nii_path}
        output_dir=${processed_data_dir}/${id}
        
        # Check if T1w exists
        if [ ! -f "${t1w_input}" ]; then
            echo "Warning: T1w file not found for ${id}: ${t1w_input}"
            continue
        fi
        
        # Create output directory
        mkdir -p ${output_dir}
        
        # Compute relative paths from mount_dir for running ibeat
        t1w_relative=$(realpath --relative-to=${mount_dir} ${t1w_input})
        output_relative=$(realpath --relative-to=${mount_dir} ${output_dir})
        
        echo "Processing ${id}: age=${age_rounded} months"
        echo "  Input:  ${t1w_input}"
        echo "  Output: ${output_dir}"
        
        # Run iBEAT v2 (removed -it flags for non-interactive execution)
        docker run --rm \
            --gpus "device=${gpu_device}" \
            -v "${mount_dir}:/InfantData" \
            -v "${license_dir}:/InfantData/License" \
            --user $(id -u):$(id -g) \
            ${docker_image} \
            --t1 /InfantData/${t1w_relative} \
            --age ${age_rounded} \
            --out_dir /InfantData/${output_relative} \
            --sub_name ${id} \
            2>&1 | tee ${log_dir}/run_ibeat_${id}.log
        
        echo "Completed ${id}"
        echo "----------------------------------------"
        
    done < ${id2nii}
    
    date
    echo "Stage 0 completed: All subjects processed"
fi

echo "All stages completed!"

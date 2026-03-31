#!/bin/bash
INPUT_DIR="MBM_v3.0.1"
OUTPUT_DIR="MBM_v3.0.1_0.5mm"
mkdir -p $OUTPUT_DIR

# Anatomical templates (linear interpolation)
for file in ${INPUT_DIR}/template_*.nii.gz ${INPUT_DIR}/mask_*.nii.gz; do
    # Check if file exists (in case no matches)
    [ -f "$file" ] || continue
    
    # Get just the filename without path
    basename=$(basename "$file" .nii.gz)
    
    echo "Processing: $file"
    3dresample -dxyz 0.5 0.5 0.5 \
               -prefix ${OUTPUT_DIR}/${basename}_0.5mm.nii.gz \
               -input "$file"
done

# Atlas and segmentation files (nearest neighbor)
for file in ${INPUT_DIR}/atlas_*.nii.gz ${INPUT_DIR}/segmentation_*.nii.gz; do
    # Check if file exists (in case no matches)
    [ -f "$file" ] || continue
    
    # Get just the filename without path
    basename=$(basename "$file" .nii.gz)
    
    echo "Processing: $file"
    3dresample -dxyz 0.5 0.5 0.5 \
               -rmode NN \
               -prefix ${OUTPUT_DIR}/${basename}_0.5mm.nii.gz \
               -input "$file"
done

echo "Downsampling complete! Output in: $OUTPUT_DIR"

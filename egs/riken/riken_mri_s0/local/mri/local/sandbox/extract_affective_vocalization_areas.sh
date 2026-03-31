#!/bin/bash

# ==========================================
# MASK EXTRACTION FOR INFANT FREESURFER
# ==========================================
# Template: Desikan-Killiany Atlas (FreeSurfer standard cortical parcellation)
# Reference: Desikan et al. 2006, NeuroImage
# Atlas includes 34 cortical regions per hemisphere plus subcortical structures
# Verified label IDs from subject's actual segmentation output
# ==========================================

# Configuration
subjid="id1"
subjects_dir="$PWD/exp/mri/test_pure_ifs/ifs_outputs"
t1w_input="/data02/share/bin-wu/data/human/brain/harvard_mri/raw/new_england/ds006169-1.0.3/sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz"
output_dir="$subjects_dir/$subjid/masks"

# Setup FreeSurfer environment
export FREESURFER_HOME=/work02/home/bin-wu/.local/freesurfer/v8.1.0/8.1.0
export SUBJECTS_DIR=$subjects_dir
source $FREESURFER_HOME/SetUpFreeSurfer.sh

mkdir -p $output_dir

echo "==================================="
echo "Extracting Brain Region Masks"
echo "Subject: $subjid"
echo "Template: Desikan-Killiany Atlas"
echo "FreeSurfer Version: 8.1.0"
echo "==================================="
echo ""

# ==========================================
# 1. AUDITORY CORTEX
# ==========================================
# Auditory processing regions in temporal lobe
# Primary auditory cortex (A1): Heschl's gyrus (transverse temporal)
# Secondary auditory cortex: Superior temporal gyrus
# Auditory association cortex: Banks of superior temporal sulcus
# ==========================================

echo "Creating auditory cortex mask..."

# Left hemisphere auditory regions
# 1001: ctx-lh-bankssts (auditory association cortex)
# 1030: ctx-lh-superiortemporal (secondary auditory cortex)
# 1034: ctx-lh-transversetemporal (Heschl's gyrus - primary auditory cortex A1)
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz \
             --match 1001 1030 1034 \
             --o $output_dir/lh_auditory_cortex.mgz

# Right hemisphere auditory regions
# 2001: ctx-rh-bankssts
# 2030: ctx-rh-superiortemporal
# 2034: ctx-rh-transversetemporal
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz \
             --match 2001 2030 2034 \
             --o $output_dir/rh_auditory_cortex.mgz

# Combine both hemispheres
mri_concat $output_dir/lh_auditory_cortex.mgz \
           $output_dir/rh_auditory_cortex.mgz \
           --max --o $output_dir/bilateral_auditory_cortex.mgz

echo "[OK] Auditory cortex masks created"
echo "  - Left: 3 regions (bankssts, superiortemporal, transversetemporal)"
echo "  - Right: 3 regions (bankssts, superiortemporal, transversetemporal)"
echo ""

# ==========================================
# 2. PREFRONTAL CORTEX
# ==========================================
# Executive functions, decision making, working memory
# Includes Broca's area (language production): pars opercularis, triangularis, orbitalis
# Dorsolateral PFC: superior frontal, rostral middle frontal
# Orbitofrontal cortex: lateral and medial orbitofrontal
# ==========================================

echo "Creating prefrontal cortex mask..."

# Left hemisphere prefrontal regions
# 1003: ctx-lh-caudalmiddlefrontal (dorsolateral PFC)
# 1012: ctx-lh-lateralorbitofrontal (orbitofrontal cortex)
# 1014: ctx-lh-medialorbitofrontal (orbitofrontal cortex)
# 1018: ctx-lh-parsopercularis (Broca's area - pars opercularis)
# 1019: ctx-lh-parsorbitalis (Broca's area - pars orbitalis)
# 1020: ctx-lh-parstriangularis (Broca's area - pars triangularis)
# 1027: ctx-lh-rostralmiddlefrontal (dorsolateral PFC)
# 1028: ctx-lh-superiorfrontal (dorsolateral PFC)
# 1032: ctx-lh-frontalpole (anterior PFC)
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz \
             --match 1003 1012 1014 1018 1019 1020 1027 1028 1032 \
             --o $output_dir/lh_prefrontal_cortex.mgz

# Right hemisphere prefrontal regions
# 2003: ctx-rh-caudalmiddlefrontal
# 2012: ctx-rh-lateralorbitofrontal
# 2014: ctx-rh-medialorbitofrontal
# 2018: ctx-rh-parsopercularis
# 2019: ctx-rh-parsorbitalis
# 2020: ctx-rh-parstriangularis
# 2027: ctx-rh-rostralmiddlefrontal
# 2028: ctx-rh-superiorfrontal
# 2032: ctx-rh-frontalpole
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz \
             --match 2003 2012 2014 2018 2019 2020 2027 2028 2032 \
             --o $output_dir/rh_prefrontal_cortex.mgz

# Combine both hemispheres
mri_concat $output_dir/lh_prefrontal_cortex.mgz \
           $output_dir/rh_prefrontal_cortex.mgz \
           --max --o $output_dir/bilateral_prefrontal_cortex.mgz

echo "[OK] Prefrontal cortex masks created"
echo "  - Left: 9 regions (including Broca's area: parsopercularis, parstriangularis, parsorbitalis)"
echo "  - Right: 9 regions"
echo ""

# ==========================================
# 3. LIMBIC SYSTEM
# ==========================================
# Emotion, memory, motivation, learning
# Subcortical: Hippocampus (memory), Amygdala (emotion/fear)
# Cortical: Cingulate cortex (emotion regulation, decision making)
#           Parahippocampal gyrus (spatial memory)
#           Entorhinal cortex (memory gateway to hippocampus)
# ==========================================

echo "Creating limbic system mask..."

# Subcortical limbic structures from aseg (automatic segmentation)
# 17: Left-Hippocampus (memory formation, consolidation)
# 53: Right-Hippocampus
# 18: Left-Amygdala (emotion processing, fear conditioning)
# 54: Right-Amygdala
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz \
             --match 17 53 18 54 \
             --o $output_dir/subcortical_limbic.mgz

# Cortical limbic structures from aparc+aseg (cortical parcellation + segmentation)
# Cingulate cortex components:
# 1002: ctx-lh-caudalanteriorcingulate (emotion, pain processing)
# 1010: ctx-lh-isthmuscingulate (connects posterior and anterior cingulate)
# 1023: ctx-lh-posteriorcingulate (default mode network, episodic memory)
# 1026: ctx-lh-rostralanteriorcingulate (emotion regulation, conflict monitoring)
# 2002: ctx-rh-caudalanteriorcingulate
# 2010: ctx-rh-isthmuscingulate
# 2023: ctx-rh-posteriorcingulate
# 2026: ctx-rh-rostralanteriorcingulate
# Memory-related structures:
# 1006: ctx-lh-entorhinal (gateway to hippocampus, memory encoding)
# 1016: ctx-lh-parahippocampal (spatial memory, scene recognition)
# 2016: ctx-rh-parahippocampal
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz \
             --match 1002 1006 1010 1016 1023 1026 2002 2010 2016 2023 2026 \
             --o $output_dir/cortical_limbic.mgz

# Combine cortical and subcortical limbic structures
mri_concat $output_dir/subcortical_limbic.mgz \
           $output_dir/cortical_limbic.mgz \
           --max --o $output_dir/complete_limbic_system.mgz

echo "[OK] Limbic system masks created"
echo "  - Subcortical: 4 structures (bilateral hippocampus, amygdala)"
echo "  - Cortical: 11 regions (cingulate cortex, parahippocampal, entorhinal)"
echo ""

# ==========================================
# 4. CONVERT TO NIFTI FORMAT
# ==========================================
# Convert from FreeSurfer's MGZ format to standard NIfTI format
# NIfTI is more widely compatible with neuroimaging software
# ==========================================

echo "Converting masks to NIfTI format..."

for mask in $output_dir/bilateral_*.mgz $output_dir/complete_*.mgz; do
    if [ -f "$mask" ]; then
        basename=$(basename $mask .mgz)
        mri_convert $mask $output_dir/${basename}.nii.gz
        echo "  [OK] Converted: ${basename}.nii.gz"
    fi
done

echo ""

# ==========================================
# 5. TRANSFORM TO NATIVE T1 SPACE
# ==========================================
# Transform masks from FreeSurfer conformed space back to original T1 space
# FreeSurfer conforms images to 256^3 isotropic 1mm voxels
# This step returns masks to the original scan dimensions and orientation
# Uses header-based registration (--regheader) which is appropriate for same-subject data
# Nearest neighbor interpolation preserves binary mask values (0 or 1)
# ==========================================

echo "Transforming masks to native T1 space..."

for mask in $output_dir/bilateral_*.mgz $output_dir/complete_*.mgz; do
    if [ -f "$mask" ]; then
        basename=$(basename $mask .mgz)
        
        # Transform from FreeSurfer space to native T1 space
        # --mov: moving volume (FreeSurfer space mask)
        # --targ: target volume (original T1)
        # --regheader: register using header geometry information
        # --interp nearest: preserve binary mask values
        mri_vol2vol --mov $mask \
                    --targ $t1w_input \
                    --regheader \
                    --o $output_dir/${basename}_native.nii.gz \
                    --no-save-reg \
                    --interp nearest
        
        echo "  [OK] Transformed: ${basename}_native.nii.gz"
    fi
done

echo ""

# ==========================================
# 6. GENERATE SUMMARY STATISTICS
# ==========================================
# Calculate volume statistics for each mask
# Assumes 1mm^3 voxel size (FreeSurfer standard)
# ==========================================

echo "==================================="
echo "SUMMARY STATISTICS"
echo "==================================="
echo ""

for mask in $output_dir/bilateral_*.mgz $output_dir/complete_*.mgz; do
    if [ -f "$mask" ]; then
        basename=$(basename $mask .mgz)
        
        # Get volume statistics using mri_segstats
        # Extract number of voxels from output
        nvoxels=$(mri_segstats --i $mask --sum /tmp/mask_stats_${basename}.txt 2>&1 | \
                  grep -oP "(?<=Number of Voxels = )\d+" || echo "N/A")
        
        # Calculate volume in mm^3 and cm^3
        # 1 voxel = 1 mm^3 (FreeSurfer standard)
        # 1000 mm^3 = 1 cm^3
        if [ "$nvoxels" != "N/A" ]; then
            volume_mm3=$nvoxels
            volume_cm3=$(echo "scale=2; $nvoxels/1000" | bc)
            echo "${basename}:"
            echo "  Voxels: $nvoxels"
            echo "  Volume: ${volume_mm3} mm^3 (${volume_cm3} cm^3)"
        else
            echo "${basename}:"
            echo "  Statistics unavailable"
        fi
        echo ""
    fi
done

# ==========================================
# 7. CREATE INDIVIDUAL REGION MASKS
# ==========================================
# Extract individual regions for detailed analysis
# Useful for region-specific connectivity or morphometry studies
# ==========================================

echo "Creating individual region masks for each structure..."

# Create subdirectory for individual masks
mkdir -p $output_dir/individual_regions

# Auditory cortex - individual regions
echo "  Extracting individual auditory regions..."
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1001 \
             --o $output_dir/individual_regions/lh_bankssts.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1030 \
             --o $output_dir/individual_regions/lh_superiortemporal.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1034 \
             --o $output_dir/individual_regions/lh_transversetemporal.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 2001 \
             --o $output_dir/individual_regions/rh_bankssts.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 2030 \
             --o $output_dir/individual_regions/rh_superiortemporal.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 2034 \
             --o $output_dir/individual_regions/rh_transversetemporal.mgz

# Prefrontal cortex - Broca's area components
echo "  Extracting Broca's area components..."
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1018 \
             --o $output_dir/individual_regions/lh_parsopercularis_brocas.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1019 \
             --o $output_dir/individual_regions/lh_parsorbitalis_brocas.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1020 \
             --o $output_dir/individual_regions/lh_parstriangularis_brocas.mgz

# Limbic system - key structures
echo "  Extracting key limbic structures..."
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz --match 17 \
             --o $output_dir/individual_regions/left_hippocampus.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz --match 53 \
             --o $output_dir/individual_regions/right_hippocampus.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz --match 18 \
             --o $output_dir/individual_regions/left_amygdala.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz --match 54 \
             --o $output_dir/individual_regions/right_amygdala.mgz

echo "[OK] Individual region masks created in: individual_regions/"
echo ""

# ==========================================
# 8. CREATE REGION LABEL MAPPING FILE
# ==========================================
# Comprehensive reference of all label IDs used
# Based on Desikan-Killiany Atlas (FreeSurfer)
# Cortical labels: 1000-1035 (left), 2000-2035 (right)
# Subcortical labels: 1-99
# ==========================================

cat > $output_dir/label_mapping.txt << 'EOFMAP'
====================================================================================
LABEL MAPPING FOR EXTRACTED BRAIN REGIONS
====================================================================================
Template: Desikan-Killiany Atlas (FreeSurfer standard parcellation)
Reference: Desikan et al. 2006, NeuroImage, 31(3):968-980
Total regions: 34 cortical regions per hemisphere + subcortical structures
====================================================================================

AUDITORY CORTEX REGIONS
------------------------
Label   Region Name                          Hemisphere  Function
1001    ctx-lh-bankssts                      Left        Auditory association cortex
1030    ctx-lh-superiortemporal              Left        Secondary auditory cortex
1034    ctx-lh-transversetemporal            Left        Primary auditory cortex (Heschl's gyrus, A1)
2001    ctx-rh-bankssts                      Right       Auditory association cortex
2030    ctx-rh-superiortemporal              Right       Secondary auditory cortex
2034    ctx-rh-transversetemporal            Right       Primary auditory cortex (Heschl's gyrus, A1)

PREFRONTAL CORTEX REGIONS
--------------------------
Label   Region Name                          Hemisphere  Function
1003    ctx-lh-caudalmiddlefrontal           Left        Dorsolateral PFC, working memory
1012    ctx-lh-lateralorbitofrontal          Left        Orbitofrontal cortex, reward processing
1014    ctx-lh-medialorbitofrontal           Left        Orbitofrontal cortex, decision making
1018    ctx-lh-parsopercularis               Left        Broca's area (language production)
1019    ctx-lh-parsorbitalis                 Left        Broca's area (language production)
1020    ctx-lh-parstriangularis              Left        Broca's area (language production)
1027    ctx-lh-rostralmiddlefrontal          Left        Dorsolateral PFC, executive function
1028    ctx-lh-superiorfrontal               Left        Dorsolateral PFC, working memory
1032    ctx-lh-frontalpole                   Left        Anterior PFC, abstract reasoning
2003    ctx-rh-caudalmiddlefrontal           Right       Dorsolateral PFC
2012    ctx-rh-lateralorbitofrontal          Right       Orbitofrontal cortex
2014    ctx-rh-medialorbitofrontal           Right       Orbitofrontal cortex
2018    ctx-rh-parsopercularis               Right       Inferior frontal gyrus
2019    ctx-rh-parsorbitalis                 Right       Inferior frontal gyrus
2020    ctx-rh-parstriangularis              Right       Inferior frontal gyrus
2027    ctx-rh-rostralmiddlefrontal          Right       Dorsolateral PFC
2028    ctx-rh-superiorfrontal               Right       Dorsolateral PFC
2032    ctx-rh-frontalpole                   Right       Anterior PFC

LIMBIC SYSTEM REGIONS
----------------------
SUBCORTICAL STRUCTURES (from aseg.mgz)
Label   Region Name                          Hemisphere  Function
17      Left-Hippocampus                     Left        Memory formation, spatial navigation
53      Right-Hippocampus                    Right       Memory formation, spatial navigation
18      Left-Amygdala                        Left        Emotion processing, fear conditioning
54      Right-Amygdala                       Right       Emotion processing, fear conditioning

CORTICAL LIMBIC STRUCTURES (from aparc+aseg.mgz)
Label   Region Name                          Hemisphere  Function
1002    ctx-lh-caudalanteriorcingulate       Left        Emotion, autonomic function, pain
1006    ctx-lh-entorhinal                    Left        Memory gateway, spatial memory
1010    ctx-lh-isthmuscingulate              Left        Connects posterior/anterior cingulate
1016    ctx-lh-parahippocampal               Left        Spatial memory, scene recognition
1023    ctx-lh-posteriorcingulate            Left        Default mode network, episodic memory
1026    ctx-lh-rostralanteriorcingulate      Left        Emotion regulation, conflict monitoring
2002    ctx-rh-caudalanteriorcingulate       Right       Emotion, autonomic function
2010    ctx-rh-isthmuscingulate              Right       Cingulate connection
2016    ctx-rh-parahippocampal               Right       Spatial memory, scene recognition
2023    ctx-rh-posteriorcingulate            Right       Default mode network
2026    ctx-rh-rostralanteriorcingulate      Right       Emotion regulation

====================================================================================
NOTES:
- Cortical labels follow pattern: 1000 + region_id (left), 2000 + region_id (right)
- Subcortical labels are from automatic subcortical segmentation (aseg.mgz)
- All measurements assume 1mm^3 isotropic voxels (FreeSurfer standard)
- Template reference: $FREESURFER_HOME/average/mni305.cor.mgz
====================================================================================
EOFMAP

echo "[OK] Label mapping file created: label_mapping.txt"
echo ""

# ==========================================
# 9. CREATE VISUALIZATION SCRIPT
# ==========================================
# Generate script for easy visualization in FreeView
# FreeView is FreeSurfer's integrated visualization tool
# ==========================================

cat > $output_dir/view_masks.sh << 'EOFVIEW'
#!/bin/bash

# Visualization script for extracted brain masks
# Opens FreeView with all masks overlaid on the original T1
# FreeView is part of FreeSurfer distribution

MASK_DIR="$(cd "$(dirname "$0")" && pwd)"
T1_INPUT="/data02/share/bin-wu/data/human/brain/harvard_mri/raw/new_england/ds006169-1.0.3/sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz"

echo "==================================="
echo "Opening FreeView Visualization"
echo "==================================="
echo ""
echo "Color coding:"
echo "  [RED]   Auditory Cortex"
echo "  [GREEN] Prefrontal Cortex"
echo "  [BLUE]  Limbic System"
echo ""
echo "Navigation tips:"
echo "  - Use mouse wheel to zoom"
echo "  - Click and drag to pan"
echo "  - Right-click region to see label info"
echo "  - Press 'h' for help menu"
echo ""

# Load T1 image with three mask overlays
# Each mask uses different colormap for distinction
# Opacity set to 0.4 for semi-transparent overlay
freeview -v $T1_INPUT \
         $MASK_DIR/bilateral_auditory_cortex_native.nii.gz:colormap=heat:opacity=0.4 \
         $MASK_DIR/bilateral_prefrontal_cortex_native.nii.gz:colormap=gecolor:opacity=0.4 \
         $MASK_DIR/complete_limbic_system_native.nii.gz:colormap=jet:opacity=0.4
EOFVIEW

chmod +x $output_dir/view_masks.sh

echo "[OK] Visualization script created: view_masks.sh"
echo ""

# ==========================================
# 10. FINAL SUMMARY
# ==========================================

echo "==================================="
echo "EXTRACTION COMPLETE!"
echo "==================================="
echo ""
echo "Template Used: Desikan-Killiany Atlas (FreeSurfer)"
echo "FreeSurfer Version: 8.1.0"
echo "Output directory: $output_dir"
echo ""
echo "Generated files:"
echo ""
echo "Main masks (FreeSurfer space - 256^3 1mm isotropic):"
ls -lh $output_dir/bilateral_*.mgz $output_dir/complete_*.mgz 2>/dev/null | \
   awk '{printf "  %-50s %8s\n", $9, $5}'
echo ""
echo "Main masks (Native T1 space - original scan dimensions):"
ls -lh $output_dir/*_native.nii.gz 2>/dev/null | \
   awk '{printf "  %-50s %8s\n", $9, $5}'
echo ""
echo "Individual region masks:"
num_regions=$(ls $output_dir/individual_regions/*.mgz 2>/dev/null | wc -l)
echo "  $num_regions region-specific masks in individual_regions/"
echo ""
echo "Reference files:"
echo "  - label_mapping.txt: Complete label ID reference with functional descriptions"
echo "  - view_masks.sh: Visualization script for FreeView"
echo ""
echo "To visualize masks:"
echo "  $output_dir/view_masks.sh"
echo ""
echo "All masks are ready for analysis!"
echo ""

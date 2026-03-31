#!/bin/bash

# ==========================================
# MARMOSET VOCALIZATION ANALYSIS - SETUP
# ==========================================
# Creates directory structure and organizes data
# Location: /work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_mri_s0/local/fmri/local/
# ==========================================

# ==========================================
# PATHS CONFIGURATION
# ==========================================

# Raw data location
# https://www.marmosetbrainconnectome.org/download.html
RAW_DATA_DIR="/data02/share/bin-wu/data/marmoset/brain/nih_uwo/nih/NIH-data"

# MBM template directory (0.5mm downsampled)
# https://gitlab.com/cfmm/marmoset-connectivity/-/blob/master/registration_MASTER.sh
# Suggests: "downsampling MBM template https://marmosetbrainmapping.org/atlas.html#v3 to .5mm isotropic (e.g., use 3dresample)"
MBM_TEMPLATE_DIR="/data02/share/bin-wu/data/marmoset/brain/nih_uwo/marmoset_brain_mapping_v3/Marmoset_Brain_Mappping_v3.0.1/MBM_v3.0.1_0.5mm"

# Original marmoset_connectivity scripts location (git clone official code)
# https://gitlab.com/cfmm/marmoset-connectivity
ORIGINAL_SCRIPTS_DIR="/work02/home/bin-wu/workspace/projects/tests/test_fmri/marmoset_connectivity"

# Working base directory for all outputs
WORK_BASE="/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_mri_s0/exp/mri/sandbox/marmoset_registration"

# Scripts directory (where this script is located)
SCRIPTS_DIR="/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_mri_s0/local/fmri/local"

echo "=========================================="
echo "MARMOSET VOCALIZATION ANALYSIS - SETUP"
echo "=========================================="
echo ""
echo "Scripts location: $SCRIPTS_DIR"
echo "Working directory: $WORK_BASE"
echo ""

# ==========================================
# STEP 1: Create directory structure
# ==========================================

echo "Step 1: Creating directory structure..."
echo ""

mkdir -p $WORK_BASE/{raw_bold,preprocessed,anatomical,template_space,correlation,temp,roi_masks,logs,results}

echo "  Created directories in: $WORK_BASE"
echo "    - raw_bold/         : Raw BOLD data (copied from NIH dataset)"
echo "    - preprocessed/     : Preprocessed BOLD output (from rs_MASTER.sh)"
echo "    - anatomical/       : T2 and mask files per subject"
echo "    - template_space/   : Intermediate registration outputs"
echo "    - correlation/      : Final connectivity-ready files"
echo "    - temp/             : Temporary files"
echo "    - roi_masks/        : Vocalization network ROI masks"
echo "    - logs/             : Processing logs"
echo "    - results/          : Final analysis results"
echo ""

# ==========================================
# STEP 2: Copy/link raw BOLD data and create symbolic links
# ==========================================

echo "Step 2: Linking raw BOLD data and creating preprocessing-compatible symlinks..."
echo ""

# NIH subjects
SUBJECTS=(m6 m7 m8 m9 m10 m11 m12 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32)

for subj in "${SUBJECTS[@]}"; do
    if [ -d "$RAW_DATA_DIR/$subj" ]; then
        echo "  Processing $subj..."

        # Create subject directories
        mkdir -p $WORK_BASE/raw_bold/$subj
        mkdir -p $WORK_BASE/anatomical/$subj
        mkdir -p $WORK_BASE/preprocessed/$subj

        # Link BOLD files to working directory
        if [ -f "$RAW_DATA_DIR/$subj/BOLD_up_1.nii.gz" ]; then
            ln -sf "$RAW_DATA_DIR/$subj"/BOLD_*.nii.gz "$WORK_BASE/raw_bold/$subj/"
            ln -sf "$RAW_DATA_DIR/$subj"/SEEPI_*.nii.gz "$WORK_BASE/raw_bold/$subj/" 2>/dev/null
            bold_count=$(ls $WORK_BASE/raw_bold/$subj/BOLD_*.nii.gz 2>/dev/null | wc -l)
            echo "    ✓ Linked $bold_count BOLD files to working directory"
            
            # Create preprocessing-compatible symlinks in working directory
            cd $WORK_BASE/raw_bold/$subj
            
            # Count runs
            num_runs=$(ls BOLD_up_*.nii.gz 2>/dev/null | wc -l)
            
            # Create symlinks with expected naming convention
            for i in $(seq 1 $num_runs); do
                if [ -f "BOLD_up_${i}.nii.gz" ]; then
                    ln -sf "BOLD_up_${i}.nii.gz" "${subj}_rest-up_bold_${i}.nii.gz"
                fi
                
                if [ -f "BOLD_down_${i}.nii.gz" ]; then
                    ln -sf "BOLD_down_${i}.nii.gz" "${subj}_rest-down_bold_${i}.nii.gz"
                fi
            done
            
            # Create SEEPI symlinks if they exist
            if [ -f "SEEPI_up.nii.gz" ]; then
                ln -sf "SEEPI_up.nii.gz" "${subj}_SEEPI_up.nii.gz"
            fi
            
            if [ -f "SEEPI_down.nii.gz" ]; then
                ln -sf "SEEPI_down.nii.gz" "${subj}_SEEPI_down.nii.gz"
            fi
            
            echo "    ✓ Created preprocessing symlinks (${num_runs} runs)"
        fi

        # Copy anatomical files
        if [ -f "$RAW_DATA_DIR/$subj/InplaneT2.nii.gz" ]; then
            cp "$RAW_DATA_DIR/$subj/InplaneT2.nii.gz" "$WORK_BASE/anatomical/$subj/"
            echo "    ✓ Copied InplaneT2.nii.gz"
        else
            echo "    ✗ InplaneT2.nii.gz not found"
        fi

        if [ -f "$RAW_DATA_DIR/$subj/mask.nii.gz" ]; then
            cp "$RAW_DATA_DIR/$subj/mask.nii.gz" "$WORK_BASE/anatomical/$subj/"
            echo "    ✓ Copied mask.nii.gz"
        else
            echo "    ✗ mask.nii.gz not found"
        fi
    else
        echo "  ⚠ Subject directory not found: $RAW_DATA_DIR/$subj"
    fi
done

echo ""

# ==========================================
# STEP 3: Verify template files
# ==========================================

echo "Step 3: Verifying MBM template files..."
echo ""

REQUIRED_FILES=(
    "template_T2w_brain_0.5mm.nii.gz"
    "segmentation_three_types_prob_1_gray_0.5mm.nii.gz"
    "segmentation_three_types_prob_2_white_0.5mm.nii.gz"
    "segmentation_three_types_prob_3_csf_0.5mm.nii.gz"
    "atlas_MBM_cortex_vPaxinos_0.5mm.nii.gz"
    "atlas_MBM_subcortical_beta_0.5mm.nii.gz"
)

MISSING=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$MBM_TEMPLATE_DIR/$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ MISSING: $file"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -gt 0 ]; then
    echo ""
    echo "  ERROR: $MISSING template files missing!"
    exit 1
else
    echo ""
    echo "  ✓ All template files found!"
fi

echo ""

# ==========================================
# STEP 4: Create subject metadata file
# ==========================================

echo "Step 4: Creating subject metadata file..."
echo ""

METADATA_SRC="/data02/share/bin-wu/data/marmoset/brain/nih_uwo/nih_uwo_meta.csv"

if [ -f "$METADATA_SRC" ]; then
    cp "$METADATA_SRC" "$WORK_BASE/subject_metadata.csv"
    echo "  ✓ Metadata copied"
    echo ""
    echo "  Subject summary (NIH 7T only):"
    tail -n +2 "$WORK_BASE/subject_metadata.csv" | grep "NIH" | \
        awk -F',' '{printf "    %-10s Age: %3smo, Sex: %s, Runs: %s\n", $1, $2, $3, $7}' | head -15
    echo "    ..."
else
    echo "  ⚠ Metadata file not found: $METADATA_SRC"
fi

echo ""

# ==========================================
# STEP 5: Create configuration file
# ==========================================

cat > $WORK_BASE/config_paths.sh << EOFCONFIG
#!/bin/bash
# ==========================================
# MARMOSET VOCALIZATION ANALYSIS - PATH CONFIG
# ==========================================
# Source this file in all processing scripts
# Usage: source config_paths.sh
# ==========================================

# Raw data
export RAW_DATA_DIR="$RAW_DATA_DIR"

# MBM templates
export MBM_TEMPLATE_DIR="$MBM_TEMPLATE_DIR"

# Original scripts
export ORIGINAL_SCRIPTS_DIR="$ORIGINAL_SCRIPTS_DIR"

# Working directories
export WORK_BASE="$WORK_BASE"
export SCRIPTS_DIR="$SCRIPTS_DIR"

# Subdirectories
export RAW_BOLD_DIR="\${WORK_BASE}/raw_bold"
export PREPROCESSED_DIR="\${WORK_BASE}/preprocessed"
export ANATOMICAL_DIR="\${WORK_BASE}/anatomical"
export TEMPLATE_SPACE_DIR="\${WORK_BASE}/template_space"
export CORRELATION_DIR="\${WORK_BASE}/correlation"
export TEMP_DIR="\${WORK_BASE}/temp"
export ROI_MASKS_DIR="\${WORK_BASE}/roi_masks"
export LOGS_DIR="\${WORK_BASE}/logs"
export RESULTS_DIR="\${WORK_BASE}/results"

# Template files (specific)
export TEMPLATE_BRAIN="\${MBM_TEMPLATE_DIR}/template_T2w_brain_0.5mm.nii.gz"
export TEMPLATE_GRAY="\${MBM_TEMPLATE_DIR}/segmentation_three_types_prob_1_gray_0.5mm.nii.gz"
export TEMPLATE_WHITE="\${MBM_TEMPLATE_DIR}/segmentation_three_types_prob_2_white_0.5mm.nii.gz"
export TEMPLATE_CSF="\${MBM_TEMPLATE_DIR}/segmentation_three_types_prob_3_csf_0.5mm.nii.gz"

# Atlas files
export ATLAS_CORTICAL="\${MBM_TEMPLATE_DIR}/atlas_MBM_cortex_vPaxinos_0.5mm.nii.gz"
export ATLAS_SUBCORTICAL="\${MBM_TEMPLATE_DIR}/atlas_MBM_subcortical_beta_0.5mm.nii.gz"

# Subject list (NIH only)
export NIH_SUBJECTS=(m6 m7 m8 m9 m10 m11 m12 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32)

echo "Configuration loaded:"
echo "  Working base: \$WORK_BASE"
echo "  Scripts: \$SCRIPTS_DIR"
echo "  Templates: \$MBM_TEMPLATE_DIR"
EOFCONFIG

chmod +x $WORK_BASE/config_paths.sh

echo "  ✓ Configuration file created: $WORK_BASE/config_paths.sh"
echo ""

# ==========================================
# STEP 6: Create status checker
# ==========================================

cat > $WORK_BASE/check_status.sh << 'EOFSTATUS'
#!/bin/bash

source $(dirname $0)/config_paths.sh

echo "=========================================="
echo "MARMOSET REGISTRATION STATUS"
echo "=========================================="
echo ""

echo "Working directory: $WORK_BASE"
echo ""

echo "1. Raw BOLD data:"
ls -d $RAW_BOLD_DIR/m* 2>/dev/null | wc -l | xargs echo "   Subjects with raw data:"
ls $RAW_BOLD_DIR/m6/BOLD_*.nii.gz 2>/dev/null | wc -l | xargs echo "   Example (m6) BOLD files:"

echo ""
echo "2. Anatomical files:"
ls -d $ANATOMICAL_DIR/m* 2>/dev/null | wc -l | xargs echo "   Subjects ready:"

echo ""
echo "3. Preprocessed BOLD:"
ls $PREPROCESSED_DIR/m*/errts.*.nii.gz 2>/dev/null | wc -l | xargs echo "   Preprocessed files:"

echo ""
echo "4. Template space (registered):"
ls $CORRELATION_DIR/*_to_template_0.5mm_masked_gm.nii.gz 2>/dev/null | wc -l | xargs echo "   Registered files:"

echo ""
echo "5. ROI masks:"
ls $ROI_MASKS_DIR/*.nii.gz 2>/dev/null | wc -l | xargs echo "   ROI files:"

echo ""
echo "Disk usage:"
du -sh $WORK_BASE
du -sh $WORK_BASE/* | sort -h

echo ""
EOFSTATUS

chmod +x $WORK_BASE/check_status.sh

echo "  ✓ Status checker created: $WORK_BASE/check_status.sh"
echo ""

# ==========================================
# STEP 7: Create README
# ==========================================

cat > $WORK_BASE/README.txt << 'EOFREADME'
MARMOSET VOCALIZATION DEVELOPMENT ANALYSIS
==========================================

Directory Structure:
--------------------
raw_bold/        - Raw BOLD data from NIH dataset (linked)
preprocessed/    - Preprocessed BOLD from run_preprocessing.sh
anatomical/      - T2 and mask files (ready)
template_space/  - Intermediate registration files
correlation/     - Final connectivity-ready files
temp/           - Temporary processing files
roi_masks/      - Vocalization network ROI masks
logs/           - All processing logs
results/        - Final analysis outputs

Complete Workflow:
------------------

STEP 1: Setup (DONE by running setup_directories_and_data.sh)
   ✓ Directories created
   ✓ Raw data linked
   ✓ Anatomical files copied
   ✓ Templates verified

STEP 2: Preprocess BOLD data
   cd local/fmri/local
   bash run_preprocessing.sh m6    # Test with one subject
   bash run_preprocessing.sh all   # Process all subjects

   Output: preprocessed/m6/errts.m6_u_bold_1.tproject.nii.gz
                        /m6_u_bold_1.mean.nii.gz

STEP 3: Register to template
   bash run_registration.sh m6     # Test with one subject
   bash run_registration.sh all    # Process all subjects

   Output: correlation/m6_u_bold_1_to_template_0.5mm_masked_gm.nii.gz

STEP 4: Extract vocalization ROIs
   bash run_stage1_rois.sh

STEP 5: Extract ROI time series
   python run_stage4_timeseries.py

STEP 6: Compute connectivity
   python run_stage5_connectivity.py

STEP 7: Developmental analysis
   python run_stage6_development.py

STEP 8: Cross-species comparison
   python run_stage7_comparison.py

Check Status:
-------------
bash check_status.sh

Configuration:
--------------
All paths are in: config_paths.sh
Source this in any custom scripts: source config_paths.sh

Logs:
-----
All logs saved to: logs/
EOFREADME

echo "  ✓ README created: $WORK_BASE/README.txt"
echo ""

# ==========================================
# SUMMARY
# ==========================================

echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Working directory: $WORK_BASE"
echo ""
echo "Files ready:"
echo "  ✓ Raw BOLD data linked ($(ls -d $WORK_BASE/raw_bold/m* 2>/dev/null | wc -l) subjects)"
echo "  ✓ Anatomical files copied ($(ls -d $WORK_BASE/anatomical/m* 2>/dev/null | wc -l) subjects)"
echo "  ✓ Configuration: $WORK_BASE/config_paths.sh"
echo "  ✓ Status checker: $WORK_BASE/check_status.sh"
echo ""
echo "Next step: Preprocess BOLD data"
echo "  cd $SCRIPTS_DIR"
echo "  bash run_preprocessing.sh m6"
echo ""
echo "For detailed instructions, see:"
echo "  - $WORK_BASE/README.txt"
echo "  - $SCRIPTS_DIR/PREPROCESSING_GUIDE.md"
echo ""

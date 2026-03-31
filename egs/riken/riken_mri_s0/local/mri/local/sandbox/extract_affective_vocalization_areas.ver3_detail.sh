#!/bin/bash

# ==========================================
# AFFECTIVE VOCALIZATION NETWORK EXTRACTION
# ==========================================
# Extracts brain regions relevant to emotional vocalization production and perception
# Reference: Jurgens (2002, 2009); Petkov & Jarvis (2012); Ackermann et al. (2014)
# ==========================================

# Configuration
t1w_input="/data02/share/bin-wu/data/human/brain/harvard_mri/raw/new_england/ds006169-1.0.3/sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz"
subjid="id1"

# Processed results at $fs_subject_dir/$subjid; use absolute path required by infant_all
subjects_dir="$PWD/exp/mri/test_pure_ifs/ifs_outputs"

. ./local/scripts/parse_options.sh || exit 1

# Setup FreeSurfer
export FREESURFER_HOME=/work02/home/bin-wu/.local/freesurfer/v8.1.0/8.1.0
export SUBJECTS_DIR=$subjects_dir
source $FREESURFER_HOME/SetUpFreeSurfer.sh

output_dir="$subjects_dir/$subjid/masks/vocalization_network"
mkdir -p $output_dir
mkdir -p $output_dir/details/vocal_motor
mkdir -p $output_dir/details/cerebellum
mkdir -p $output_dir/details/limbic_vocalization
mkdir -p $output_dir/details/auditory_vocal
mkdir -p $output_dir/details/basal_ganglia

echo "==========================================="
echo "AFFECTIVE VOCALIZATION NETWORK EXTRACTION"
echo "Subject: $subjid"
echo "Template: Desikan-Killiany Atlas"
echo "==========================================="
echo ""

# ==========================================
# 0. COPY AND PREPARE T1 IMAGES
# ==========================================

echo "Preparing T1 reference images..."

# Get T1 basename
t1_basename=$(basename $t1w_input)
t1_name="${t1_basename%.nii.gz}"

# Copy raw T1
cp $t1w_input $output_dir/${t1_name}.nii.gz
echo "  [OK] Raw T1: ${t1_name}.nii.gz"

# Copy skull-stripped brain (FreeSurfer brainmask - this is intensity image, not binary)
if [ -f "$SUBJECTS_DIR/$subjid/mri/brainmask.mgz" ]; then
    mri_convert $SUBJECTS_DIR/$subjid/mri/brainmask.mgz \
                $output_dir/${t1_name}.skullstripped.nii.gz
    echo "  [OK] Skull-stripped T1: ${t1_name}.skullstripped.nii.gz (FreeSurfer brainmask)"
fi

# Copy brain-only image (FreeSurfer brain.mgz - this is also intensity image)
if [ -f "$SUBJECTS_DIR/$subjid/mri/brain.mgz" ]; then
    mri_convert $SUBJECTS_DIR/$subjid/mri/brain.mgz \
                $output_dir/${t1_name}.brain.nii.gz
    echo "  [OK] Brain-only T1: ${t1_name}.brain.nii.gz (FreeSurfer brain)"
fi

# Create whole brain MASK (binary)
if [ -f "$SUBJECTS_DIR/$subjid/mri/brainmask.mgz" ]; then
    mri_binarize --i $SUBJECTS_DIR/$subjid/mri/brainmask.mgz \
                 --min 0.5 \
                 --o $output_dir/brain_mask.mgz
    mri_convert $output_dir/brain_mask.mgz $output_dir/brain_mask.nii.gz
    rm -f $output_dir/brain_mask.mgz
    echo "  [OK] Brain mask: brain_mask.nii.gz (binary mask, 0/1 values)"
fi

# Create cerebrum mask (cortex + white matter, excluding cerebellum) - BINARY
echo "  Creating cerebrum mask (excluding cerebellum)..."
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz \
             --match 2 41 \
             --o $output_dir/cerebral_wm.mgz

mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz \
             --match 1000 1001 1002 1003 1005 1006 1007 1008 1009 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 1030 1031 1032 1033 1034 1035 \
             --match 2000 2001 2002 2003 2005 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026 2027 2028 2029 2030 2031 2032 2033 2034 2035 \
             --o $output_dir/cortex.mgz

mri_concat $output_dir/cerebral_wm.mgz \
           $output_dir/cortex.mgz \
           --max --o $output_dir/cerebrum_mask.mgz

mri_convert $output_dir/cerebrum_mask.mgz $output_dir/cerebrum_mask.nii.gz
echo "  [OK] Cerebrum mask: cerebrum_mask.nii.gz (binary mask, cortex + WM, no cerebellum)"

# Clean up temporary files
rm -f $output_dir/cerebral_wm.mgz $output_dir/cortex.mgz $output_dir/cerebrum_mask.mgz

echo ""

# ==========================================
# 1. VOCAL MOTOR SYSTEM (CORTICAL ONLY)
# ==========================================
# Primary motor, premotor, supplementary motor areas
# Controls articulatory movements, vocalization execution, and timing
# ==========================================

echo "Creating vocal motor system mask (cortical only)..."

# Left hemisphere cortical motor regions
# 1018: ctx-lh-parsopercularis (Broca's area - motor speech)
# 1020: ctx-lh-parstriangularis (Broca's area - speech planning)
# 1024: ctx-lh-precentral (primary motor cortex - laryngeal/articulatory)
# 1028: ctx-lh-superiorfrontal (supplementary motor area - SMA)
# 1003: ctx-lh-caudalmiddlefrontal (premotor cortex)
# 1031: ctx-lh-supramarginal (inferior parietal - phonemic/word formation)
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz \
             --match 1018 1020 1024 1028 1003 1031 \
             --o $output_dir/details/vocal_motor/lh_vocal_motor_cortical.mgz

# Right hemisphere cortical motor regions
# 2018: ctx-rh-parsopercularis
# 2020: ctx-rh-parstriangularis
# 2024: ctx-rh-precentral
# 2028: ctx-rh-superiorfrontal (SMA)
# 2003: ctx-rh-caudalmiddlefrontal
# 2031: ctx-rh-supramarginal
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz \
             --match 2018 2020 2024 2028 2003 2031 \
             --o $output_dir/details/vocal_motor/rh_vocal_motor_cortical.mgz

# Combine cortical hemispheres
mri_concat $output_dir/details/vocal_motor/lh_vocal_motor_cortical.mgz \
           $output_dir/details/vocal_motor/rh_vocal_motor_cortical.mgz \
           --max --o $output_dir/vocal_motor_cortical.mgz

# Extract individual cortical components
echo "  Extracting individual motor components..."
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1018 \
             --o $output_dir/details/vocal_motor/lh_parsopercularis_brocas.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1020 \
             --o $output_dir/details/vocal_motor/lh_parstriangularis_brocas.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1024 \
             --o $output_dir/details/vocal_motor/lh_precentral_M1.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1028 \
             --o $output_dir/details/vocal_motor/lh_superiorfrontal_SMA.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1003 \
             --o $output_dir/details/vocal_motor/lh_caudalmiddlefrontal_premotor.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1031 \
             --o $output_dir/details/vocal_motor/lh_supramarginal_IPL.mgz

mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 2018 \
             --o $output_dir/details/vocal_motor/rh_parsopercularis.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 2020 \
             --o $output_dir/details/vocal_motor/rh_parstriangularis.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 2024 \
             --o $output_dir/details/vocal_motor/rh_precentral_M1.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 2028 \
             --o $output_dir/details/vocal_motor/rh_superiorfrontal_SMA.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 2003 \
             --o $output_dir/details/vocal_motor/rh_caudalmiddlefrontal_premotor.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 2031 \
             --o $output_dir/details/vocal_motor/rh_supramarginal_IPL.mgz

echo "[OK] Vocal motor system mask created (12 cortical components)"
echo ""

# ==========================================
# 2. CEREBELLUM VOCAL CONTROL (SEPARATE)
# ==========================================
# Motor timing and coordination
# ==========================================

echo "Creating cerebellum vocal control mask..."

# Cerebellar structures (motor timing and coordination)
# 8: Left-Cerebellum-Cortex
# 47: Right-Cerebellum-Cortex
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz \
             --match 8 47 \
             --o $output_dir/cerebellum_vocal.mgz

# Extract cerebellar components
echo "  Extracting individual cerebellar components..."
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz --match 8 \
             --o $output_dir/details/cerebellum/left_cerebellum_cortex.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz --match 47 \
             --o $output_dir/details/cerebellum/right_cerebellum_cortex.mgz

echo "[OK] Cerebellum vocal control mask created (2 components)"
echo ""

# ==========================================
# 3. LIMBIC VOCALIZATION SYSTEM
# ==========================================
# Emotional drive and affective content of vocalizations
# Critical for spontaneous emotional vocalizations (crying, laughing)
# ==========================================

echo "Creating limbic vocalization system mask..."

# Subcortical limbic structures
# 18: Left-Amygdala (emotional valence, fear/distress calls)
# 54: Right-Amygdala
# 26: Left-Accumbens (reward, motivation for vocal communication)
# 58: Right-Accumbens
# 10: Left-Thalamus (sensorimotor relay, emotional modulation)
# 49: Right-Thalamus
# 16: Brain-Stem (PAG, nucleus ambiguus, reticular formation)
# 28: Left-VentralDC (hypothalamus, substantia nigra - autonomic drive)
# 60: Right-VentralDC (hypothalamus, substantia nigra - gating)
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz \
             --match 18 54 26 58 10 49 16 28 60 \
             --o $output_dir/details/limbic_vocalization/subcortical_limbic_vocal.mgz

# Cortical limbic structures - anterior cingulate + orbitofrontal
# 1002: ctx-lh-caudalanteriorcingulate (emotional vocalization initiation)
# 1026: ctx-lh-rostralanteriorcingulate (vocalization drive, call production)
# 2002: ctx-rh-caudalanteriorcingulate
# 2026: ctx-rh-rostralanteriorcingulate
# 1014: ctx-lh-medialorbitofrontal (emotional regulation, social decision-making)
# 2014: ctx-rh-medialorbitofrontal
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz \
             --match 1002 1026 2002 2026 1014 2014 \
             --o $output_dir/details/limbic_vocalization/cortical_limbic_vocal.mgz

# Combine limbic components
mri_concat $output_dir/details/limbic_vocalization/subcortical_limbic_vocal.mgz \
           $output_dir/details/limbic_vocalization/cortical_limbic_vocal.mgz \
           --max --o $output_dir/limbic_vocalization.mgz

# Extract individual subcortical components
echo "  Extracting individual limbic components..."
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz --match 18 \
             --o $output_dir/details/limbic_vocalization/left_amygdala.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz --match 54 \
             --o $output_dir/details/limbic_vocalization/right_amygdala.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz --match 26 \
             --o $output_dir/details/limbic_vocalization/left_accumbens.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz --match 58 \
             --o $output_dir/details/limbic_vocalization/right_accumbens.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz --match 10 \
             --o $output_dir/details/limbic_vocalization/left_thalamus.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz --match 49 \
             --o $output_dir/details/limbic_vocalization/right_thalamus.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz --match 16 \
             --o $output_dir/details/limbic_vocalization/brainstem.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz --match 28 \
             --o $output_dir/details/limbic_vocalization/left_ventraldc_hypothalamus.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz --match 60 \
             --o $output_dir/details/limbic_vocalization/right_ventraldc_hypothalamus.mgz

# Extract individual cortical components
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1002 \
             --o $output_dir/details/limbic_vocalization/lh_caudalanteriorcingulate_ACC.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1026 \
             --o $output_dir/details/limbic_vocalization/lh_rostralanteriorcingulate_ACC.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 2002 \
             --o $output_dir/details/limbic_vocalization/rh_caudalanteriorcingulate_ACC.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 2026 \
             --o $output_dir/details/limbic_vocalization/rh_rostralanteriorcingulate_ACC.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1014 \
             --o $output_dir/details/limbic_vocalization/lh_medialorbitofrontal_OFC.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 2014 \
             --o $output_dir/details/limbic_vocalization/rh_medialorbitofrontal_OFC.mgz

echo "[OK] Limbic vocalization system mask created (15 components)"
echo ""

# ==========================================
# 4. AUDITORY-VOCAL INTEGRATION
# ==========================================
# Auditory feedback processing and voice perception
# Critical for vocal learning and self-monitoring
# ==========================================

echo "Creating auditory-vocal integration mask..."

# Left hemisphere auditory regions
# 1034: ctx-lh-transversetemporal (Heschl's gyrus - primary auditory, auditory feedback)
# 1030: ctx-lh-superiortemporal (secondary auditory, voice processing)
# 1001: ctx-lh-bankssts (superior temporal sulcus - social voice perception)
# 1015: ctx-lh-middletemporal (auditory association)
# 1035: ctx-lh-insula (auditory-motor integration, speech perception)
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz \
             --match 1034 1030 1001 1015 1035 \
             --o $output_dir/details/auditory_vocal/lh_auditory_vocal.mgz

# Right hemisphere auditory regions
# 2034: ctx-rh-transversetemporal
# 2030: ctx-rh-superiortemporal
# 2001: ctx-rh-bankssts
# 2015: ctx-rh-middletemporal
# 2035: ctx-rh-insula
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz \
             --match 2034 2030 2001 2015 2035 \
             --o $output_dir/details/auditory_vocal/rh_auditory_vocal.mgz

# Combine hemispheres
mri_concat $output_dir/details/auditory_vocal/lh_auditory_vocal.mgz \
           $output_dir/details/auditory_vocal/rh_auditory_vocal.mgz \
           --max --o $output_dir/auditory_vocal_integration.mgz

# Extract individual auditory components
echo "  Extracting individual auditory components..."
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1034 \
             --o $output_dir/details/auditory_vocal/lh_transversetemporal_A1.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1030 \
             --o $output_dir/details/auditory_vocal/lh_superiortemporal_STG.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1001 \
             --o $output_dir/details/auditory_vocal/lh_bankssts_STS.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1015 \
             --o $output_dir/details/auditory_vocal/lh_middletemporal_MTG.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1035 \
             --o $output_dir/details/auditory_vocal/lh_insula.mgz

mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 2034 \
             --o $output_dir/details/auditory_vocal/rh_transversetemporal_A1.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 2030 \
             --o $output_dir/details/auditory_vocal/rh_superiortemporal_STG.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 2001 \
             --o $output_dir/details/auditory_vocal/rh_bankssts_STS.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 2015 \
             --o $output_dir/details/auditory_vocal/rh_middletemporal_MTG.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 2035 \
             --o $output_dir/details/auditory_vocal/rh_insula.mgz

echo "[OK] Auditory-vocal integration mask created (10 components)"
echo ""

# ==========================================
# 5. BASAL GANGLIA VOCAL CONTROL
# ==========================================
# Motor sequencing and emotional prosody
# Important for vocalization timing and affective expression
# ==========================================

echo "Creating basal ganglia vocal control mask..."

# Basal ganglia structures
# 11: Left-Caudate (motor sequencing, vocal learning)
# 50: Right-Caudate
# 12: Left-Putamen (motor execution)
# 51: Right-Putamen
# 13: Left-Pallidum (motor timing)
# 52: Right-Pallidum
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz \
             --match 11 50 12 51 13 52 \
             --o $output_dir/basal_ganglia_vocal.mgz

# Extract individual basal ganglia components
echo "  Extracting individual basal ganglia components..."
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz --match 11 \
             --o $output_dir/details/basal_ganglia/left_caudate.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz --match 50 \
             --o $output_dir/details/basal_ganglia/right_caudate.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz --match 12 \
             --o $output_dir/details/basal_ganglia/left_putamen.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz --match 51 \
             --o $output_dir/details/basal_ganglia/right_putamen.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz --match 13 \
             --o $output_dir/details/basal_ganglia/left_pallidum.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz --match 52 \
             --o $output_dir/details/basal_ganglia/right_pallidum.mgz

echo "[OK] Basal ganglia vocal control mask created (6 components)"
echo ""

# ==========================================
# 6. COMPLETE VOCALIZATION NETWORK
# ==========================================
# Integrate all components into comprehensive network
# ==========================================

echo "Creating complete vocalization network mask..."

mri_concat $output_dir/vocal_motor_cortical.mgz \
           $output_dir/cerebellum_vocal.mgz \
           $output_dir/limbic_vocalization.mgz \
           $output_dir/auditory_vocal_integration.mgz \
           $output_dir/basal_ganglia_vocal.mgz \
           --max --o $output_dir/complete_vocalization_network.mgz

echo "[OK] Complete vocalization network mask created"
echo ""

# ==========================================
# 7. CONVERT TO NIFTI AND NATIVE SPACE
# ==========================================

echo "Converting masks to NIfTI and native space..."

# Convert main masks in root directory only
for mask in $output_dir/*.mgz; do
    if [ -f "$mask" ]; then
        mask_basename=$(basename $mask .mgz)

        # Convert to NIfTI
        mri_convert $mask $output_dir/${mask_basename}.nii.gz

        # Transform to native space
        mri_vol2vol --mov $mask \
                    --targ $t1w_input \
                    --regheader \
                    --o $output_dir/${mask_basename}_native.nii.gz \
                    --no-save-reg \
                    --interp nearest

        echo "  [OK] Processed: ${mask_basename}"
    fi
done

# Convert detail masks
for category in vocal_motor cerebellum limbic_vocalization auditory_vocal basal_ganglia; do
    echo "  Converting $category details..."
    for mask in $output_dir/details/$category/*.mgz; do
        if [ -f "$mask" ]; then
            mask_basename=$(basename $mask .mgz)

            # Convert to NIfTI
            mri_convert $mask $output_dir/details/$category/${mask_basename}.nii.gz

            # Transform to native space
            mri_vol2vol --mov $mask \
                        --targ $t1w_input \
                        --regheader \
                        --o $output_dir/details/$category/${mask_basename}_native.nii.gz \
                        --no-save-reg \
                        --interp nearest
        fi
    done
done

echo ""

# ==========================================
# 8. CREATE LABEL REFERENCE
# ==========================================

cat > $output_dir/vocalization_network_labels.txt << 'EOFVOC'
====================================================================================
AFFECTIVE VOCALIZATION NETWORK - LABEL REFERENCE
====================================================================================
Template: Desikan-Killiany Atlas (FreeSurfer)
Research Focus: Infant emotional vocalization production and perception

KEY REFERENCES:
- Jürgens (2002). Neural pathways underlying vocal control. Neurosci Biobehav Rev.
- Jürgens (2009). The neural control of vocalization in mammals. J Voice.
- Petkov & Jarvis (2012). Birds, primates, and spoken language origins. Front Evol Neurosci.
- Ackermann et al. (2014). Brain mechanisms of acoustic communication. Brain Lang.
====================================================================================

FILE NAMING CONVENTION:
- *.nii.gz: FreeSurfer space (256x256x256, 1mm isotropic)
- *_native.nii.gz: Native T1 space (original scan dimensions)
- All masks are BINARY (0=background, 1=region of interest)

====================================================================================

DIRECTORY STRUCTURE:
--------------------
ROOT DIRECTORY: Major network components (5 systems + complete network)
  - vocal_motor_cortical.nii.gz          Cortical vocal motor system
  - cerebellum_vocal.nii.gz              Cerebellar vocal control
  - limbic_vocalization.nii.gz           Limbic vocalization system
  - auditory_vocal_integration.nii.gz    Auditory-vocal integration
  - basal_ganglia_vocal.nii.gz           Basal ganglia vocal control
  - complete_vocalization_network.nii.gz All systems combined

DETAILS/: Individual anatomical regions for each system
  - vocal_motor/          12 cortical regions (left/right hemispheres)
  - cerebellum/           2 cerebellar regions (left/right)
  - limbic_vocalization/  15 regions (subcortical + cortical)
  - auditory_vocal/       10 regions (left/right temporal regions)
  - basal_ganglia/        6 regions (caudate, putamen, pallidum bilateral)

====================================================================================

REFERENCE IMAGES (ROOT):
------------------------
sub-01_ses-01_T1w.nii.gz               Raw T1-weighted image (original scan)
sub-01_ses-01_T1w.skullstripped.nii.gz Skull-stripped, normalized T1
sub-01_ses-01_T1w.brain.nii.gz         Brain-only image
brain_mask.nii.gz                      Whole brain binary mask
cerebrum_mask.nii.gz                   Cerebrum only (no cerebellum)

====================================================================================

1. VOCAL MOTOR CORTICAL (details/vocal_motor/)
------------------
Function: Motor execution and planning of vocalization
Includes: Broca's area, primary motor cortex, supplementary motor area

Left Hemisphere Components:
lh_parsopercularis_brocas.nii.gz        1018  Broca's area (motor speech) Jürgens (2002)
lh_parstriangularis_brocas.nii.gz       1020  Broca's area (planning) Jürgens (2002)
lh_precentral_M1.nii.gz                 1024  Primary motor (laryngeal) Jürgens (2002, 2009)
lh_superiorfrontal_SMA.nii.gz           1028  Supplementary motor area Jürgens (2002)
lh_caudalmiddlefrontal_premotor.nii.gz  1003  Premotor cortex Jürgens (2002)
lh_supramarginal_IPL.nii.gz             1031  Inferior parietal Jürgens (2002)
Right Hemisphere Components:
rh_parsopercularis.nii.gz               2018  Right IFG (aprosodia) Ackermann et al. (2014)
rh_parstriangularis.nii.gz              2020  Right IFG Ackermann et al. (2014)
rh_precentral_M1.nii.gz                 2024  Right primary motor
rh_superiorfrontal_SMA.nii.gz           2028  Right SMA
rh_caudalmiddlefrontal_premotor.nii.gz  2003  Right premotor
rh_supramarginal_IPL.nii.gz             2031  Right inferior parietal
Aggregated Masks:
lh_vocal_motor_cortical.nii.gz          Left hemisphere motor regions combined
rh_vocal_motor_cortical.nii.gz          Right hemisphere motor regions combined
====================================================================================

CEREBELLUM VOCAL CONTROL (details/cerebellum/)


Function: Articulatory timing, speech smoothness, predictive motor control
Critical for: Syllable duration, rhythm, correcting vocal errors
Components:
left_cerebellum_cortex.nii.gz    8   Left cerebellar timing Jürgens (2009)
right_cerebellum_cortex.nii.gz   47  Right cerebellar timing Jürgens (2009)
====================================================================================

LIMBIC VOCALIZATION SYSTEM (details/limbic_vocalization/)


Function: Emotional drive and affective content of vocalizations
Critical for: Spontaneous emotional calls (crying, laughing, distress)
Subcortical Components:
left_amygdala.nii.gz                    18   Emotional valence Jürgens (2002, 2009)
right_amygdala.nii.gz                   54   Emotional processing Jürgens (2002, 2009)
left_accumbens.nii.gz                   26   Reward/motivation Ackermann et al. (2014)
right_accumbens.nii.gz                  58   Reward processing Ackermann et al. (2014)
left_thalamus.nii.gz                    10   Sensorimotor relay Jürgens (2002, 2009)
right_thalamus.nii.gz                   49   Thalamic pathways Jürgens (2002, 2009)
brainstem.nii.gz                        16   PAG, nucleus ambiguus Jürgens (2002, 2009)
left_ventraldc_hypothalamus.nii.gz      28   Hypothalamus/autonomic Ackermann et al. (2014)
right_ventraldc_hypothalamus.nii.gz     60   Hypothalamus/gating Ackermann et al. (2014)
Cortical Components:
lh_caudalanteriorcingulate_ACC.nii.gz   1002  ACC - vocalization drive Jürgens (2002, 2009)
lh_rostralanteriorcingulate_ACC.nii.gz  1026  ACC - initiation Jürgens (2002, 2009)
rh_caudalanteriorcingulate_ACC.nii.gz   2002  Right ACC Jürgens (2002, 2009)
rh_rostralanteriorcingulate_ACC.nii.gz  2026  Right ACC Jürgens (2002, 2009)
lh_medialorbitofrontal_OFC.nii.gz       1014  Medial OFC Petkov & Jarvis (2012)
rh_medialorbitofrontal_OFC.nii.gz       2014  Medial OFC Ackermann et al. (2014)
Aggregated Masks:
subcortical_limbic_vocal.nii.gz         Subcortical limbic regions combined
cortical_limbic_vocal.nii.gz            Cortical limbic regions combined
====================================================================================

AUDITORY-VOCAL INTEGRATION (details/auditory_vocal/)


Function: Auditory feedback processing and voice perception
Critical for: Vocal learning, self-monitoring, social voice recognition
Left Hemisphere Components:
lh_transversetemporal_A1.nii.gz   1034  Primary auditory Petkov & Jarvis (2012)
lh_superiortemporal_STG.nii.gz    1030  Voice processing Petkov & Jarvis (2012)
lh_bankssts_STS.nii.gz            1001  Social voice Jürgens (2002)
lh_middletemporal_MTG.nii.gz      1015  Auditory association Jürgens (2002)
lh_insula.nii.gz                  1035  Auditory-motor integration Jürgens (2002)
Right Hemisphere Components:
rh_transversetemporal_A1.nii.gz   2034  Right primary auditory
rh_superiortemporal_STG.nii.gz    2030  Right voice processing
rh_bankssts_STS.nii.gz            2001  Right STS (prosody)
rh_middletemporal_MTG.nii.gz      2015  Right auditory association
rh_insula.nii.gz                  2035  Right auditory-motor
Aggregated Masks:
lh_auditory_vocal.nii.gz          Left hemisphere auditory regions combined
rh_auditory_vocal.nii.gz          Right hemisphere auditory regions combined
====================================================================================

BASAL GANGLIA VOCAL CONTROL (details/basal_ganglia/)


Function: Motor sequencing, timing, and emotional prosody
Important for: Vocalization rhythm, learned vocalizations
Components:
left_caudate.nii.gz    11   Motor sequencing Ackermann et al. (2014)
right_caudate.nii.gz   50   Vocalization control Ackermann et al. (2014)
left_putamen.nii.gz    12   Motor execution Jürgens (2002, 2009)
right_putamen.nii.gz   51   Motor timing Jürgens (2002, 2009)
left_pallidum.nii.gz   13   Motor gating Jürgens (2009)
right_pallidum.nii.gz  52   Vocalization timing Jürgens (2009)
====================================================================================
NETWORK SUMMARY:
Total Components: 45 individual masks
Vocal Motor Cortical: 12 components (bilateral frontal/parietal)
Cerebellum Vocal: 2 components (bilateral cerebellum)
Limbic Vocalization: 15 components (9 subcortical + 6 cortical)
Auditory-Vocal Integration: 10 components (bilateral temporal)
Basal Ganglia Vocal: 6 components (bilateral striatum/pallidum)
====================================================================================
MAJOR UPDATES (Version 4.0):
✅ REORGANIZED: Cerebellum separated from vocal motor as independent system
✅ SIMPLIFIED: Root directory contains only 5 major systems + complete network
✅ CLARIFIED: All individual components moved to details/
✅ IMPROVED: Clear hierarchical structure (system → hemisphere → region)
✅ MAINTAINED: All 45 individual anatomical components preserved
EOFVOC
echo "[OK] Label reference created: vocalization_network_labels.txt"
echo ""
==========================================
9. CLEANUP INTERMEDIATE FILES
==========================================
echo "Cleaning up intermediate .mgz files..."
find $output_dir -name "*.mgz" -type f -delete
echo "[OK] Cleanup complete"
echo ""
echo "==========================================="
echo "EXTRACTION COMPLETE"
echo "==========================================="
echo "Output directory: $output_dir"
echo ""
echo "ROOT DIRECTORY (Major Systems):"
echo "  - vocal_motor_cortical.nii.gz"
echo "  - cerebellum_vocal.nii.gz"
echo "  - limbic_vocalization.nii.gz"
echo "  - auditory_vocal_integration.nii.gz"
echo "  - basal_ganglia_vocal.nii.gz"
echo "  - complete_vocalization_network.nii.gz"
echo ""
echo "DETAILS: details/{vocal_motor,cerebellum,limbic_vocalization,auditory_vocal,basal_ganglia}/"
echo "  - 45 individual anatomical regions"
echo ""
echo "==========================================="

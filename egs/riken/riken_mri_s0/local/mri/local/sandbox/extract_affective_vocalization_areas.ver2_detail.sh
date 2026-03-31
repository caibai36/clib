#!/bin/bash

# ==========================================
# AFFECTIVE VOCALIZATION NETWORK EXTRACTION
# ==========================================
# Extracts brain regions relevant to emotional vocalization production and perception
# Reference: Jurgens (2009) Brain Res Rev; Petkov & Jarvis (2012) Nat Rev Neurosci
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
# 1. VOCAL MOTOR SYSTEM
# ==========================================
# Primary motor, premotor, and supplementary motor areas
# Controls articulatory movements and vocalization execution
# ==========================================

echo "Creating vocal motor system mask..."

# Left hemisphere motor regions
# 1018: ctx-lh-parsopercularis (Broca's area - motor speech)
# 1020: ctx-lh-parstriangularis (Broca's area - speech planning)
# 1024: ctx-lh-precentral (primary motor cortex - laryngeal/articulatory)
# 1017: ctx-lh-paracentral (supplementary motor area)
# 1003: ctx-lh-caudalmiddlefrontal (premotor cortex)
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz \
             --match 1018 1020 1024 1017 1003 \
             --o $output_dir/lh_vocal_motor.mgz

# Right hemisphere motor regions
# 2018: ctx-rh-parsopercularis
# 2020: ctx-rh-parstriangularis
# 2024: ctx-rh-precentral
# 2017: ctx-rh-paracentral
# 2003: ctx-rh-caudalmiddlefrontal
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz \
             --match 2018 2020 2024 2017 2003 \
             --o $output_dir/rh_vocal_motor.mgz

# Combine hemispheres
mri_concat $output_dir/lh_vocal_motor.mgz \
           $output_dir/rh_vocal_motor.mgz \
           --max --o $output_dir/bilateral_vocal_motor.mgz

# Extract individual components
echo "  Extracting individual motor components..."
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1018 \
             --o $output_dir/details/vocal_motor/lh_parsopercularis_brocas.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1020 \
             --o $output_dir/details/vocal_motor/lh_parstriangularis_brocas.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1024 \
             --o $output_dir/details/vocal_motor/lh_precentral_M1.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1017 \
             --o $output_dir/details/vocal_motor/lh_paracentral_SMA.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1003 \
             --o $output_dir/details/vocal_motor/lh_caudalmiddlefrontal_premotor.mgz

mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 2018 \
             --o $output_dir/details/vocal_motor/rh_parsopercularis.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 2020 \
             --o $output_dir/details/vocal_motor/rh_parstriangularis.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 2024 \
             --o $output_dir/details/vocal_motor/rh_precentral_M1.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 2017 \
             --o $output_dir/details/vocal_motor/rh_paracentral_SMA.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 2003 \
             --o $output_dir/details/vocal_motor/rh_caudalmiddlefrontal_premotor.mgz

echo "[OK] Vocal motor system mask created"
echo ""

# ==========================================
# 2. LIMBIC VOCALIZATION SYSTEM
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
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aseg.mgz \
             --match 18 54 26 58 10 49 \
             --o $output_dir/subcortical_limbic_vocal.mgz

# Cortical limbic structures - focus on anterior cingulate (vocalization drive)
# 1002: ctx-lh-caudalanteriorcingulate (emotional vocalization initiation)
# 1026: ctx-lh-rostralanteriorcingulate (vocalization drive, call production)
# 2002: ctx-rh-caudalanteriorcingulate
# 2026: ctx-rh-rostralanteriorcingulate
# 1010: ctx-lh-isthmuscingulate
# 2010: ctx-rh-isthmuscingulate
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz \
             --match 1002 1026 2002 2026 1010 2010 \
             --o $output_dir/cortical_limbic_vocal.mgz

# Combine limbic components
mri_concat $output_dir/subcortical_limbic_vocal.mgz \
           $output_dir/cortical_limbic_vocal.mgz \
           --max --o $output_dir/complete_limbic_vocal.mgz

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

# Extract individual cortical components
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1002 \
             --o $output_dir/details/limbic_vocalization/lh_caudalanteriorcingulate_ACC.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1026 \
             --o $output_dir/details/limbic_vocalization/lh_rostralanteriorcingulate_ACC.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 2002 \
             --o $output_dir/details/limbic_vocalization/rh_caudalanteriorcingulate_ACC.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 2026 \
             --o $output_dir/details/limbic_vocalization/rh_rostralanteriorcingulate_ACC.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 1010 \
             --o $output_dir/details/limbic_vocalization/lh_isthmuscingulate.mgz
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz --match 2010 \
             --o $output_dir/details/limbic_vocalization/rh_isthmuscingulate.mgz

echo "[OK] Limbic vocalization system mask created"
echo ""

# ==========================================
# 3. AUDITORY-VOCAL INTEGRATION
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
             --o $output_dir/lh_auditory_vocal.mgz

# Right hemisphere auditory regions
# 2034: ctx-rh-transversetemporal
# 2030: ctx-rh-superiortemporal
# 2001: ctx-rh-bankssts
# 2015: ctx-rh-middletemporal
# 2035: ctx-rh-insula
mri_binarize --i $SUBJECTS_DIR/$subjid/mri/aparc+aseg.mgz \
             --match 2034 2030 2001 2015 2035 \
             --o $output_dir/rh_auditory_vocal.mgz

# Combine hemispheres
mri_concat $output_dir/lh_auditory_vocal.mgz \
           $output_dir/rh_auditory_vocal.mgz \
           --max --o $output_dir/bilateral_auditory_vocal.mgz

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

echo "[OK] Auditory-vocal integration mask created"
echo ""

# ==========================================
# 4. BASAL GANGLIA VOCAL CONTROL
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
             --o $output_dir/bilateral_basal_ganglia.mgz

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

echo "[OK] Basal ganglia vocal control mask created"
echo ""

# ==========================================
# 5. COMPLETE VOCALIZATION NETWORK
# ==========================================
# Integrate all components into comprehensive network
# ==========================================

echo "Creating complete vocalization network mask..."

mri_concat $output_dir/bilateral_vocal_motor.mgz \
           $output_dir/complete_limbic_vocal.mgz \
           $output_dir/bilateral_auditory_vocal.mgz \
           $output_dir/bilateral_basal_ganglia.mgz \
           --max --o $output_dir/complete_vocalization_network.mgz

echo "[OK] Complete vocalization network mask created"
echo ""

# ==========================================
# 6. CONVERT TO NIFTI AND NATIVE SPACE
# ==========================================

echo "Converting masks to NIfTI and native space..."

# Convert main masks
for mask in $output_dir/bilateral_*.mgz $output_dir/complete_*.mgz $output_dir/*cortical*.mgz $output_dir/*subcortical*.mgz $output_dir/lh_*.mgz $output_dir/rh_*.mgz; do
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
for category in vocal_motor limbic_vocalization auditory_vocal basal_ganglia; do
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
# 7. CREATE LABEL REFERENCE
# ==========================================

cat > $output_dir/vocalization_network_labels.txt << 'EOFVOC'
====================================================================================
AFFECTIVE VOCALIZATION NETWORK - LABEL REFERENCE
====================================================================================
Template: Desikan-Killiany Atlas (FreeSurfer)
Research Focus: Infant emotional vocalization production and perception

KEY REFERENCES:
- Jurgens (2009). The neural control of vocalization in mammals. Brain Res Rev.
- Petkov & Jarvis (2012). Birds, primates, and spoken language origins. Nat Rev Neurosci.
- Ackermann et al. (2014). Brain mechanisms of acoustic communication. Brain Lang.
====================================================================================

FILE NAMING CONVENTION:
- *.nii.gz: FreeSurfer space (256x256x256, 1mm isotropic)
- *_native.nii.gz: Native T1 space (original scan dimensions)
- All masks are BINARY (0=background, 1=region of interest)

====================================================================================

REFERENCE IMAGES:
------------------
T1.nii.gz                      Raw T1-weighted image (original scan)
T1.skullstripped.nii.gz        Skull-stripped, normalized T1 (FreeSurfer brainmask)
T1.brain.nii.gz                Brain-only image (FreeSurfer brain.mgz)
brain_mask.nii.gz              Whole brain binary mask
cerebrum_mask.nii.gz           Cerebrum only (cortex + white matter, no cerebellum)

====================================================================================

VOCAL MOTOR SYSTEM (details/vocal_motor/)
------------------
Function: Motor execution and planning of vocalization
Includes: Broca's area, primary motor cortex, supplementary motor area

Left Hemisphere Components:
lh_parsopercularis_brocas.nii.gz        1018  Broca's area (motor speech programming)
lh_parstriangularis_brocas.nii.gz      1020  Broca's area (speech motor planning)
lh_precentral_M1.nii.gz                 1024  Primary motor cortex (laryngeal control)
lh_paracentral_SMA.nii.gz               1017  Supplementary motor area (initiation)
lh_caudalmiddlefrontal_premotor.nii.gz  1003  Premotor cortex

Right Hemisphere Components:
rh_parsopercularis.nii.gz               2018  Right inferior frontal gyrus
rh_parstriangularis.nii.gz              2020  Right inferior frontal gyrus
rh_precentral_M1.nii.gz                 2024  Right primary motor cortex
rh_paracentral_SMA.nii.gz               2017  Right supplementary motor area
rh_caudalmiddlefrontal_premotor.nii.gz  2003  Right premotor cortex

====================================================================================

LIMBIC VOCALIZATION SYSTEM (details/limbic_vocalization/)
---------------------------
Function: Emotional drive and affective content of vocalizations
Critical for: Spontaneous emotional calls (crying, laughing, distress)

Subcortical Components:
left_amygdala.nii.gz           18   Emotional valence, fear vocalizations
right_amygdala.nii.gz          54   Emotional processing
left_accumbens.nii.gz          26   Reward, motivation for vocal communication
right_accumbens.nii.gz         58   Reward processing
left_thalamus.nii.gz           10   Sensorimotor relay, emotional modulation
right_thalamus.nii.gz          49   Thalamic vocalization pathways

Cortical Components (Anterior Cingulate Cortex):
lh_caudalanteriorcingulate_ACC.nii.gz    1002  ACC - vocalization drive
lh_rostralanteriorcingulate_ACC.nii.gz   1026  ACC - emotional vocalization initiation
rh_caudalanteriorcingulate_ACC.nii.gz    2002  Right ACC
rh_rostralanteriorcingulate_ACC.nii.gz   2026  Right ACC
lh_isthmuscingulate.nii.gz               1010  Cingulate connections
rh_isthmuscingulate.nii.gz               2010  Cingulate connections

====================================================================================

AUDITORY-VOCAL INTEGRATION (details/auditory_vocal/)
---------------------------
Function: Auditory feedback processing and voice perception
Critical for: Vocal learning, self-monitoring, social voice recognition

Left Hemisphere Components:
lh_transversetemporal_A1.nii.gz   1034  Primary auditory cortex (Heschl's gyrus)
lh_superiortemporal_STG.nii.gz    1030  Secondary auditory, voice processing
lh_bankssts_STS.nii.gz            1001  Superior temporal sulcus (social voice)
lh_middletemporal_MTG.nii.gz      1015  Auditory association cortex
lh_insula.nii.gz                  1035  Auditory-motor integration

Right Hemisphere Components:
rh_transversetemporal_A1.nii.gz   2034  Right primary auditory cortex
rh_superiortemporal_STG.nii.gz    2030  Right voice processing areas
rh_bankssts_STS.nii.gz            2001  Right STS (prosody, emotional voice)
rh_middletemporal_MTG.nii.gz      2015  Right auditory association
rh_insula.nii.gz                  2035  Right auditory-motor integration

====================================================================================

BASAL GANGLIA VOCAL CONTROL (details/basal_ganglia/)
----------------------------
Function: Motor sequencing, timing, and emotional prosody
Important for: Vocalization rhythm, learned vocalizations

Components:
left_caudate.nii.gz    11   Motor sequencing, vocal learning
right_caudate.nii.gz   50   Basal ganglia vocalization control
left_putamen.nii.gz    12   Motor execution, articulation
right_putamen.nii.gz   51   Motor timing
left_pallidum.nii.gz   13   Motor gating, vocal control
right_pallidum.nii.gz  52   Vocalization timing

====================================================================================

DEVELOPMENTAL NOTES FOR INFANT RESEARCH:

1. INFANT VOCALIZATIONS (0-12 months):
   - Reflexive crying (birth): Brainstem + limbic system
   - Cooing (2-3 months): Emerging cortical-limbic integration
   - Canonical babbling (6-10 months): Motor cortex + auditory feedback
   - First words (10-14 months): Full network integration

2. KEY STRUCTURES FOR INFANT AFFECTIVE VOCALIZATIONS:
   - Anterior cingulate cortex: Cry initiation and emotional expression
   - Amygdala: Emotional valence (distress, pleasure)
   - Auditory cortex: Caregiver voice recognition
   - Motor cortex: Articulatory development

3. LATERALIZATION:
   - Left hemisphere: Speech-like vocalizations, linguistic processing
   - Right hemisphere: Emotional prosody, musical aspects of speech

====================================================================================
EOFVOC

echo "[OK] Label reference created: vocalization_network_labels.txt"
echo ""

# ==========================================
# 8. CREATE VISUALIZATION SCRIPT
# ==========================================

# Get T1 basename for the viewer script
t1_basename=$(basename $t1w_input)

cat > $output_dir/view_vocalization_network.sh << EOFVIEW
#!/bin/bash

MASK_DIR="\$(cd "\$(dirname "\$0")" && pwd)"
# T1 is in the same directory
T1_INPUT="./$t1_basename"

echo "==========================================="
echo "AFFECTIVE VOCALIZATION NETWORK VIEWER"
echo "==========================================="
echo ""
echo "Color scheme (Look-up Table):"
echo "  YELLOW (#ffff00) - Vocal Motor System"
echo "  ORANGE (#ffa500) - Limbic Vocalization"
echo "  VIOLET (#aa00ff) - Auditory-Vocal Integration"
echo "  RED    (#ff0000) - Basal Ganglia Vocal Control"
echo ""
echo "Navigation tips:"
echo "  - Mouse wheel: Zoom"
echo "  - Click and drag: Pan"
echo "  - Right-click: Region info"
echo ""

# Load T1 FIRST, then masks with binary colormap and exact RGB colors
# Binary colormap with custom colors provides clean, distinct visualization
freeview -v \$T1_INPUT \\
         -v \$MASK_DIR/bilateral_vocal_motor_native.nii.gz:colormap=binary:binary_color=255,255,0:opacity=0.42 \\
         -v \$MASK_DIR/complete_limbic_vocal_native.nii.gz:colormap=binary:binary_color=255,165,0:opacity=0.42 \\
         -v \$MASK_DIR/bilateral_auditory_vocal_native.nii.gz:colormap=binary:binary_color=170,0,255:opacity=0.42 \\
         -v \$MASK_DIR/bilateral_basal_ganglia_native.nii.gz:colormap=binary:binary_color=255,0,0:opacity=0.42
EOFVIEW

chmod +x $output_dir/view_vocalization_network.sh

echo "[OK] Visualization script created with LUT-matching colors"
echo ""

cat > $output_dir/README.txt << 'EOFREADME'
====================================================================================
AFFECTIVE VOCALIZATION NETWORK - OUTPUT DIRECTORY
====================================================================================

DIRECTORY STRUCTURE:
--------------------

./
├── README.txt                                    This file
├── vocalization_network_labels.txt               Complete label reference
├── view_vocalization_network.sh                  Main viewer (raw T1 background)
├── view_on_brain.sh                              Viewer (brain-only background)
├── view_on_skullstripped.sh                      Viewer (skull-stripped background)
│
├── REFERENCE IMAGES (Intensity images for viewing):
│   ├── sub-01_ses-01_T1w.nii.gz                 Raw T1 image (with skull)
│   ├── sub-01_ses-01_T1w.skullstripped.nii.gz   Skull-stripped T1 (INTENSITY image)
│   ├── sub-01_ses-01_T1w.brain.nii.gz           Brain-only T1 (INTENSITY image)
│   ├── brain_mask.nii.gz                         Whole brain BINARY mask (0/1)
│   └── cerebrum_mask.nii.gz                      Cerebrum BINARY mask (0/1)
│
│   NOTE: .skullstripped and .brain are INTENSITY images (for visualization)
│         .mask files are BINARY (0/1 values for analysis)
│
├── NETWORK MASKS (FreeSurfer space):
│   ├── bilateral_vocal_motor.nii.gz
│   ├── complete_limbic_vocal.nii.gz
│   ├── bilateral_auditory_vocal.nii.gz
│   ├── bilateral_basal_ganglia.nii.gz
│   └── complete_vocalization_network.nii.gz
│
├── NETWORK MASKS (Native T1 space):
│   ├── bilateral_vocal_motor_native.nii.gz
│   ├── complete_limbic_vocal_native.nii.gz
│   ├── bilateral_auditory_vocal_native.nii.gz
│   ├── bilateral_basal_ganglia_native.nii.gz
│   └── complete_vocalization_network_native.nii.gz
│
└── details/                                       Individual component masks
    ├── vocal_motor/                              10 components (Broca's, M1, SMA, etc.)
    ├── limbic_vocalization/                      12 components (ACC, amygdala, thalamus, etc.)
    ├── auditory_vocal/                           10 components (A1, STG, STS, insula, etc.)
    └── basal_ganglia/                            6 components (caudate, putamen, pallidum)

====================================================================================

COLOR SCHEME (Look-up Table):
-----------------------------
í ½í¿¡ YELLOW (#ffff00) - Vocal Motor System
í ½í¿  ORANGE (#ffa500) - Limbic Vocalization
í ½í¿£ VIOLET (#aa00ff) - Auditory-Vocal Integration
í ½í´´ RED    (#ff0000) - Basal Ganglia Vocal Control

====================================================================================

QUICK START:
------------

1. VIEW ALL NETWORKS:
   ./view_vocalization_network.sh              # On raw T1
   ./view_on_brain.sh                          # On brain-only (no skull)
   ./view_on_skullstripped.sh                  # On skull-stripped

2. LOAD IN PYTHON:
   import nibabel as nib
   import numpy as np
   
   # Load limbic system mask
   img = nib.load('complete_limbic_vocal_native.nii.gz')
   mask = img.get_fdata()
   
   # Check it's binary (0 and 1)
   print(np.unique(mask))  # [0. 1.]
   
   # Count voxels
   print(f"Limbic voxels: {np.sum(mask == 1)}")
   
   # Load brain image for visualization
   brain_img = nib.load('sub-01_ses-01_T1w.brain.nii.gz')
   brain_data = brain_img.get_fdata()  # Intensity values

3. USE WITH FSL:
   # Apply mask to functional data
   fslmaths fmri_data.nii.gz -mas complete_limbic_vocal_native.nii.gz limbic_fmri.nii.gz
   
   # Get statistics within mask
   fslstats fmri_data.nii.gz -k complete_limbic_vocal_native.nii.gz -M

4. ANALYZE INDIVIDUAL COMPONENTS:
   cd details/limbic_vocalization/
   # Each .nii.gz file is a binary mask for one structure
   freeview ../sub-01_ses-01_T1w.brain.nii.gz left_amygdala_native.nii.gz

====================================================================================

FILE TYPES:
-----------

INTENSITY IMAGES (for viewing):
- sub-01_ses-01_T1w.nii.gz              Raw T1 (values: 0-4095)
- sub-01_ses-01_T1w.skullstripped.nii.gz Processed T1 (values: 0-255 typically)
- sub-01_ses-01_T1w.brain.nii.gz        Brain-only T1 (values: 0-255 typically)

BINARY MASKS (for analysis):
- *_mask.nii.gz                          Binary masks (values: 0 or 1 ONLY)
- bilateral_*.nii.gz                     Network masks (values: 0 or 1 ONLY)
- complete_*.nii.gz                      Combined masks (values: 0 or 1 ONLY)
- details/*/*.nii.gz                     Component masks (values: 0 or 1 ONLY)

====================================================================================
EOFREADME

# ==========================================
# 8B. CREATE INDIVIDUAL COMPONENT VIEWERS
# ==========================================

# View with brain-only background
cat > $output_dir/view_on_brain.sh << EOFBRAIN
#!/bin/bash

MASK_DIR="\$(cd "\$(dirname "\$0")" && pwd)"
T1_BRAIN="./${t1_basename%.nii.gz}.brain.nii.gz"

echo "Viewing on brain-only image (no skull)..."

freeview -v \$T1_BRAIN \\
         -v \$MASK_DIR/bilateral_vocal_motor_native.nii.gz:colormap=binary:binary_color=255,255,0:opacity=0.42 \\
         -v \$MASK_DIR/complete_limbic_vocal_native.nii.gz:colormap=binary:binary_color=255,165,0:opacity=0.42 \\
         -v \$MASK_DIR/bilateral_auditory_vocal_native.nii.gz:colormap=binary:binary_color=170,0,255:opacity=0.42 \\
         -v \$MASK_DIR/bilateral_basal_ganglia_native.nii.gz:colormap=binary:binary_color=255,0,0:opacity=0.42
EOFBRAIN

chmod +x $output_dir/view_on_brain.sh

# View with skull-stripped background
cat > $output_dir/view_on_skullstripped.sh << EOFSKULL
#!/bin/bash

MASK_DIR="\$(cd "\$(dirname "\$0")" && pwd)"
T1_STRIPPED="./${t1_basename%.nii.gz}.skullstripped.nii.gz"

echo "Viewing on skull-stripped image..."

freeview -v \$T1_STRIPPED \\
         -v \$MASK_DIR/bilateral_vocal_motor_native.nii.gz:colormap=binary:binary_color=255,255,0:opacity=0.42 \\
         -v \$MASK_DIR/complete_limbic_vocal_native.nii.gz:colormap=binary:binary_color=255,165,0:opacity=0.42 \\
         -v \$MASK_DIR/bilateral_auditory_vocal_native.nii.gz:colormap=binary:binary_color=170,0,255:opacity=0.42 \\
         -v \$MASK_DIR/bilateral_basal_ganglia_native.nii.gz:colormap=binary:binary_color=255,0,0:opacity=0.42
EOFSKULL

chmod +x $output_dir/view_on_skullstripped.sh

echo "[OK] Additional viewer scripts created:"
echo "     - view_on_brain.sh (brain-only background)"
echo "     - view_on_skullstripped.sh (skull-stripped background)"
echo ""

# ==========================================
# 9. CREATE README FILE
# ==========================================

cat > $output_dir/README.txt << 'EOFREADME'
====================================================================================
AFFECTIVE VOCALIZATION NETWORK - OUTPUT DIRECTORY
====================================================================================

DIRECTORY STRUCTURE:
--------------------

./
├── README.txt                                    This file
├── vocalization_network_labels.txt               Complete label reference
├── view_vocalization_network.sh                  Main viewer (raw T1 background)
├── view_on_brain.sh                              Viewer (brain-only background)
├── view_on_skullstripped.sh                      Viewer (skull-stripped background)
│
├── REFERENCE IMAGES (Intensity images for viewing):
│   ├── sub-01_ses-01_T1w.nii.gz                 Raw T1 image (with skull)
│   ├── sub-01_ses-01_T1w.skullstripped.nii.gz   Skull-stripped T1 (INTENSITY image)
│   ├── sub-01_ses-01_T1w.brain.nii.gz           Brain-only T1 (INTENSITY image)
│   ├── brain_mask.nii.gz                         Whole brain BINARY mask (0/1)
│   └── cerebrum_mask.nii.gz                      Cerebrum BINARY mask (0/1)
│
│   NOTE: .skullstripped and .brain are INTENSITY images (for visualization)
│         .mask files are BINARY (0/1 values for analysis)
│
├── NETWORK MASKS (FreeSurfer space):
│   ├── bilateral_vocal_motor.nii.gz
│   ├── complete_limbic_vocal.nii.gz
│   ├── bilateral_auditory_vocal.nii.gz
│   ├── bilateral_basal_ganglia.nii.gz
│   └── complete_vocalization_network.nii.gz
│
├── NETWORK MASKS (Native T1 space):
│   ├── bilateral_vocal_motor_native.nii.gz
│   ├── complete_limbic_vocal_native.nii.gz
│   ├── bilateral_auditory_vocal_native.nii.gz
│   ├── bilateral_basal_ganglia_native.nii.gz
│   └── complete_vocalization_network_native.nii.gz
│
└── details/                                       Individual component masks
    ├── vocal_motor/                              10 components (Broca's, M1, SMA, etc.)
    ├── limbic_vocalization/                      12 components (ACC, amygdala, thalamus, etc.)
    ├── auditory_vocal/                           10 components (A1, STG, STS, insula, etc.)
    └── basal_ganglia/                            6 components (caudate, putamen, pallidum)

====================================================================================

COLOR SCHEME (Look-up Table):
-----------------------------
í ½í¿¡ YELLOW (#ffff00) - Vocal Motor System
í ½í¿  ORANGE (#ffa500) - Limbic Vocalization
í ½í¿£ VIOLET (#aa00ff) - Auditory-Vocal Integration
í ½í´´ RED    (#ff0000) - Basal Ganglia Vocal Control

====================================================================================

QUICK START:
------------

1. VIEW ALL NETWORKS:
   ./view_vocalization_network.sh              # On raw T1
   ./view_on_brain.sh                          # On brain-only (no skull)
   ./view_on_skullstripped.sh                  # On skull-stripped

2. LOAD IN PYTHON:
   import nibabel as nib
   import numpy as np
   
   # Load limbic system mask
   img = nib.load('complete_limbic_vocal_native.nii.gz')
   mask = img.get_fdata()
   
   # Check it's binary (0 and 1)
   print(np.unique(mask))  # [0. 1.]
   
   # Count voxels
   print(f"Limbic voxels: {np.sum(mask == 1)}")
   
   # Load brain image for visualization
   brain_img = nib.load('sub-01_ses-01_T1w.brain.nii.gz')
   brain_data = brain_img.get_fdata()  # Intensity values

3. USE WITH FSL:
   # Apply mask to functional data
   fslmaths fmri_data.nii.gz -mas complete_limbic_vocal_native.nii.gz limbic_fmri.nii.gz
   
   # Get statistics within mask
   fslstats fmri_data.nii.gz -k complete_limbic_vocal_native.nii.gz -M

4. ANALYZE INDIVIDUAL COMPONENTS:
   cd details/limbic_vocalization/
   # Each .nii.gz file is a binary mask for one structure
   freeview ../sub-01_ses-01_T1w.brain.nii.gz left_amygdala_native.nii.gz

====================================================================================

FILE TYPES:
-----------

INTENSITY IMAGES (for viewing):
- sub-01_ses-01_T1w.nii.gz              Raw T1 (values: 0-4095)
- sub-01_ses-01_T1w.skullstripped.nii.gz Processed T1 (values: 0-255 typically)
- sub-01_ses-01_T1w.brain.nii.gz        Brain-only T1 (values: 0-255 typically)

BINARY MASKS (for analysis):
- *_mask.nii.gz                          Binary masks (values: 0 or 1 ONLY)
- bilateral_*.nii.gz                     Network masks (values: 0 or 1 ONLY)
- complete_*.nii.gz                      Combined masks (values: 0 or 1 ONLY)
- details/*/*.nii.gz                     Component masks (values: 0 or 1 ONLY)

====================================================================================
EOFREADME

echo "[OK] README created"
echo ""

# ==========================================
# 10. SUMMARY
# ==========================================

echo "==========================================="
echo "EXTRACTION COMPLETE!"
echo "==========================================="
echo ""
echo "Output directory: $output_dir"
echo ""
echo "Network components extracted:"
echo "  1. Vocal Motor System (10 components)"
echo "     - Broca's area, M1, SMA, premotor cortex"
echo "  2. Limbic Vocalization System (12 components)"
echo "     - ACC, amygdala, thalamus, accumbens"
echo "  3. Auditory-Vocal Integration (10 components)"
echo "     - A1, STG, STS, insula, MTG"
echo "  4. Basal Ganglia Vocal Control (6 components)"
echo "     - Caudate, putamen, pallidum"
echo ""
echo "Reference images:"
echo "  - Raw T1: $t1_basename"
echo "  - Processed T1: ${t1_basename%.nii.gz}.skullstripped.nii.gz"
echo "  - Brain mask: brain_mask.nii.gz"
echo "  - Cerebrum mask: cerebrum_mask.nii.gz"
echo ""
echo "Main network masks (native space):"
ls -lh $output_dir/*_native.nii.gz 2>/dev/null | grep -E "bilateral|complete" | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Individual components:"
echo "  Vocal motor:           $(ls $output_dir/details/vocal_motor/*.nii.gz 2>/dev/null | wc -l) masks"
echo "  Limbic vocalization:   $(ls $output_dir/details/limbic_vocalization/*.nii.gz 2>/dev/null | wc -l) masks"
echo "  Auditory-vocal:        $(ls $output_dir/details/auditory_vocal/*.nii.gz 2>/dev/null | wc -l) masks"
echo "  Basal ganglia:         $(ls $output_dir/details/basal_ganglia/*.nii.gz 2>/dev/null | wc -l) masks"
echo ""
echo "Documentation:"
echo "  - README.txt: Quick start guide and directory structure"
echo "  - vocalization_network_labels.txt: Complete label reference"
echo ""
echo "To visualize:"
echo "  cd $output_dir"
echo "  ./view_vocalization_network.sh"
echo ""
echo "All masks are BINARY (0=background, 1=ROI)"
echo "Ready for ROI analysis, connectivity studies, and volumetric measurements!"
echo ""

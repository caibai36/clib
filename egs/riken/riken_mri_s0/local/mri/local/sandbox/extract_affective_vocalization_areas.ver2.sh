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

echo "==========================================="
echo "AFFECTIVE VOCALIZATION NETWORK EXTRACTION"
echo "Subject: $subjid"
echo "Template: Desikan-Killiany Atlas"
echo "==========================================="
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

for mask in $output_dir/bilateral_*.mgz $output_dir/complete_*.mgz; do
    if [ -f "$mask" ]; then
        basename=$(basename $mask .mgz)
        
        # Convert to NIfTI
        mri_convert $mask $output_dir/${basename}.nii.gz
        
        # Transform to native space
        mri_vol2vol --mov $mask \
                    --targ $t1w_input \
                    --regheader \
                    --o $output_dir/${basename}_native.nii.gz \
                    --no-save-reg \
                    --interp nearest
        
        echo "  [OK] Processed: ${basename}"
    fi
done

echo ""

# ==========================================
# 7. COPY T1 TO OUTPUT DIRECTORY
# ==========================================

echo "Copying T1 image to output directory for easy visualization..."

# Copy T1 to vocalization_network directory preserving original filename
t1_basename=$(basename $t1w_input)
cp $t1w_input $output_dir/$t1_basename

echo "[OK] T1 image copied to: $output_dir/$t1_basename"
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
- Jurgens (2009). The neural control of vocalization in mammals. Brain Res Rev.
- Petkov & Jarvis (2012). Birds, primates, and spoken language origins. Nat Rev Neurosci.
- Ackermann et al. (2014). Brain mechanisms of acoustic communication. Brain Lang.
====================================================================================

VOCAL MOTOR SYSTEM
------------------
Function: Motor execution and planning of vocalization
Includes: Broca's area, primary motor cortex, supplementary motor area

Left Hemisphere:
1018    ctx-lh-parsopercularis         Broca's area (motor speech programming)
1020    ctx-lh-parstriangularis        Broca's area (speech motor planning)
1024    ctx-lh-precentral              Primary motor cortex (laryngeal control)
1017    ctx-lh-paracentral             Supplementary motor area (vocalization initiation)
1003    ctx-lh-caudalmiddlefrontal     Premotor cortex

Right Hemisphere:
2018    ctx-rh-parsopercularis         Right inferior frontal gyrus
2020    ctx-rh-parstriangularis        Right inferior frontal gyrus
2024    ctx-rh-precentral              Right primary motor cortex
2017    ctx-rh-paracentral             Right supplementary motor area
2003    ctx-rh-caudalmiddlefrontal     Right premotor cortex


LIMBIC VOCALIZATION SYSTEM
---------------------------
Function: Emotional drive and affective content of vocalizations
Critical for: Spontaneous emotional calls (crying, laughing, distress)

Subcortical:
18      Left-Amygdala                  Emotional valence, fear vocalizations
54      Right-Amygdala                 Emotional processing
26      Left-Accumbens                 Reward, motivation for vocal communication
58      Right-Accumbens                Reward processing
10      Left-Thalamus                  Sensorimotor relay, emotional modulation
49      Right-Thalamus                 Thalamic vocalization pathways

Cortical:
1002    ctx-lh-caudalanteriorcingulate    Anterior cingulate cortex (ACC) - vocalization drive
1026    ctx-lh-rostralanteriorcingulate   ACC - emotional vocalization initiation
2002    ctx-rh-caudalanteriorcingulate    Right ACC
2026    ctx-rh-rostralanteriorcingulate   Right ACC
1010    ctx-lh-isthmuscingulate           Cingulate connections
2010    ctx-rh-isthmuscingulate           Cingulate connections


AUDITORY-VOCAL INTEGRATION
---------------------------
Function: Auditory feedback processing and voice perception
Critical for: Vocal learning, self-monitoring, social voice recognition

Left Hemisphere:
1034    ctx-lh-transversetemporal      Primary auditory cortex (Heschl's gyrus)
1030    ctx-lh-superiortemporal        Secondary auditory, voice-selective regions
1001    ctx-lh-bankssts                Superior temporal sulcus (social voice)
1015    ctx-lh-middletemporal          Auditory association cortex
1035    ctx-lh-insula                  Auditory-motor integration

Right Hemisphere:
2034    ctx-rh-transversetemporal      Right primary auditory cortex
2030    ctx-rh-superiortemporal        Right voice processing areas
2001    ctx-rh-bankssts                Right STS (prosody, emotional voice)
2015    ctx-rh-middletemporal          Right auditory association
2035    ctx-rh-insula                  Right auditory-motor integration


BASAL GANGLIA VOCAL CONTROL
----------------------------
Function: Motor sequencing, timing, and emotional prosody
Important for: Vocalization rhythm, learned vocalizations

11      Left-Caudate                   Motor sequencing, vocal learning
50      Right-Caudate                  Basal ganglia vocalization control
12      Left-Putamen                   Motor execution, articulation
51      Right-Putamen                  Motor timing
13      Left-Pallidum                  Motor gating, vocal control
52      Right-Pallidum                 Vocalization timing

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
# 9. CREATE VISUALIZATION SCRIPT
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

# Load T1 FIRST, then masks with LOWER opacity
# Use -v for each volume separately to ensure proper loading
freeview -v \$T1_INPUT \\
	 -v \$MASK_DIR/bilateral_vocal_motor_native.nii.gz:colormap=binary:binary_color=255,255,0:opacity=0.42 \\
         -v \$MASK_DIR/complete_limbic_vocal_native.nii.gz:colormap=binary:binary_color=255,165,0:opacity=0.42 \\
         -v \$MASK_DIR/bilateral_auditory_vocal_native.nii.gz:colormap=binary:binary_color=170,0,255:opacity=0.42 \\
         -v \$MASK_DIR/bilateral_basal_ganglia_native.nii.gz:colormap=binary:binary_color=255,0,0:opacity=0.42
EOFVIEW

chmod +x $output_dir/view_vocalization_network.sh

echo "[OK] Visualization script created"
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
echo "  1. Vocal Motor System (Broca's, motor cortex, SMA)"
echo "  2. Limbic Vocalization System (ACC, amygdala, thalamus, accumbens)"
echo "  3. Auditory-Vocal Integration (auditory cortex, STS, insula)"
echo "  4. Basal Ganglia Vocal Control (caudate, putamen, pallidum)"
echo ""
echo "Files ready for analysis:"
ls -lh $output_dir/*_native.nii.gz 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "T1 reference image:"
echo "  $output_dir/$t1_basename"
echo ""
echo "To visualize:"
echo "  cd $output_dir"
echo "  ./view_vocalization_network.sh"
echo ""
echo "Reference files:"
echo "  $output_dir/vocalization_network_labels.txt"
echo ""

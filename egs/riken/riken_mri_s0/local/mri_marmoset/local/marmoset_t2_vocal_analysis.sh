#!/bin/bash

# ==========================================
# MARMOSET T2 VOCALIZATION NETWORK ANALYSIS
# ==========================================
# Extracts and analyzes affective vocalization networks from marmoset T2 anatomical data
# for cross-species developmental comparison with human data
#
# Key References:
#   - Jurgens (2009). Brain Res Rev 62:155-168. [Neural control of vocalization]
#   - Petkov & Jarvis (2012). Nat Rev Neurosci 13:730-744. [Cross-species vocal circuits]
#   - Liu et al. (2020). Neuroimage 226:117620. [MBM template v3.0.1]
#   - Takahashi et al. (2015). Science 349:734-738. [Marmoset vocal development]
#
# Pipeline: https://gitlab.com/cfmm/marmoset-connectivity (registration_MASTER.sh)
# Template: https://marmosetbrainmapping.org/atlas.html#v3 (MBM v3.0.1)
# Dataset: https://www.marmosetbrainconnectome.org/download.html
# ==========================================

# Configuration
stage=8 # start from which stage

# Paths
raw_data_dir="/data02/share/bin-wu/data/marmoset/brain/nih_uwo/nih/NIH-data"
mbm_template_dir="/data02/share/bin-wu/data/marmoset/brain/nih_uwo/marmoset_brain_mapping_v3/Marmoset_Brain_Mappping_v3.0.1/MBM_v3.0.1_0.5mm" # downsampled by 3dresample

# Input file names (allows flexibility for different datasets)
t2_name="InplaneT2.nii.gz"      # T2-weighted anatomical image
mask_name="mask.nii.gz"          # Brain mask

# Subject selection
subjects="m6"  # Space-separated list

# Output
expdir="exp/mri/sandbox/marmoset_vocalization_t2_analysis"

. ./local/scripts/parse_options.sh || exit 1

# Setup
mkdir -p $expdir/{anatomical,template_space,roi_masks,native_rois,measurements,logs}
logdir=$expdir/logs

echo "==========================================="
echo "MARMOSET VOCALIZATION NETWORK ANALYSIS"
echo "==========================================="
echo "Stage: $stage"
echo "Subjects: $subjects"
echo "Input files: T2=$t2_name, Mask=$mask_name"
echo "Output: $expdir"
echo "Atlas: RIKEN BMA (cortical) + MBM subcortical beta"
echo ""

# ==========================================
# Stage 0: Prepare anatomical data
# ==========================================

if [ $stage -le 0 ]; then
  echo "-------------------------------------------"
  echo "Stage 0: Preparing anatomical data"
  echo "-------------------------------------------"
  
  for subject in $subjects; do
    echo "Processing: $subject"
    
    subj_anat=$expdir/anatomical/$subject
    mkdir -p $subj_anat
    
    # Check if T2 file exists
    if [ -f $raw_data_dir/$subject/$t2_name ]; then
      ln -sf $raw_data_dir/$subject/$t2_name $subj_anat/
      
      # Check if mask file exists (optional for some datasets)
      if [ -f $raw_data_dir/$subject/$mask_name ]; then
        ln -sf $raw_data_dir/$subject/$mask_name $subj_anat/
        echo "  [OK] $subject: $t2_name + $mask_name linked"
      else
        echo "  [OK] $subject: $t2_name linked (no mask file)"
      fi
    else
      echo "  [WARNING] $subject: $t2_name not found in $raw_data_dir/$subject"
    fi
  done
  
  echo "[Stage 0 Complete]"
  echo ""
fi

# ==========================================
# Stage 1: Register T2 to MBM template
# ==========================================

if [ $stage -le 1 ]; then
  echo "-------------------------------------------"
  echo "Stage 1: T2 to MBM template registration"
  echo "-------------------------------------------"
  
  template_brain=$mbm_template_dir/template_T2w_brain_0.5mm.nii.gz
  
  if [ ! -f $template_brain ]; then
    echo "[ERROR] Template not found: $template_brain"
    exit 1
  fi
  
  for subject in $subjects; do
    echo "Registering: $subject"
    
    subj_anat=$expdir/anatomical/$subject
    subj_template=$expdir/template_space/$subject
    mkdir -p $subj_template
    
    # Check if T2 exists
    if [ ! -f $subj_anat/$t2_name ]; then
      echo "  [ERROR] T2 file not found: $subj_anat/$t2_name"
      continue
    fi
    
    # Step 1: Mask alignment (from registration_MASTER.sh line 40)
    # Only if mask file exists
    if [ -f $subj_anat/$mask_name ]; then
      if [ ! -f $subj_template/${subject}_mask_to_t2.nii.gz ]; then
        flirt -searchrx -360 360 -searchry -360 360 -searchrz -360 360 \
              -in $subj_anat/$mask_name \
              -ref $subj_anat/$t2_name \
              -out $subj_template/${subject}_mask_to_t2.nii.gz \
              2>&1 | tee $logdir/${subject}_mask_align.log
      fi
      
      # Step 2: Binarize mask (from registration_MASTER.sh line 43)
      if [ ! -f $subj_template/${subject}_mask_binary.nii.gz ]; then
        3dcalc -a $subj_template/${subject}_mask_to_t2.nii.gz \
               -expr 'ispositive(a)' \
               -prefix $subj_template/${subject}_mask_binary.nii.gz
      fi
      
      # Step 3: Apply mask (from registration_MASTER.sh line 46)
      if [ ! -f $subj_template/${subject}_InplaneT2_masked.nii.gz ]; then
        3dcalc -a $subj_anat/$t2_name \
               -b $subj_template/${subject}_mask_binary.nii.gz \
               -expr '(a*b)' \
               -prefix $subj_template/${subject}_InplaneT2_masked.nii.gz
      fi
    else
      # If no mask file, create brain mask using AFNI's 3dAutomask
      echo "  No mask file found, using 3dAutomask..."
      if [ ! -f $subj_template/${subject}_InplaneT2_masked.nii.gz ]; then
        3dAutomask -prefix $subj_template/${subject}_mask_binary.nii.gz $subj_anat/$t2_name
        3dcalc -a $subj_anat/$t2_name \
               -b $subj_template/${subject}_mask_binary.nii.gz \
               -expr '(a*b)' \
               -prefix $subj_template/${subject}_InplaneT2_masked.nii.gz
      fi
    fi
    
    # Step 4: ANTs registration to template (from registration_MASTER.sh line 49)
    if [ ! -f $subj_template/${subject}_t2_to_template_Warped.nii.gz ]; then
      echo "  Running ANTs registration (this may take 10-30 minutes)..."
      antsRegistrationSyNQuick.sh \
          -d 3 \
          -f $template_brain \
          -m $subj_template/${subject}_InplaneT2_masked.nii.gz \
          -o $subj_template/${subject}_t2_to_template_ \
          2>&1 | tee $logdir/${subject}_ants_registration.log
      
      echo "  [OK] $subject registered to template"
    else
      echo "  [SKIP] $subject already registered"
    fi
  done
  
  echo "[Stage 1 Complete]"
  echo ""
fi

# ==========================================
# Stage 2: Extract vocalization ROIs from template
# ==========================================
# Network Components (compare with human extract_affective_vocalization_areas.ver2.sh):
#
# 1. VOCAL MOTOR SYSTEM (cortical only)
#    RIKEN BMA: 31=A4ab, 32=A4c, 33=A6DC, 34=A6DR, 35=A6M (SMA-like),
#               36=A6Va, 37=A6Vb, 38=A8aD, 39=A8Av
#    Human: Broca's (1018,1020,2018,2020), Motor (1024,2024), SMA (1017,2017)
#    Note: Marmosets LACK Broca's area homolog
#
# 2. LIMBIC VOCALIZATION SYSTEM (cortical + subcortical)
#    Cortical - RIKEN BMA: 57=A24a, 58=A24b, 59=A24c, 60=A24d, 61=A25, 66=A32, 67=A32V
#    Subcortical - MBM: 2=Amygdala, 3=Thalamus, 10=Accumbens, 18=PAG
#    Human: ACC (1002,1026,2002,2026), Amygdala (18,54), Thalamus (10,49), Accumbens (26,58)
#    Note: PAG (periaqueductal gray) = brainstem vocal pattern generator, critical!
#
# 3. AUDITORY-VOCAL INTEGRATION (cortical only)
#    RIKEN BMA: 81=AuA1 (core), 76=AuAL, 77=AuCL, 78=AuCM, 80=AuML, 82=AuR,
#               126=STR, 138=TPO
#    Human: Heschl's (1034,2034), STG (1030,2030), STS (1001,2001), Insula (1035,2035)
#    Note: Primary auditory (A1) highly conserved
#
# 4. BASAL GANGLIA VOCAL CONTROL (subcortical)
#    MBM: 8=Caudate, 9=Putamen, 14=Globus_pallidus
#    Human: Caudate (11,50), Putamen (12,51), Pallidum (13,52)
#    Note: MBM labels are BILATERAL (single label for L+R)
# ==========================================

if [ $stage -le 2 ]; then
  echo "-------------------------------------------"
  echo "Stage 2: Extract vocalization ROIs"
  echo "-------------------------------------------"
  
  roi_output=$expdir/roi_masks
  cortical_atlas=$mbm_template_dir/atlas_RikenBMA_cortex_0.5mm.nii.gz
  subcortical_atlas=$mbm_template_dir/atlas_MBM_subcortical_beta_0.5mm.nii.gz
  
  echo "Using atlases:"
  echo "  Cortical: RIKEN BMA"
  echo "  Subcortical: MBM subcortical beta"
  
  if [ ! -f $cortical_atlas ]; then
    echo "[ERROR] Cortical atlas not found: $cortical_atlas"
    exit 1
  fi
  
  if [ ! -f $subcortical_atlas ]; then
    echo "[ERROR] Subcortical atlas not found: $subcortical_atlas"
    exit 1
  fi
  
  # 1. Vocal Motor System (cortical only)
  if [ ! -f $roi_output/vocal_motor_template.nii.gz ]; then
    echo "Creating vocal motor ROI..."
    3dcalc -a $cortical_atlas \
           -expr 'amongst(a,31,32,33,34,35,36,37,38,39)' \
           -prefix $roi_output/vocal_motor_template.nii.gz
    echo "  [OK] Vocal motor: A4ab,A4c,A6DC,A6DR,A6M,A6Va,A6Vb,A8aD,A8Av (9 regions)"
  fi
  
  # 2. Limbic Vocalization System
  # 2a. Cortical limbic (anterior cingulate)
  if [ ! -f $roi_output/limbic_vocal_cortical_template.nii.gz ]; then
    echo "Creating limbic cortical ROI..."
    3dcalc -a $cortical_atlas \
           -expr 'amongst(a,57,58,59,60,61,66,67)' \
           -prefix $roi_output/limbic_vocal_cortical_template.nii.gz
    echo "  [OK] Limbic cortical: A24a,A24b,A24c,A24d,A25,A32,A32V (7 regions)"
  fi
  
  # 2b. Subcortical limbic (amygdala, thalamus, accumbens, PAG)
  # MBM labels: 2=Amy, 3=Thal, 10=Acb, 18=PAG
  if [ ! -f $roi_output/limbic_vocal_subcortical_template.nii.gz ]; then
    echo "Creating limbic subcortical ROI..."
    3dcalc -a $subcortical_atlas \
           -expr 'amongst(a,2,3,10,18)' \
           -prefix $roi_output/limbic_vocal_subcortical_template.nii.gz
    echo "  [OK] Limbic subcortical: Amygdala,Thalamus,Accumbens,PAG (4 structures)"
  fi
  
  # 2c. Complete limbic (cortical + subcortical)
  if [ ! -f $roi_output/limbic_vocal_complete_template.nii.gz ]; then
    3dcalc -a $roi_output/limbic_vocal_cortical_template.nii.gz \
           -b $roi_output/limbic_vocal_subcortical_template.nii.gz \
           -expr 'step(a+b)' \
           -prefix $roi_output/limbic_vocal_complete_template.nii.gz
    echo "  [OK] Limbic complete: cortical (7) + subcortical (4) = 11 structures"
  fi
  
  # 3. Auditory-Vocal Integration (cortical only)
  # RIKEN labels: 81=AuA1, 76=AuAL, 77=AuCL, 78=AuCM, 80=AuML, 82=AuR, 126=STR, 138=TPO
  if [ ! -f $roi_output/auditory_vocal_template.nii.gz ]; then
    echo "Creating auditory-vocal integration ROI..."
    3dcalc -a $cortical_atlas \
           -expr 'amongst(a,81,76,77,78,80,82,126,138)' \
           -prefix $roi_output/auditory_vocal_template.nii.gz
    echo "  [OK] Auditory-vocal: AuA1,AuAL,AuCL,AuCM,AuML,AuR,STR,TPO (8 regions)"
  fi
  
  # 4. Basal Ganglia Vocal Control (subcortical)
  # MBM labels: 8=Caudate, 9=Putamen, 14=Globus_pallidus
  if [ ! -f $roi_output/basal_ganglia_template.nii.gz ]; then
    echo "Creating basal ganglia ROI..."
    3dcalc -a $subcortical_atlas \
           -expr 'amongst(a,8,9,14)' \
           -prefix $roi_output/basal_ganglia_template.nii.gz
    echo "  [OK] Basal ganglia: Caudate,Putamen,Globus_pallidus (3 structures)"
  fi
  
  # 5. Complete vocalization network
  if [ ! -f $roi_output/complete_vocalization_network_template.nii.gz ]; then
    echo "Creating complete vocalization network..."
    3dcalc -a $roi_output/vocal_motor_template.nii.gz \
           -b $roi_output/limbic_vocal_complete_template.nii.gz \
           -c $roi_output/auditory_vocal_template.nii.gz \
           -d $roi_output/basal_ganglia_template.nii.gz \
           -expr 'step(a+b+c+d)' \
           -prefix $roi_output/complete_vocalization_network_template.nii.gz
    echo "  [OK] Complete network: 9 motor + 11 limbic + 8 auditory + 3 BG = 31 structures"
  fi
  
  echo "[Stage 2 Complete]"
  echo ""
fi

# ==========================================
# Stage 3: Transform ROIs to native space
# ==========================================

if [ $stage -le 3 ]; then
  echo "-------------------------------------------"
  echo "Stage 3: Transform ROIs to native space"
  echo "-------------------------------------------"
  
  roi_template_dir=$expdir/roi_masks
  
  for subject in $subjects; do
    echo "Processing: $subject"
    
    subj_template=$expdir/template_space/$subject
    subj_native=$expdir/native_rois/$subject
    mkdir -p $subj_native
    
    if [ ! -f $subj_template/${subject}_t2_to_template_Warped.nii.gz ]; then
      echo "  [ERROR] Registration not found for $subject, run stage 1 first"
      continue
    fi
    
    # Transform each ROI (from template_to_native.sh line 13)
    for roi in $roi_template_dir/*_template.nii.gz; do
      roi_name=$(basename $roi _template.nii.gz)
      
      if [ ! -f $subj_native/${roi_name}_native.nii.gz ]; then
        antsApplyTransforms \
            -e 3 \
            -i $roi \
            -r $subj_template/${subject}_InplaneT2_masked.nii.gz \
            -o $subj_native/${roi_name}_native.nii.gz \
            -t $subj_template/${subject}_t2_to_template_0GenericAffine.mat \
            -t $subj_template/${subject}_t2_to_template_1Warp.nii.gz \
            -n NearestNeighbor \
            2>&1 | tee $logdir/${subject}_${roi_name}_transform.log
      fi
    done
    
    cp $subj_template/${subject}_InplaneT2_masked.nii.gz $subj_native/
    echo "  [OK] $subject ROIs transformed"
  done
  
  echo "[Stage 3 Complete]"
  echo ""
fi

# ==========================================
# Stage 4: Measure ROI volumes
# ==========================================
# Outputs network component volumes with subcategories
# Volume in mm³ (0.5mm isotropic = 0.125 mm³ per voxel)
# ==========================================

if [ $stage -le 4 ]; then
  echo "-------------------------------------------"
  echo "Stage 4: Volume measurements"
  echo "-------------------------------------------"
  
  measure_dir=$expdir/measurements
  
  # Create volume measurement file
  volume_csv=$measure_dir/all_subjects_volumes.csv
  echo "subject,network_component,subcategory,volume_mm3,voxel_count" > $volume_csv
  
  for subject in $subjects; do
    echo "Measuring: $subject"
    
    subj_native=$expdir/native_rois/$subject
    
    if [ ! -d $subj_native ]; then
      echo "  [WARNING] Native ROIs not found for $subject"
      continue
    fi
    
    # 1. Vocal Motor System
    if [ -f $subj_native/vocal_motor_native.nii.gz ]; then
      voxels=$(3dBrickStat -count -non-zero $subj_native/vocal_motor_native.nii.gz 2>/dev/null)
      volume=$(echo "$voxels * 0.125" | bc)
      echo "$subject,vocal_motor,total,$volume,$voxels" >> $volume_csv
    fi
    
    # 2. Limbic Vocalization System
    if [ -f $subj_native/limbic_vocal_cortical_native.nii.gz ]; then
      voxels=$(3dBrickStat -count -non-zero $subj_native/limbic_vocal_cortical_native.nii.gz 2>/dev/null)
      volume=$(echo "$voxels * 0.125" | bc)
      echo "$subject,limbic_vocal,cortical,$volume,$voxels" >> $volume_csv
    fi
    
    if [ -f $subj_native/limbic_vocal_subcortical_native.nii.gz ]; then
      voxels=$(3dBrickStat -count -non-zero $subj_native/limbic_vocal_subcortical_native.nii.gz 2>/dev/null)
      volume=$(echo "$voxels * 0.125" | bc)
      echo "$subject,limbic_vocal,subcortical,$volume,$voxels" >> $volume_csv
    fi
    
    if [ -f $subj_native/limbic_vocal_complete_native.nii.gz ]; then
      voxels=$(3dBrickStat -count -non-zero $subj_native/limbic_vocal_complete_native.nii.gz 2>/dev/null)
      volume=$(echo "$voxels * 0.125" | bc)
      echo "$subject,limbic_vocal,total,$volume,$voxels" >> $volume_csv
    fi
    
    # 3. Auditory-Vocal Integration
    if [ -f $subj_native/auditory_vocal_native.nii.gz ]; then
      voxels=$(3dBrickStat -count -non-zero $subj_native/auditory_vocal_native.nii.gz 2>/dev/null)
      volume=$(echo "$voxels * 0.125" | bc)
      echo "$subject,auditory_vocal,total,$volume,$voxels" >> $volume_csv
    fi
    
    # 4. Basal Ganglia
    if [ -f $subj_native/basal_ganglia_native.nii.gz ]; then
      voxels=$(3dBrickStat -count -non-zero $subj_native/basal_ganglia_native.nii.gz 2>/dev/null)
      volume=$(echo "$voxels * 0.125" | bc)
      echo "$subject,basal_ganglia,total,$volume,$voxels" >> $volume_csv
    fi
    
    # 5. Complete Network
    if [ -f $subj_native/complete_vocalization_network_native.nii.gz ]; then
      voxels=$(3dBrickStat -count -non-zero $subj_native/complete_vocalization_network_native.nii.gz 2>/dev/null)
      volume=$(echo "$voxels * 0.125" | bc)
      echo "$subject,complete_network,total,$volume,$voxels" >> $volume_csv
    fi
    
    echo "  [OK] $subject measurements complete"
  done
  
  echo ""
  echo "==========================================="
  echo "ANALYSIS COMPLETE!"
  echo "==========================================="
  echo ""
  echo "Results: $expdir"
  echo "Volumes: $measure_dir/all_subjects_volumes.csv"
  echo "Documentation: $expdir/README.txt"
  echo ""
  echo "Volume measurements (mm³):"
  cat $volume_csv
  echo ""
fi

echo "[Pipeline Complete]"

# ==========================================
# DOCUMENTATION
# ==========================================

cat > $expdir/README.txt << 'EOFREADME'
MARMOSET VOCALIZATION NETWORK ANALYSIS
======================================

USAGE
-----
bash local/mri_marmoset/local/marmoset_vocalization_analysis.sh \
     --subjects "m6 m7 m8" \
     --t2_name "InplaneT2.nii.gz" \
     --mask_name "mask.nii.gz" \
     --stage 0

Parameters:
  --subjects        Space-separated subject IDs
  --t2_name         T2-weighted anatomical filename (default: InplaneT2.nii.gz)
  --mask_name       Brain mask filename (default: mask.nii.gz, optional)
  --stage           Starting stage (0-4)
  --raw_data_dir    Path to raw data directory
  --mbm_template_dir Path to MBM template directory
  --expdir          Output directory

Stages:
0 - Link anatomical data (T2 + mask)
1 - Register T2 to MBM template (10-30 min/subject)
2 - Extract ROIs from template atlases
3 - Transform ROIs to native space
4 - Measure volumes

Note: If mask_name not found, 3dAutomask will be used automatically.

NETWORK COMPONENTS
------------------

1. VOCAL MOTOR SYSTEM (9 cortical regions)
   RIKEN BMA Labels:
   31   A4ab        Primary motor - orofacial/laryngeal
   32   A4c         Primary motor - supplementary motor-like
   33   A6DC        Dorsal caudal premotor
   34   A6DR        Dorsal rostral premotor
   35   A6M         Medial premotor (SMA-like vocalization initiation)
   36   A6Va        Ventral anterior premotor
   37   A6Vb        Ventral anterior premotor
   38   A8aD        Frontal eye field (dorsal) - coordinates calls with gaze
   39   A8Av        Frontal eye field (ventral)
   
   Human Comparison: BA4/6 motor + Broca's area (BA44/45)
   Key Difference: Marmosets LACK Broca's area homolog (no syntactic language)

2. LIMBIC VOCALIZATION SYSTEM (11 structures: 7 cortical + 4 subcortical)
   Cortical - RIKEN BMA:
   57   A24a        Anterior cingulate (ventral ACC)
   58   A24b        Anterior cingulate
   59   A24c        Anterior cingulate (dorsal ACC)
   60   A24d        Anterior cingulate
   61   A25         Subgenual cingulate (emotional regulation)
   66   A32         Dorsal anterior cingulate (vocalization drive)
   67   A32V        Ventral ACC (call initiation)
   
   Subcortical - MBM:
   2    Amy         Amygdala (emotional valence of calls)
   3    Thal        Thalamus (sensorimotor relay)
   10   Acb         Nucleus accumbens (reward, motivation)
   18   PAG         Periaqueductal gray (vocal pattern generator) **CRITICAL**
   
   Human Comparison: ACC + Amygdala + Thalamus + Accumbens (highly conserved)
   Note: PAG = brainstem structure, directly generates vocalization patterns

3. AUDITORY-VOCAL INTEGRATION (8 cortical regions)
   RIKEN BMA Labels:
   81   AuA1        Primary auditory cortex (core)
   76   AuAL        Anterolateral auditory belt
   77   AuCL        Caudolateral auditory belt
   78   AuCM        Caudomedial auditory belt
   80   AuML        Mediolateral auditory belt
   82   AuR         Rostral auditory (voice-selective)
   126  STR         Superior temporal rostral
   138  TPO         Temporoparietal occipital (multisensory)
   
   Human Comparison: A1/Heschl's gyrus + STG (core auditory conserved)

4. BASAL GANGLIA VOCAL CONTROL (3 bilateral structures)
   MBM Labels:
   8    Cd          Caudate (motor sequencing, vocal learning)
   9    Pu          Putamen (motor execution)
   14   GP          Globus pallidus (motor gating, vocal timing)
   
   Human Comparison: Caudate + Putamen + Pallidum (parallel organization)
   Note: MBM labels are BILATERAL (single label covers both L+R hemispheres)

VOLUME OUTPUT FORMAT
--------------------
CSV: subject, network_component, subcategory, volume_mm³, voxel_count

Example:
m6,vocal_motor,total,125.5,1004
m6,limbic_vocal,cortical,89.3,714
m6,limbic_vocal,subcortical,45.2,362
m6,limbic_vocal,total,134.5,1076
m6,auditory_vocal,total,210.8,1686
m6,basal_ganglia,total,98.4,787
m6,complete_network,total,569.2,4553

CROSS-SPECIES NORMALIZATION
----------------------------
For comparison with human data:
- Normalize by total brain volume (marmoset ~8g, human ~1400g = 175:1 ratio)
- Or normalize by body weight (marmoset ~350g, human ~70kg = 200:1 ratio)

REFERENCES
----------
Takahashi DY, Fenley AR, Teramoto Y, et al. (2015). The developmental dynamics 
of marmoset monkey vocal production. Science 349:734-738.

Takahashi DY, Liao DA, Ghazanfar AA (2017). Vocal learning via social 
reinforcement by infant marmoset monkeys. Curr Biol 27:1844-1852.

Gultekin YB, Hage SR (2017). Limiting parental feedback disrupts vocal 
development in marmoset monkeys. Nat Commun 8:14046.

Liu C, et al. (2020). Marmoset Brain Mapping V3. Neuroimage 226:117620.

Miller CT, et al. (2016). Marmosets: A neuroscientific model of human social 
behavior. Neuron 90:219-233.

Schultz-Darken NJ, Braun KM, Emborg ME (2016). Neurobehavioral development 
of common marmoset monkeys. Dev Psychobiol 58:141-158.

ATLAS SOURCES
-------------
RIKEN BMA: Woodward et al. (2018). Sci Data 5:180009.
MBM v3: Liu et al. (2020). Neuroimage 226:117620.
Label tables: MAM_v3.0.1_labels/*.csv
EOFREADME

echo "Documentation created: $expdir/README.txt"

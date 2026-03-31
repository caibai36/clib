#!/bin/bash

# ==========================================
# MARMOSET-HUMAN ANATOMICAL DEVELOPMENTAL ANALYSIS
# ==========================================
# Goal: Compare brain region volumes of affective vocalization areas
#       across development in marmosets vs humans
#
# Input:
#   - Marmoset T2 scans (ages 14-115 months, N=32)
#   - Human T1 FreeSurfer segmentations (infant data)
#
# Output:
#   - ROI volumes for each subject (marmoset & human)
#   - Developmental trajectories (volume vs age)
#   - Cross-species comparison statistics
# ==========================================

# Paths
MBM_DIR="/data02/share/bin-wu/data/marmoset/brain/nih_uwo/marmoset_brain_mapping_v3/Marmoset_Brain_Mappping_v3.0.1/MBM_v3.0.1_0.5mm"
BASE_DIR="/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_mri_s0/exp/mri/sandbox/marmoset_registration"
OUTPUT_DIR="$BASE_DIR/anatomical_development"

mkdir -p $OUTPUT_DIR/roi_volumes
mkdir -p $OUTPUT_DIR/statistics
mkdir -p $OUTPUT_DIR/figures

echo "=========================================="
echo "ANATOMICAL DEVELOPMENTAL ANALYSIS"
echo "=========================================="
echo ""

# ==========================================
# PART 1: MARMOSET VOCALIZATION ROI HOMOLOGY
# ==========================================

cat > $OUTPUT_DIR/marmoset_human_roi_mapping.txt << 'EOFMAP'
====================================================================================
AFFECTIVE VOCALIZATION NETWORK: MARMOSET-HUMAN HOMOLOGY
====================================================================================

Based on comparative neuroscience literature:
- Jürgens U. (2009). The neural control of vocalization in mammals
- Hage & Nieder (2016). Dual neural network model for primate vocalization
- Miller et al. (2016). Marmosets as a model for vocal development

====================================================================================
1. VOCAL MOTOR NETWORK
====================================================================================

HUMAN (Desikan-Killiany)              MARMOSET (MBM Paxinos)          FUNCTION
--------------------                   ---------------------           --------
Precentral (BA 4)                 →   Area 4 (M1)                    Primary motor
Paracentral                       →   Medial Area 6                  Supplementary motor
Caudal Middle Frontal             →   Dorsal Area 6 (PMd)            Premotor
Pars Opercularis (BA 44)          →   Area 45/Pro (ventral PFC)      Motor planning*

*Note: Marmosets have vocal motor cortex but NOT full Broca's homolog
      Ventral premotor involved in oro-facial control

MARMOSET MBM LABELS TO EXTRACT:
- Motor cortex (Area 4): Labels TBD from atlas
- Premotor (Area 6): Labels TBD
- Frontal regions: Labels TBD

====================================================================================
2. LIMBIC VOCALIZATION NETWORK
====================================================================================

HUMAN (Desikan-Killiany)              MARMOSET (MBM Paxinos)          FUNCTION
--------------------                   ---------------------           --------
Insula                            →   Insula (Ia, Id, Ig)            Interoception, emotion
Rostral Anterior Cingulate        →   ACC (Area 24, 32)              Voluntary vocal control
Caudal Anterior Cingulate         →   Mid Cingulate (Area 24)        Conflict monitoring
Medial Orbitofrontal              →   Medial OFC (Area 14)           Emotional valence
Lateral Orbitofrontal             →   Lateral OFC (Area 11, 13)      Reward processing

Subcortical (FreeSurfer aseg)         Marmoset Subcortical            Function
-----------------------------         --------------------            --------
Amygdala                          →   Amygdala                       Emotional vocalization
Periaqueductal Gray (PAG)         →   PAG                            Innate call production

MARMOSET MBM LABELS:
- Insula: Labels TBD
- Anterior cingulate: Labels for Area 24, 32
- Orbitofrontal: Area 11, 13, 14
- Amygdala: Subcortical atlas
- PAG: Brainstem atlas (if available)

====================================================================================
3. AUDITORY-VOCAL INTEGRATION NETWORK
====================================================================================

HUMAN (Desikan-Killiany)              MARMOSET (MBM Paxinos)          FUNCTION
--------------------                   ---------------------           --------
Transverse Temporal (Heschl's)    →   Core Auditory (A1)             Primary auditory
Superior Temporal                 →   Belt Auditory (R, CM, CL)      Auditory association
Banks STS                         →   Superior Temporal Sulcus       Audiovisual integration
Supramarginal                     →   Parietal (Area 7)              Sensorimotor integration

MARMOSET MBM LABELS:
- Primary auditory: A1 labels
- Auditory belt: R, CM, CL, ML labels
- Superior temporal regions
- Parietal areas

====================================================================================
4. BASAL GANGLIA VOCAL CONTROL
====================================================================================

HUMAN (FreeSurfer aseg)               MARMOSET (Subcortical)          FUNCTION
-------------------                   ----------------------          --------
Caudate                           →   Caudate                        Vocal initiation
Putamen                           →   Putamen                        Motor sequencing
Pallidum                          →   Globus Pallidus                Motor gating
Thalamus                          →   Thalamus                       Relay/timing

MARMOSET MBM SUBCORTICAL LABELS:
- Caudate
- Putamen
- Globus pallidus
- Thalamus (specific nuclei if available)

====================================================================================
DEVELOPMENTAL PREDICTIONS
====================================================================================

Based on vocal learning literature:

EARLY DEVELOPMENT (Infancy):
- Subcortical structures (amygdala, PAG, striatum) mature early
- Innate call production functional from birth
- Limited cortical-subcortical connectivity

MID DEVELOPMENT (Juvenile):
- Cortical motor regions develop
- Auditory-motor integration emerges
- Vocal learning window opens
- Increased frontal-limbic connectivity

LATE DEVELOPMENT (Adult):
- Full cortical-subcortical integration
- Mature vocal motor control
- Complex turn-taking abilities
- Adult-like connectivity patterns

CROSS-SPECIES COMPARISON:
- Marmosets: Faster maturation timeline (weeks-months)
- Humans: Slower maturation (months-years)
- Similar network architecture
- Different developmental trajectories

EOFMAP

echo "[OK] Created marmoset-human ROI mapping guide"
echo "     File: $OUTPUT_DIR/marmoset_human_roi_mapping.txt"
echo ""

# ==========================================
# PART 2: EXTRACT MARMOSET ROI VOLUMES
# ==========================================

cat > $OUTPUT_DIR/extract_marmoset_roi_volumes.sh << 'EOFVOL'
#!/bin/bash

# ==========================================
# EXTRACT MARMOSET ROI VOLUMES
# ==========================================
# For each subject, extract volumes of vocalization-related regions
# ==========================================

MBM_DIR="/data02/share/bin-wu/data/marmoset/brain/nih_uwo/marmoset_brain_mapping_v3/Marmoset_Brain_Mappping_v3.0.1/MBM_v3.0.1_0.5mm"
BASE_DIR="/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_mri_s0/exp/mri/sandbox/marmoset_registration"
OUTPUT_DIR="$BASE_DIR/anatomical_development"

# Subject list (all 32 marmosets)
SUBJECTS=(m6 m7 m8 m9 m10 m11 m12 m14 m15 m16 m17 m18 m19 m20 m21 m22 m23 m24 m25 m26 m27 m28 m29 m30 m31 m32)

# Age metadata (months) - UPDATE WITH ACTUAL AGES!
declare -A AGES
AGES[m6]=14
AGES[m7]=18
# ... ADD ALL SUBJECT AGES ...
AGES[m32]=115

echo "Extracting ROI volumes for ${#SUBJECTS[@]} subjects..."
echo ""

# Create output CSV
CSV_FILE="$OUTPUT_DIR/roi_volumes/marmoset_roi_volumes.csv"
echo "subject,age_months,network,roi_name,hemisphere,volume_mm3" > $CSV_FILE

# ==========================================
# ROI LABEL DEFINITIONS
# ==========================================
# TODO: UPDATE THESE LABELS BASED ON ACTUAL MBM ATLAS!
#
# Check labels with:
#   3dinfo -label $MBM_DIR/atlas_MBM_cortex_vPaxinos_0.5mm.nii.gz
#   3dinfo -label $MBM_DIR/atlas_MBM_subcortical_beta_0.5mm.nii.gz
# ==========================================

# Example structure (LABELS ARE PLACEHOLDERS!)
declare -A MOTOR_LABELS
MOTOR_LABELS[left_M1]=101
MOTOR_LABELS[right_M1]=102
MOTOR_LABELS[left_PMd]=103
MOTOR_LABELS[right_PMd]=104

declare -A LIMBIC_LABELS
LIMBIC_LABELS[left_insula]=201
LIMBIC_LABELS[right_insula]=202
LIMBIC_LABELS[left_ACC]=203
LIMBIC_LABELS[right_ACC]=204

declare -A AUDITORY_LABELS
AUDITORY_LABELS[left_A1]=301
AUDITORY_LABELS[right_A1]=302
AUDITORY_LABELS[left_belt]=303
AUDITORY_LABELS[right_belt]=304

declare -A STRIATAL_LABELS
STRIATAL_LABELS[left_caudate]=401
STRIATAL_LABELS[right_caudate]=402
STRIATAL_LABELS[left_putamen]=403
STRIATAL_LABELS[right_putamen]=404

# ==========================================
# EXTRACT VOLUMES FOR EACH SUBJECT
# ==========================================

for subj in "${SUBJECTS[@]}"; do
    echo "Processing $subj (age: ${AGES[$subj]} months)..."

    # Check if subject has registered T2 to template
    ANAT_DIR="$BASE_DIR/anatomical/$subj"

    if [ ! -d "$ANAT_DIR" ]; then
        echo "  WARNING: No anatomical data for $subj"
        continue
    fi

    # We need the transform from native T2 space to MBM template
    # This should have been created during registration
    TRANSFORM="$ANAT_DIR/transforms/T2_to_template_"

    # Option 1: Extract volumes directly from template-space segmentation
    # (After registering subject T2 to template, apply atlas labels)

    # Option 2: Warp atlas to subject space and calculate volumes there
    # (Better if you want native-space measurements)

    # For now, we'll use template space (simpler, allows group comparison)

    # Extract Motor ROI volumes
    for roi in "${!MOTOR_LABELS[@]}"; do
        label=${MOTOR_LABELS[$roi]}

        # Create ROI mask
        3dcalc -a $MBM_DIR/atlas_MBM_cortex_vPaxinos_0.5mm.nii.gz \
               -expr "equals(a,$label)" \
               -prefix /tmp/${subj}_${roi}_mask.nii.gz

        # Calculate volume (voxels × voxel size)
        # For 0.5mm isotropic: voxel volume = 0.5^3 = 0.125 mm³
        nvoxels=$(3dBrickStat -count -non-zero /tmp/${subj}_${roi}_mask.nii.gz)
        volume=$(echo "$nvoxels * 0.125" | bc -l)

        # Determine hemisphere
        if [[ $roi == left_* ]]; then
            hemi="L"
        else
            hemi="R"
        fi

        roi_clean=${roi#left_}
        roi_clean=${roi_clean#right_}

        # Add to CSV
        echo "$subj,${AGES[$subj]},motor,$roi_clean,$hemi,$volume" >> $CSV_FILE

        rm /tmp/${subj}_${roi}_mask.nii.gz
    done

    # Repeat for LIMBIC, AUDITORY, STRIATAL networks...
    # (Same pattern as above)

done

echo ""
echo "[OK] ROI volumes extracted"
echo "     Output: $CSV_FILE"
echo ""
echo "Next: Run developmental analysis with this CSV"

EOFVOL

chmod +x $OUTPUT_DIR/extract_marmoset_roi_volumes.sh

echo "[OK] Created marmoset volume extraction script"
echo "     File: $OUTPUT_DIR/extract_marmoset_roi_volumes.sh"
echo ""

# ==========================================
# PART 3: HUMAN ROI VOLUME EXTRACTION
# ==========================================

cat > $OUTPUT_DIR/extract_human_roi_volumes.sh << 'EOFHUM'
#!/bin/bash

# ==========================================
# EXTRACT HUMAN ROI VOLUMES FROM FREESURFER
# ==========================================
# Based on your existing human T1 analysis script
# ==========================================

OUTPUT_DIR="/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_mri_s0/exp/mri/sandbox/marmoset_registration/anatomical_development"

# Path to FreeSurfer subjects
SUBJECTS_DIR="/path/to/your/freesurfer/subjects"  # UPDATE THIS!

# Human subject list
HUMAN_SUBJECTS=(infant01 infant02 infant03)  # UPDATE WITH YOUR SUBJECTS!

# Age metadata (months)
declare -A HUMAN_AGES
HUMAN_AGES[infant01]=3
HUMAN_AGES[infant02]=6
HUMAN_AGES[infant03]=12

echo "Extracting human ROI volumes from FreeSurfer..."
echo ""

# Create output CSV
CSV_FILE="$OUTPUT_DIR/roi_volumes/human_roi_volumes.csv"
echo "subject,age_months,network,roi_name,hemisphere,volume_mm3" > $CSV_FILE

# ==========================================
# FREESURFER REGION DEFINITIONS
# ==========================================
# Based on Desikan-Killiany atlas
# ==========================================

# Vocal Motor Network
MOTOR_ROIS_LH=(
    "ctx-lh-precentral"
    "ctx-lh-paracentral"
    "ctx-lh-caudalmiddlefrontal"
    "ctx-lh-parsopercularis"
)

MOTOR_ROIS_RH=(
    "ctx-rh-precentral"
    "ctx-rh-paracentral"
    "ctx-rh-caudalmiddlefrontal"
    "ctx-rh-parsopercularis"
)

# Limbic Network
LIMBIC_ROIS_LH=(
    "ctx-lh-insula"
    "ctx-lh-rostralanteriorcingulate"
    "ctx-lh-caudalanteriorcingulate"
    "ctx-lh-medialorbitofrontal"
    "ctx-lh-lateralorbitofrontal"
)

LIMBIC_ROIS_RH=(
    "ctx-rh-insula"
    "ctx-rh-rostralanteriorcingulate"
    "ctx-rh-caudalanteriorcingulate"
    "ctx-rh-medialorbitofrontal"
    "ctx-rh-lateralorbitofrontal"
)

# Subcortical limbic
LIMBIC_SUBCORT=(
    "Left-Amygdala"
    "Right-Amygdala"
)

# Auditory Network
AUDITORY_ROIS_LH=(
    "ctx-lh-transversetemporal"
    "ctx-lh-superiortemporal"
    "ctx-lh-bankssts"
    "ctx-lh-supramarginal"
)

AUDITORY_ROIS_RH=(
    "ctx-rh-transversetemporal"
    "ctx-rh-superiortemporal"
    "ctx-rh-bankssts"
    "ctx-rh-supramarginal"
)

# Basal Ganglia
STRIATAL_ROIS=(
    "Left-Caudate"
    "Right-Caudate"
    "Left-Putamen"
    "Right-Putamen"
    "Left-Pallidum"
    "Right-Pallidum"
    "Left-Thalamus-Proper"
    "Right-Thalamus-Proper"
)

# ==========================================
# EXTRACT VOLUMES
# ==========================================

for subj in "${HUMAN_SUBJECTS[@]}"; do
    echo "Processing $subj (age: ${HUMAN_AGES[$subj]} months)..."

    SUBJ_DIR="$SUBJECTS_DIR/$subj"

    if [ ! -f "$SUBJ_DIR/stats/aseg.stats" ]; then
        echo "  WARNING: No FreeSurfer segmentation for $subj"
        continue
    fi

    # Extract Motor ROI volumes (left hemisphere)
    for roi in "${MOTOR_ROIS_LH[@]}"; do
        roi_clean=${roi#ctx-lh-}
        volume=$(grep "^${roi}" $SUBJ_DIR/stats/lh.aparc.stats | awk '{print $4}')

        if [ ! -z "$volume" ]; then
            echo "$subj,${HUMAN_AGES[$subj]},motor,$roi_clean,L,$volume" >> $CSV_FILE
        fi
    done

    # Motor right hemisphere
    for roi in "${MOTOR_ROIS_RH[@]}"; do
        roi_clean=${roi#ctx-rh-}
        volume=$(grep "^${roi}" $SUBJ_DIR/stats/rh.aparc.stats | awk '{print $4}')

        if [ ! -z "$volume" ]; then
            echo "$subj,${HUMAN_AGES[$subj]},motor,$roi_clean,R,$volume" >> $CSV_FILE
        fi
    done

    # Limbic cortical (left)
    for roi in "${LIMBIC_ROIS_LH[@]}"; do
        roi_clean=${roi#ctx-lh-}
        volume=$(grep "^${roi}" $SUBJ_DIR/stats/lh.aparc.stats | awk '{print $4}')

        if [ ! -z "$volume" ]; then
            echo "$subj,${HUMAN_AGES[$subj]},limbic,$roi_clean,L,$volume" >> $CSV_FILE
        fi
    done

    # Limbic cortical (right)
    for roi in "${LIMBIC_ROIS_RH[@]}"; do
        roi_clean=${roi#ctx-rh-}
        volume=$(grep "^${roi}" $SUBJ_DIR/stats/rh.aparc.stats | awk '{print $4}')

        if [ ! -z "$volume" ]; then
            echo "$subj,${HUMAN_AGES[$subj]},limbic,$roi_clean,R,$volume" >> $CSV_FILE
        fi
    done

    # Limbic subcortical
    for roi in "${LIMBIC_SUBCORT[@]}"; do
        if [[ $roi == Left-* ]]; then
            roi_clean=${roi#Left-}
            hemi="L"
        else
            roi_clean=${roi#Right-}
            hemi="R"
        fi

        volume=$(grep "^${roi}" $SUBJ_DIR/stats/aseg.stats | awk '{print $4}')

        if [ ! -z "$volume" ]; then
            echo "$subj,${HUMAN_AGES[$subj]},limbic,$roi_clean,$hemi,$volume" >> $CSV_FILE
        fi
    done

    # Auditory (left)
    for roi in "${AUDITORY_ROIS_LH[@]}"; do
        roi_clean=${roi#ctx-lh-}
        volume=$(grep "^${roi}" $SUBJ_DIR/stats/lh.aparc.stats | awk '{print $4}')

        if [ ! -z "$volume" ]; then
            echo "$subj,${HUMAN_AGES[$subj]},auditory,$roi_clean,L,$volume" >> $CSV_FILE
        fi
    done

    # Auditory (right)
    for roi in "${AUDITORY_ROIS_RH[@]}"; do
        roi_clean=${roi#ctx-rh-}
        volume=$(grep "^${roi}" $SUBJ_DIR/stats/rh.aparc.stats | awk '{print $4}')

        if [ ! -z "$volume" ]; then
            echo "$subj,${HUMAN_AGES[$subj]},auditory,$roi_clean,R,$volume" >> $CSV_FILE
        fi
    done

    # Basal ganglia
    for roi in "${STRIATAL_ROIS[@]}"; do
        if [[ $roi == Left-* ]]; then
            roi_clean=${roi#Left-}
            hemi="L"
        else
            roi_clean=${roi#Right-}
            hemi="R"
        fi

        volume=$(grep "^${roi}" $SUBJ_DIR/stats/aseg.stats | awk '{print $4}')

        if [ ! -z "$volume" ]; then
            echo "$subj,${HUMAN_AGES[$subj]},striatal,$roi_clean,$hemi,$volume" >> $CSV_FILE
        fi
    done

done

echo ""
echo "[OK] Human ROI volumes extracted"
echo "     Output: $CSV_FILE"

EOFHUM

chmod +x $OUTPUT_DIR/extract_human_roi_volumes.sh

echo "[OK] Created human volume extraction script"
echo "     File: $OUTPUT_DIR/extract_human_roi_volumes.sh"
echo ""

# ==========================================
# PART 4: DEVELOPMENTAL TRAJECTORY ANALYSIS
# ==========================================

cat > $OUTPUT_DIR/analyze_developmental_trajectories.R << 'EOFR'
#!/usr/bin/env Rscript

# ==========================================
# CROSS-SPECIES DEVELOPMENTAL TRAJECTORY ANALYSIS
# ==========================================
# Compares volumetric growth of vocalization regions
# between marmosets and humans
# ==========================================

library(tidyverse)
library(lme4)
library(ggplot2)
library(cowplot)

# Load data
marmoset_data <- read_csv("roi_volumes/marmoset_roi_volumes.csv")
human_data <- read_csv("roi_volumes/human_roi_volumes.csv")

# Add species column
marmoset_data$species <- "marmoset"
human_data$species <- "human"

# Combine
all_data <- bind_rows(marmoset_data, human_data)

# ==========================================
# SCALE AGES FOR COMPARISON
# ==========================================
# Marmosets reach adulthood ~18-24 months
# Humans reach vocal maturity ~7-10 years
# Scale both to 0-1 (proportion of maturation)

all_data <- all_data %>%
  mutate(
    age_scaled = case_when(
      species == "marmoset" ~ age_months / 24,  # Adult at 24 months
      species == "human" ~ age_months / 120     # Adult at 10 years
    ),
    age_scaled = pmin(age_scaled, 1.0)  # Cap at 1.0
  )

# ==========================================
# STATISTICAL MODELS
# ==========================================

# Linear mixed model for each network
networks <- c("motor", "limbic", "auditory", "striatal")

results <- list()

for (net in networks) {
  cat(paste("\nAnalyzing", net, "network...\n"))

  net_data <- all_data %>% filter(network == net)

  # Model: volume ~ age * species + (1|subject) + (1|roi_name)
  model <- lmer(
    log(volume_mm3) ~ age_scaled * species + hemisphere +
      (1|subject) + (1|roi_name),
    data = net_data
  )

  results[[net]] <- summary(model)

  cat("Fixed effects:\n")
  print(coef(summary(model)))
  cat("\n")
}

# ==========================================
# VISUALIZATION
# ==========================================

# Plot 1: Developmental trajectories by network
p1 <- ggplot(all_data, aes(x = age_scaled, y = volume_mm3, color = species)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "loess", se = TRUE) +
  facet_wrap(~network, scales = "free_y") +
  scale_color_manual(values = c("marmoset" = "#E69F00", "human" = "#0072B2")) +
  labs(
    title = "Developmental Trajectories of Vocalization Networks",
    x = "Developmental Stage (scaled)",
    y = "Volume (mm³)",
    color = "Species"
  ) +
  theme_minimal()

ggsave("figures/developmental_trajectories_by_network.pdf", p1, width = 12, height = 8)

# Plot 2: Growth rates (slope comparison)
growth_rates <- all_data %>%
  group_by(network, species, roi_name) %>%
  summarize(
    n = n(),
    growth_rate = coef(lm(log(volume_mm3) ~ age_scaled))[2]
  ) %>%
  filter(n >= 3)  # Only ROIs with enough data points

p2 <- ggplot(growth_rates, aes(x = network, y = growth_rate, fill = species)) +
  geom_boxplot() +
  scale_fill_manual(values = c("marmoset" = "#E69F00", "human" = "#0072B2")) +
  labs(
    title = "Comparison of Growth Rates Across Networks",
    x = "Network",
    y = "Growth Rate (log volume per dev. stage)",
    fill = "Species"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("figures/growth_rate_comparison.pdf", p2, width = 10, height = 6)

# Plot 3: Specific ROI comparisons
# Focus on homologous regions

homologous_pairs <- list(
  motor = c("precentral", "M1"),
  limbic = c("insula", "insula"),
  auditory = c("transversetemporal", "A1"),
  striatal = c("Caudate", "caudate")
)

# (Additional plotting code for specific ROI comparisons)

# ==========================================
# EXPORT RESULTS
# ==========================================

# Save statistical results
sink("statistics/developmental_trajectory_models.txt")
cat("======================================================\n")
cat("DEVELOPMENTAL TRAJECTORY ANALYSIS - MODEL RESULTS\n")
cat("======================================================\n\n")

for (net in networks) {
  cat(paste("\n", toupper(net), "NETWORK\n"))
  cat("========================================\n")
  print(results[[net]])
  cat("\n")
}
sink()

# Save growth rate table
write_csv(growth_rates, "statistics/roi_growth_rates.csv")

cat("\n[OK] Analysis complete!\n")
cat("  Figures saved to: figures/\n")
cat("  Statistics saved to: statistics/\n")

EOFR

chmod +x $OUTPUT_DIR/analyze_developmental_trajectories.R

echo "[OK] Created R analysis script"
echo "     File: $OUTPUT_DIR/analyze_developmental_trajectories.R"
echo ""

# ==========================================
# SUMMARY
# ==========================================

echo "=========================================="
echo "ANATOMICAL DEVELOPMENT WORKFLOW CREATED"
echo "=========================================="
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Created files:"
echo "  1. marmoset_human_roi_mapping.txt       - Homology mapping"
echo "  2. extract_marmoset_roi_volumes.sh      - Marmoset volume extraction"
echo "  3. extract_human_roi_volumes.sh         - Human volume extraction"
echo "  4. analyze_developmental_trajectories.R - Statistical analysis"
echo ""
echo "=========================================="
echo "NEXT STEPS"
echo "=========================================="
echo ""
echo "STEP 1: Update MBM atlas labels"
echo "  Edit: extract_marmoset_roi_volumes.sh"
echo "  Check actual labels from MBM atlas"
echo "  Run: 3dinfo -label $MBM_DIR/atlas_MBM_cortex_vPaxinos_0.5mm.nii.gz"
echo ""
echo "STEP 2: Add subject age metadata"
echo "  Edit both extraction scripts"
echo "  Add actual ages for all subjects"
echo ""
echo "STEP 3: Register marmoset T2 to MBM template"
echo "  For each subject, run T2 registration"
echo "  This creates transforms needed for ROI extraction"
echo ""
echo "STEP 4: Extract marmoset volumes"
echo "  bash extract_marmoset_roi_volumes.sh"
echo ""
echo "STEP 5: Extract human volumes (if not done)"
echo "  Update FreeSurfer paths"
echo "  bash extract_human_roi_volumes.sh"
echo ""
echo "STEP 6: Run developmental analysis"
echo "  Rscript analyze_developmental_trajectories.R"
echo ""
echo "STEP 7: Interpret results"
echo "  Compare growth trajectories"
echo "  Identify species differences"
echo "  Relate to functional connectivity findings"
echo ""
echo "=========================================="
echo "IMPORTANT NOTES"
echo "=========================================="
echo ""
echo "For anatomical developmental analysis:"
echo "  - You DO NOT need BOLD preprocessing"
echo "  - You only need T2 anatomical scans"
echo "  - Register T2 → MBM template"
echo "  - Extract ROI volumes from template space"
echo "  - Compare volume trajectories across species"
echo ""
echo "This is SEPARATE from functional connectivity analysis!"
echo ""
echo "For complete developmental study, you can do BOTH:"
echo "  1. Anatomical analysis (this workflow)"
echo "  2. Functional connectivity analysis (requires BOLD preprocessing)"
echo ""
echo "Both approaches are complementary and address different questions:"
echo "  - Anatomical: How do brain STRUCTURES develop?"
echo "  - Functional: How does brain CONNECTIVITY develop?"
echo ""
echo "=========================================="

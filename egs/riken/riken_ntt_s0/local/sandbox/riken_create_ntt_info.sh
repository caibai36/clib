#!/bin/bash

# Base directory
BASE_DIR="exp/mae_feat_dim_reduction/python_script"

# Python script path
PYTHON_SCRIPT="local/sandbox/riken_create_ntt_info.py"

# Meta speaker path (shared across all cases)
META_SPEAKER_PATH="/data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/processed/metas_speaker_force_alignment_phone/meta_all_speaker.csv"

# Array of cases
CASES=(
    "mae_pretrain_sa_data_kk"
    "mae_pretrain_sa_data_ma"
    "mae_pretrain_sa_data_mk"
    "mae_pretrain_sa_data_sa"
    "mae_pretrain_sa_data_sk"
)

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Check if meta_speaker file exists
if [ ! -f "$META_SPEAKER_PATH" ]; then
    echo "ERROR: Meta speaker file not found: $META_SPEAKER_PATH"
    exit 1
fi

echo "Processing ${#CASES[@]} cases..."
echo "Using Python script: $PYTHON_SCRIPT"
echo "Using meta speaker: $META_SPEAKER_PATH"
echo "================================"

SUCCESS_COUNT=0
FAILED_COUNT=0

# Loop through each case
for case in "${CASES[@]}"; do
    echo ""
    echo "Processing: $case"
    echo "--------------------------------"

    # Define paths
    CASE_DIR="${BASE_DIR}/${case}"
    CONFIG_PATH="${CASE_DIR}/config_${case}.csv"
    TSNE_PATH="${CASE_DIR}/tsne_result_${case}.npy"
    OUTPUT_DIR="${CASE_DIR}"

    # Check if required files exist
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "ERROR: Config file not found: $CONFIG_PATH"
        ((FAILED_COUNT++))
        continue
    fi

    if [ ! -f "$TSNE_PATH" ]; then
        echo "ERROR: t-SNE file not found: $TSNE_PATH"
        ((FAILED_COUNT++))
        continue
    fi

    # Run the Python script
    python "$PYTHON_SCRIPT" \
        --config_path "$CONFIG_PATH" \
        --meta_speaker_path "$META_SPEAKER_PATH" \
        --tsne_path "$TSNE_PATH" \
        --output_dir "$OUTPUT_DIR"

    # Check if outputs were created
    INFO_PATH="${OUTPUT_DIR}/info.csv"
    INFO_MAE_PATH="${OUTPUT_DIR}/info_mae.csv"

    if [ -f "$INFO_PATH" ] && [ -f "$INFO_MAE_PATH" ]; then
        echo "✓ Successfully created both files"

        # Show line counts and sizes
        INFO_LINES=$(wc -l < "$INFO_PATH")
        INFO_SIZE=$(du -h "$INFO_PATH" | cut -f1)
        INFO_MAE_LINES=$(wc -l < "$INFO_MAE_PATH")
        INFO_MAE_SIZE=$(du -h "$INFO_MAE_PATH" | cut -f1)

        echo "  info.csv     - Lines: $INFO_LINES, Size: $INFO_SIZE"
        echo "  info_mae.csv - Lines: $INFO_MAE_LINES, Size: $INFO_MAE_SIZE"
        ((SUCCESS_COUNT++))
    else
        echo "✗ Failed to create output files"
        [ ! -f "$INFO_PATH" ] && echo "  Missing: info.csv"
        [ ! -f "$INFO_MAE_PATH" ] && echo "  Missing: info_mae.csv"
        ((FAILED_COUNT++))
    fi
done

echo ""
echo "================================"
echo "Summary:"
echo "  Successfully processed: $SUCCESS_COUNT/${#CASES[@]}"
echo "  Failed: $FAILED_COUNT/${#CASES[@]}"
echo ""

if [ $SUCCESS_COUNT -gt 0 ]; then
    echo "Created info.csv files:"
    ls -lh ${BASE_DIR}/mae_pretrain_sa_data_*/info.csv 2>/dev/null
    echo ""
    echo "Created info_mae.csv files:"
    ls -lh ${BASE_DIR}/mae_pretrain_sa_data_*/info_mae.csv 2>/dev/null
fi

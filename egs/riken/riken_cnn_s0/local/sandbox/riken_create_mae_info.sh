#!/bin/bash

# Base directory
BASE_DIR="exp/mae_feat_dim_reduction/python_script_nas5"

# Python script path
PYTHON_SCRIPT="local/sandbox/riken_create_mae_info.py"

# Array of cases
CASES=(
    "b1_f1_15weeks"
    "b1_f2_15weeks"
    "b2_f1_15weeks"
    "b2_f2_15weeks"
    "b3_f1_15weeks"
    "b3_f2_15weeks"
    "b4_f1_15weeks"
    "b4_f2_15weeks"
)

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

echo "Processing ${#CASES[@]} cases..."
echo "Using Python script: $PYTHON_SCRIPT"
echo "================================"

SUCCESS_COUNT=0
FAILED_COUNT=0

# Loop through each case
for case in "${CASES[@]}"; do
    echo ""
    echo "Processing: $case"
    echo "--------------------------------"

    # Define paths
    CONFIG_PATH="${BASE_DIR}/${case}/config_${case}.csv"
    TSNE_PATH="${BASE_DIR}/${case}/tsne_result_${case}.npy"
    OUTPUT_PATH="${BASE_DIR}/${case}/info_mae.csv"

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
        --tsne_path "$TSNE_PATH" \
        --output_path "$OUTPUT_PATH"

    # Check if output was created
    if [ -f "$OUTPUT_PATH" ]; then
        echo "✓ Successfully created: $OUTPUT_PATH"
        # Show line count
        LINE_COUNT=$(wc -l < "$OUTPUT_PATH")
        FILE_SIZE=$(du -h "$OUTPUT_PATH" | cut -f1)
        echo "  Lines: $LINE_COUNT, Size: $FILE_SIZE"
        ((SUCCESS_COUNT++))
    else
        echo "✗ Failed to create: $OUTPUT_PATH"
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
    echo "Created files:"
    ls -lh ${BASE_DIR}/*15weeks/info_mae.csv 2>/dev/null
fi

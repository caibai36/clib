#!/bin/bash

# Set bash to 'debug' mode, it will exit on:
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands'
set -euo pipefail

# General configuration
stage=0                   # Start from stage 0
pred_resolution=0.01      # Prediction resolution in seconds
# Updated model path to use linear spectrogram (FFT=512) model
eval_model="exp/sys_mae_vit/timit/division_timit_winmid0.05size0.5shift0.05_noisekeep1_lin16k/mae_vit-run0/bs256baselr0.0001wd0.3warmup5total25avgpredwin5_dim768depth12heads12mlpdim3072patch16dropout0.1maetrain6maepretrainedbase/train/model.ckpt"  # Path to the model checkpoint
result_dir="exp/sandbox/eval/mae_vit_lin16k"  # Directory to store evaluation results
label2id_yaml="conf/dict/timit_label2id.yaml"  # Label to ID mapping file
info_json="data/timit/info.json"   # Dataset information file
division_yaml="conf/data/division_timit.yaml"  # Dataset division file

# Path to Kaldi compute-wer tool
compute_wer="/home/bin-wu/share/tools/kaldi/src/bin/compute-wer"

# Path to prediction script - Updated to use linear spectrogram version
predict_script="local/sandbox/riken_mae_vit_audio2seg_highres_padding_lin16k.py"

# GPU configuration
gpu="auto"                # GPU selection: auto, cpu, or specific device id
fixed_factor=0            # Fixed factor added to segment times
padding=true              # Whether to add padding to audio for better prediction

# Parse the options from command line arguments
. local/scripts/parse_options.sh || exit 1

# Create output directories if they don't exist
mkdir -p ${result_dir}/pred
mkdir -p ${result_dir}/ref

# Flag for padding option conversion to command line argument
padding_flag=""
if $padding; then
    padding_flag="--padding"
fi

if [ ${stage} -le 0 ]; then
    echo "Getting test utterance IDs from division YAML file..."
    # Use Python to extract test IDs from YAML file
    test_ids=$(python -c "
import yaml
with open('${division_yaml}', 'r') as f:
    division = yaml.safe_load(f)
for uttid in division['test']:
    print(uttid)
")
    echo "Found $(echo "${test_ids}" | wc -l) test utterances"
fi

if [ ${stage} -le 1 ]; then
    echo "Generating predictions for test utterances using linear spectrograms (FFT=512)..."
    for uttid in ${test_ids}; do
        echo "Processing ${uttid}..."

        # Get wav path from info.json using Python
        wav_path=$(python -c "
import json
with open('${info_json}', 'r') as f:
    info = json.load(f)
print(info['${uttid}']['wav'])
")

        # Generate predictions using the neural network model
        python ${predict_script} \
            --label2id_yaml ${label2id_yaml} \
            --eval_model ${eval_model} \
            --pred_resolution ${pred_resolution} \
            --wav_file ${wav_path} \
            --out_seg_file ${result_dir}/pred/${uttid}.txt \
            --gpu ${gpu} \
            --fixed_factor ${fixed_factor} \
            ${padding_flag}
    done
fi

if [ ${stage} -le 2 ]; then
    echo "Creating reference and hypothesis files for evaluation..."

    # Clear previous reference and hypothesis files
    > ${result_dir}/ref.txt
    > ${result_dir}/hyp.txt

    for uttid in ${test_ids}; do
        # Get reference segment file path from info.json
        ref_seg_path=$(python -c "
import json
with open('${info_json}', 'r') as f:
    info = json.load(f)
print(info['${uttid}']['seg'])
")

        # Save a copy of the reference to result_dir/ref for convenience
        cp ${ref_seg_path} ${result_dir}/ref/${uttid}.txt

        pred_seg_file="${result_dir}/pred/${uttid}.txt"

        # Format reference segments for compute-wer
        # Extract phone labels from reference file and format for Kaldi compute-wer
        python -c "
import os

uttid = '${uttid}'
ref_file = '${ref_seg_path}'
ref_content = []

with open(ref_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            start, end, phone = parts[0:3]
            ref_content.append(f'{start} {end} {phone}')

with open('${result_dir}/ref.txt', 'a') as out:
    out.write(f'{uttid} ' + ' '.join([x.split()[-1] for x in ref_content]) + '\\n')
"

        # Format predicted segments for compute-wer
        # Extract phone labels from prediction file and format for Kaldi compute-wer
        python -c "
import os

uttid = '${uttid}'
hyp_file = '${pred_seg_file}'
hyp_content = []

with open(hyp_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            start, end, phone = parts[0:3]
            hyp_content.append(f'{start} {end} {phone}')

with open('${result_dir}/hyp.txt', 'a') as out:
    out.write(f'{uttid} ' + ' '.join([x.split()[-1] for x in hyp_content]) + '\\n')
"
    done
fi

if [ ${stage} -le 3 ]; then
    echo "Computing WER using Kaldi compute-wer tool..."

    # Run Kaldi's compute-wer to evaluate phone error rate
    ${compute_wer} ark,t:${result_dir}/ref.txt ark,t:${result_dir}/hyp.txt > ${result_dir}/wer_results.txt

    # Display results
    echo "Evaluation results:"
    cat ${result_dir}/wer_results.txt
fi

echo "Evaluation completed successfully!"

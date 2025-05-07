#!/bin/bash
# Implemented by bin-wu at 20:45 on 2025/05/04
#
# Script for NTT infant speech force alignment following WSJ-style pipeline
# Usage: bash local/ntt_kaldi/run.sh [options]
#
# This script performs forced alignment on Japanese infant speech data using Kaldi.
# It follows the standard WSJ recipe with adaptations for Japanese kana tokens.
#
# The script performs the following steps:
# 1. Data preparation - Converting NTT format to Kaldi format
# 2. Feature extraction - Computing MFCC features
# 3. Model training - Monophone, triphone, LDA+MLLT, SAT models
# 4. Force alignment - Aligning audio with transcripts
# 5. Output formatting - Converting alignments to begin/end/token format
#
# Two types of alignments are produced:
# - Regular alignments based directly on Kaldi
# - Fixed alignments that extend the Kaldi alignments to match original segments
#
# Example:
# ./local/sandbox/riken_ntt_kaldi_force_alignment/run.sh --stage 0 |& tee logs/kaldi_force_alignment.log

# Command line options
stage=8            # Start from this stage (0=data prep, 1=features, etc.)
train=true         # Set to false to disable model training
decode=true        # Set to false to disable decoding/alignment
nj=4               # Number of parallel jobs for feature extraction and alignment
min_segment_length=0.001 # default 0.01; if corpus has shorter segments, use smaller ones

# Directory paths
script_dir=local/sandbox/riken_ntt_kaldi_force_alignment  # Directory containing helper scripts
data_dir=data/ntt_kaldi               # Directory for data preparation

# Input files
# info.json maps utterance IDs to wav and token files
# Format: {"uttid": {"wav": "path/to/wav", "token": "path/to/token", ...}, ...}
# Each line of token files is begin_sec\tend_sec\ttoken1 token2, token3,...,
# where force alignment will align the time stamps for each token
data_info_json=data/ntt_infant/info.json

# division_yaml defines train/dev/test splits
# Format: train: [id1, id2, ...], dev: [id3, id4, ...], test: [id5, id6, ...] (id is uttid)
division_yaml=conf/data/division_ntt.yaml

# dict_yaml maps tokens to integer IDs
# Format: token1: 0, token2: 1, etc.
dict_yaml=conf/dict/ntt_label2id.yaml

# Output directories
exp_dir=exp/ntt_infant         # Directory for all Kaldi models and alignments
output_test_dir=exp/out/test   # Output directory for test set force alignment results
output_all_dir=exp/out/all     # Output directory for all data force alignment results
mfccdir=mfcc                   # Directory for MFCC features

# Parse command line options
. utils/parse_options.sh || exit 1

# Source Kaldi configuration and path setup
# kaldi_conf.sh sets up KALDI_ROOT and SRILM paths
. ${script_dir}/kaldi_conf.sh
. cmd.sh     # Command execution configuration (run.pl, queue.pl, etc.)
. path.sh    # Kaldi path configuration

# Create experiment directory
mkdir -p $exp_dir

if [ $stage -le 0 ]; then
  echo "===== STAGE 0: Data Preparation ====="
  echo "Converting NTT data format to Kaldi format..."
  echo "Using division file: $division_yaml"
  echo "Using data info file: $data_info_json"

  # Create data directories for train, dev, test
  mkdir -p ${data_dir}/{train,dev,test}

  # Convert NTT data format to Kaldi format
  # This creates wav.scp, utt2spk, spk2utt, text, and segments files
  python3 ${script_dir}/data_prep.py \
    --data_info_json $data_info_json \
    --division_yaml $division_yaml \
    --output_dir $data_dir

  # Create dictionary from yaml file
  # This creates lexicon.txt, nonsilence_phones.txt, silence_phones.txt, etc.
  python3 ${script_dir}/dict_prep.py \
    --dict_yaml $dict_yaml \
    --output_dir ${data_dir}/local/dict

  # Prepare language model
  # This creates L.fst (lexicon FST) and other language model resources
  utils/prepare_lang.sh ${data_dir}/local/dict "<UNK>" ${data_dir}/local/lang ${data_dir}/lang || exit 1

  # Create simple LM for testing
  # Using SRILM to create a bigram language model
  # Install SRILM by downloading SRILM and update the path of SRILM in MakeFile
  echo "Creating simple LM for testing..."
  mkdir -p ${data_dir}/local/lm
  cat ${data_dir}/train/text | cut -d' ' -f2- | ngram-count -text - -order 2 \
    -lm ${data_dir}/local/lm/lm.arpa

  # Compress the ARPA file before formatting (required by format_lm.sh)
  gzip -f ${data_dir}/local/lm/lm.arpa

  # Format the compressed language model
  # This creates G.fst (grammar FST) and other decoding resources
  utils/format_lm.sh ${data_dir}/lang ${data_dir}/local/lm/lm.arpa.gz \
    ${data_dir}/local/dict/lexicon.txt ${data_dir}/lang_test
fi

if [ $stage -le 1 ]; then
  echo "===== STAGE 1: Feature Extraction ====="
  echo "Extracting MFCC features for train, dev, and test sets..."

  # Make MFCC features for each data partition
  for part in train dev test; do
    echo "Processing ${part} set..."
    # Differ from step/make_mfcc.sh  (extract-segments => set --min-segment-length=0.001)
    ${script_dir}/make_mfcc.sh --min_segment_length $min_segment_length --cmd "$train_cmd" --nj $nj ${data_dir}/$part exp/make_mfcc/$part $mfccdir || exit 1

    # Compute cepstral mean and variance normalization
    steps/compute_cmvn_stats.sh ${data_dir}/$part exp/make_mfcc/$part $mfccdir || exit 1

    # Fix data directory (ensure consistency between files)
    utils/fix_data_dir.sh ${data_dir}/$part || exit 1
  done
fi

if [ $stage -le 2 ] && $train; then
  echo "===== STAGE 2: Train Monophone Model ====="
  echo "Training monophone acoustic model..."

  # Train monophone models (context-independent phones)
  # boost-silence=1.25 increases the probability of silence
  steps/train_mono.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
    ${data_dir}/train ${data_dir}/lang $exp_dir/mono || exit 1

  # Align training data with monophone model
  # This creates alignments needed for triphone training
  echo "Aligning data with monophone model..."
  steps/align_si.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
    ${data_dir}/train ${data_dir}/lang $exp_dir/mono $exp_dir/mono_ali || exit 1
fi

if [ $stage -le 3 ] && $train; then
  echo "===== STAGE 3: Train Delta + Delta-Delta Triphone Model ====="
  echo "Training triphone model with delta+delta-delta features..."

  # Train tri1 (delta+delta-delta features, context-dependent phones)
  # 2000 = number of leaves in decision tree
  # 10000 = number of Gaussians in the GMM
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" \
    2000 10000 ${data_dir}/train ${data_dir}/lang $exp_dir/mono_ali $exp_dir/tri1 || exit 1

  # Align training data with triphone model
  echo "Aligning data with triphone delta model..."
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    ${data_dir}/train ${data_dir}/lang $exp_dir/tri1 $exp_dir/tri1_ali || exit 1
fi

if [ $stage -le 4 ] && $train; then
  echo "===== STAGE 4: Train LDA-MLLT Triphone Model ====="
  echo "Training triphone model with LDA-MLLT feature transformation..."

  # Train tri2 (LDA-MLLT - Linear Discriminant Analysis + Maximum Likelihood Linear Transform)
  # LDA-MLLT improves feature discriminability
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    2500 15000 ${data_dir}/train ${data_dir}/lang $exp_dir/tri1_ali $exp_dir/tri2 || exit 1

  # Align training data with LDA-MLLT model
  echo "Aligning data with LDA-MLLT model..."
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    ${data_dir}/train ${data_dir}/lang $exp_dir/tri2 $exp_dir/tri2_ali || exit 1
fi

if [ $stage -le 5 ] && $train; then
  echo "===== STAGE 5: Train SAT Triphone Model ====="
  echo "Training triphone model with Speaker Adaptive Training (SAT)..."

  # Train tri3 (SAT - Speaker Adaptive Training with fMLLR)
  # SAT normalizes for speaker variability during training
  steps/train_sat.sh --cmd "$train_cmd" \
    4200 40000 ${data_dir}/train ${data_dir}/lang $exp_dir/tri2_ali $exp_dir/tri3 || exit 1

  # Align training data with SAT model
  echo "Aligning data with SAT model..."
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    ${data_dir}/train ${data_dir}/lang $exp_dir/tri3 $exp_dir/tri3_ali || exit 1
fi

if [ $stage -le 6 ]; then
  echo "===== STAGE 6: Force Alignment ====="
  echo "Preparing graph for decoding..."

  # Prepare graph for decoding
  # This creates HCLG.fst (integrated decoding graph)
  utils/mkgraph.sh ${data_dir}/lang_test $exp_dir/tri3 $exp_dir/tri3/graph || exit 1

  # Force align test data using the SAT model
  echo "Force aligning test data..."
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    ${data_dir}/test ${data_dir}/lang $exp_dir/tri3 $exp_dir/tri3_ali_test || exit 1

  # Process all data if output_all_dir is specified
  if [ ! -z "$output_all_dir" ]; then
    echo "Processing all data for alignment..."

    # Combine train, dev, test data into a single dataset
    echo "Combining train, dev, and test data..."
    utils/combine_data.sh ${data_dir}/all \
      ${data_dir}/train ${data_dir}/dev ${data_dir}/test || exit 1

    # Make MFCC features for the combined dataset
    echo "Extracting features for all data..."
    # Differ from step/make_mfcc.sh  (extract-segments => set --min-segment-length=0.001)
    ${script_dir}/make_mfcc.sh --min_segment_length $min_segment_length --cmd "$train_cmd" --nj $nj \
      ${data_dir}/all exp/make_mfcc/all $mfccdir || exit 1
    steps/compute_cmvn_stats.sh ${data_dir}/all exp/make_mfcc/all $mfccdir || exit 1
    utils/fix_data_dir.sh ${data_dir}/all || exit 1

    # Force align all data using the SAT model
    echo "Force aligning all data..."
    steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
      ${data_dir}/all ${data_dir}/lang $exp_dir/tri3 $exp_dir/tri3_ali_all || exit 1
  fi
fi

if [ $stage -le 7 ]; then
  echo "===== STAGE 7: Convert Alignments ====="

  # Convert alignments to desired format for test set
  echo "Converting test alignments to output format..."
  mkdir -p $output_test_dir
  python3 ${script_dir}/convert_alignments.py \
    --ali_dir $exp_dir/tri3_ali_test \
    --data_dir ${data_dir}/test \
    --data_info_json $data_info_json \
    --output_dir $output_test_dir || exit 1

  echo "Test alignments written to $output_test_dir"
  echo "Fixed test alignments written to $output_test_dir/fixed"

  # Process all data if output_all_dir is specified
  if [ ! -z "$output_all_dir" ]; then
    # Convert alignments to desired format for all data
    echo "Converting all alignments to output format..."
    mkdir -p $output_all_dir
    python3 ${script_dir}/convert_alignments.py \
      --ali_dir $exp_dir/tri3_ali_all \
      --data_dir ${data_dir}/all \
      --data_info_json $data_info_json \
      --output_dir $output_all_dir || exit 1

    echo "All alignments written to $output_all_dir"
    echo "Fixed all alignments written to $output_all_dir/fixed"
  fi
fi

echo "Force alignment completed successfully!"
echo ""
echo "Output directories:"
echo "  - Raw alignments for test data: $output_test_dir"
echo "  - Fixed alignments for test data: $output_test_dir/fixed"
if [ ! -z "$output_all_dir" ]; then
  echo "  - Raw alignments for all data: $output_all_dir"
  echo "  - Fixed alignments for all data: $output_all_dir/fixed"
fi
exit 0

#!/bin/bash
# Implemented by bin-wu at 20:10 on 20220602

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -uo pipefail

output_wav_scp=

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
# https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/utils/parse_options.sh
. utils/parse_options.sh || exit 1

if [  $# -ne 3 ]; then
    echo "Use sox to split long recordings into segments of recordings."
    echo ""
    echo "Long recordings are at the 'wav.scp' file, each line with format of <recording_id> <recording_path>."
    echo "Segments of recordings are at the 'segments' file, each line with format of <utterance_id> <recording_id> <begin_sec> <end_sec>."
    echo "Output short segmented recordings named as <utterance_id>.wav."
    echo ""
    echo "Usage: "
    echo " $0 [options] <wav_scp> <segments> <output_dir>"
    echo "e.g.:"
    echo " $0 --output_wav_scp data/dataset/segmented_recordings.scp data/train/wav.scp data/train/segments data/dataset/segmented_recordings"
    echo " $0 --output_wav_scp data/all/tmp/wav_seg_test.scp data/all/tmp/wav_all_test.scp data/all/tmp/segments_all_test data/all/tmp"
    echo "Options:"
    echo "    --output_wav_scp=<wav_scp>  # the Kaldi 'wav.scp' file of segmented recordings"
    exit 1;
fi

wav_scp=$1
segments=$2
output_dir=$3

declare -A recid2path

# Read wav.scp
while IFS=" "  read -r recid path; do
    recid2path[$recid]=$path;
done < $wav_scp

# Cut the recordings according to segments
if [ ! -z $output_wav_scp ]; then cat /dev/null > $output_wav_scp; fi
while IFS=' ' read -r uttid recid begin_sec end_sec; do
    sox ${recid2path[$recid]} $output_dir/$uttid.wav trim $begin_sec =$end_sec
    if [ ! -z $output_wav_scp ]; then echo $uttid $output_dir/$uttid.wav >> $output_wav_scp; fi
done < $segments

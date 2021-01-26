#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail

stage=8

bjava=/project/nakamura-lab09/Share/Corpora/Speech/multi/Additional_OpenASR2020/javanese/IARPA_BABEL_OP3_402_LDC2020S07/package/IARPA_BABEL_OP3_402
sph2pipe=/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi/tools/sph2pipe_v2.5/sph2pipe

morethanNwords=2 # text of each utterance has more than N words (end point included).
# all the data between line number from train_begin to train_end is the training set (end points both included; indexing starting at one)
# be careful when you use two channels, please take twice the sentences number of audios
train_begin=2
train_end=20000
dev_begin=2
dev_end=3
test_begin=1
test_end=2
single_channel=true # one channel or two; phone conversation might have more than one channels.

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

if [ $stage -le 1 ]; then
    # Info: $bjava/scripted/reference_materials/demographics.tsv
    #       outputFn        sessID  date    time    spkrCode        lineType        dialect gen     envType age     network phoneModel      sampleCount     sampleRate
    #       BABEL_OP3_402_10319_20140928_155726_C1_scripted.sph     10319   20140928        155726  10319   inLine  CENTRAL F       CAR_KIT 31      TELKOMSEL       LG      0       8000
    #       BABEL_OP3_402_10319_20140928_160053_S7_scripted.sph     10319   20140928        160053  10319   inLine  CENTRAL F       CAR_KIT 31      TELKOMSEL       LG      0       8000
    # Audio: $bjava/scripted/training/audio/BABEL_OP3_402_10319_20140928_155726_C1_scripted.sph
    # Trans: $bjava/scripted/training/transcription/BABEL_OP3_402_10319_20140928_155726_C1_scripted.txt
    #
    # Preprocessing:
    # Remove the utterances with transcription containing <int>, <breath> and <no-speech>.
    # Make sure each utterance has two words or more.
    date
    echo "Preparing all data..."
    mkdir -p data/all
    rm -rf data/all/*

#    head -4 $bjava/scripted/reference_materials/demographics.tsv | sed "1d" | sort | while read record; do
    cat $bjava/scripted/reference_materials/demographics.tsv | sed "1d" | sort | while read record; do
	audio_name=$(echo "$record" | awk '{print $1}')
	speaker=$(echo "$record" | awk '{print $5}')
	gender=$(echo "$record" | awk '{print $8}')

	uttid=$(basename $audio_name .sph)
	trans_path=$bjava/scripted/training/transcription/${uttid}.txt
	aud_path=$bjava/scripted/training/audio/${uttid}.sph

	trans=$(cat $trans_path | grep -vE '\[|\]')
	num_fields=$(echo $trans | awk '{print NF}')

	echo "${uttid} ${aud_path}"  >> data/all/wav_all.scp
	# if [ $num_fields -ge ${morethanNwords} ]; then
	if [ $num_fields -ge ${morethanNwords} ] && (echo $trans | grep -vq "<"); then
	    if $single_channel;
	    then
		echo "${uttid} ${sph2pipe} -f wav -p -c 1 ${aud_path} |" >> data/all/wav.scp # We use one channel of phone conversation, it is alternative to use two
		echo "${uttid} ${trans}" >> data/all/text
		echo "${uttid} ${gender}" >> data/all/utt2gender
		echo "${uttid} ${speaker}" >> data/all/utt2spk
	    else
		# use two channels
		echo "${uttid}-A ${sph2pipe} -f wav -p -c 1 ${aud_path} |" >> data/all/wav.scp
		echo "${uttid}-A ${trans}" >> data/all/text
		echo "${uttid}-A ${gender}" >> data/all/utt2gender
		echo "${uttid}-A ${speaker}" >> data/all/utt2spk

		echo "${uttid}-B ${sph2pipe} -f wav -p -c 2 ${aud_path} |" >> data/all/wav.scp
		echo "${uttid}-B ${trans}" >> data/all/text
		echo "${uttid}-B ${gender}" >> data/all/utt2gender
		echo "${uttid}-B ${speaker}" >> data/all/utt2spk
	    fi
	fi
    done
fi

if [ $stage -le 2 ]; then
    date
    echo "Get the segment file by removing silences at beginning and end..."
    python local/get_segment.py --min_amplitude 0.1 --wav_scp data/all/wav.scp --segment data/all/segments --num_samples_gt_min 100
fi

if [ $stage -le 3 ]; then
    date
    echo "Split data into training, development and test data..."
    rm -rf data/{train,dev,test}
    mkdir -p data/{train,dev,test}

    # add segment later
    for file in data/all/{text,utt2gender,utt2spk,wav.scp,segments}; do
	# all the data between line number from train_begin to train_end is the training set (end points both included; indexing starting at one).
        cat $file | awk -v train_begin=$train_begin -v train_end=$train_end 'NR >= train_begin && NR <= train_end' > data/train/$(basename $file)
	cat $file | awk -v dev_begin=$dev_begin -v dev_end=$dev_end 'NR >= dev_begin && NR <= dev_end' > data/dev/$(basename $file)
	cat $file | awk -v test_begin=$test_begin -v test_end=$test_end 'NR >= test_begin && NR <= test_end' > data/test/$(basename $file)
    done

    for dataset in train dev test; do
	utils/utt2spk_to_spk2utt.pl data/${dataset}/utt2spk > data/${dataset}/spk2utt
    done
fi

#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -o pipefail # without -u here for conda setting

# Prepare some basic config files of kaldi.
./local/kaldi_conf.sh # CHECKME
# Note: cmd.sh, path.sh are created by kaldi_conf.sh
. cmd.sh
. path.sh

# general configuration
stage=8  # start from 0 if you need to start from data preparation

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

# the directory of timit
timit=/project/nakamura-lab01/Share/Corpora/Speech/en/TIMIT/TIMIT # CHECKME

if [ $stage -le 4 ]; then
    echo "Getting time information from original timit mannal annotation..."

    # Collect the phn files of timit
    # run command "cp /project/nakamura-lab01/Share/Corpora/Speech/en/TIMIT/TIMIT/TEST/DR1/FCJF0/SI1027.PHN data/test_phn/FCJF0_SI1027.PHN
    #              cp /project/nakamura-lab01/Share/Corpora/Speech/en/TIMIT/TIMIT/TEST/DR1/FCJF0/SA1.PHN data/test_phn/FCJF0_SA1.PHN ...."
    # Note: use only si & sx utterances (no sa utterance) of timit testing corpus, as done in kaldi standard timit corpus.
    mkdir -p data/test_time/test_phn
    find $timit/TEST/ -name *PHN -not \( -iname 'SA*' \) | sed "s:.*/\(.*\)/\(.*\).PHN:cp \0\tdata/test_time/test_phn/\1_\2.PHN:g" | sh
    awk '{print $1 "\t" $3}' conf/phones.60-48-39.map | sed 's:^q\t$:q\tsil:' > conf/phones.61-39.map

    # from timit format to abx <frame_time label> format 
    g++ ./local/c/timit2abx.cpp -o ./local/c/timit2abx # Updated timit2abx
    mkdir -p data/test_time/test_time_phn
    for file in data/test_time/test_phn/*; do
	./local/c/timit2abx $file ./conf/phones.61-39.map > data/test_time/test_time_phn/$(basename $file).abx; # with phone map
    done
    
    # Convert timit format to <begin_frame_time end_frame_time label> format
    mkdir -p data/test_time/test_dur_phn/
    g++ local/c/timit_normal.cpp -o local/c/timit_normal
    for file in data/test_time/test_phn/*; do
	./local/c/timit_normal $file ./conf/phones.61-39.map > data/test_time/test_dur_phn/$(basename $file)
    done

    # Create item_file for timit for abx test
    g++ local/c/timit_abx_item.cpp -o local/c/timit_abx_item
    echo "#file onset offset #phone context talker" > data/test_time/timit.item
    for file in data/test_time/test_dur_phn/*; do # normalize the time and label
	./local/c/timit_abx_item $file
    done | sort -k4,4 -k5,5 >> data/test_time/timit.item
fi

if [ $stage -le 5 ]; then
    echo "Getting time information according to mfcc39 frame time..."

    # Notice this is different from the original of merging time strategy (see commented code)
    # this one assume the end of file without annotation is sil.
    result_dir=data/test_time/test_time_phn_with_mfcc39_frame_time
    log_dir=data/test_time/logs
    rm -rf $result_dir
    mkdir -p  $log_dir $result_dir
    awk '{print $1}' data/test/feats.scp  | \
	while read -r uttid; do
	    python local/get_time_phn_from_abx_time.py \
		   --file_abx eval/abx/embedding/exp/feat/mfcc39/${uttid}.txt \
		   --file_dur_phn data/test_time/test_dur_phn/${uttid}.PHN \
		   --result_dir $result_dir
	done |& tee $log_dir/test_time_phn_with_mfcc39_frame_time.log

    # echo ============================================================================
    # echo "     Get abx time by merging annotated time and mfcc frame time           "
    # echo ============================================================================
	   
    # # Merge the phn label and cluster label of abx files according to the common time index
    # mkdir -p exp/dpgmm/baseline/data/merge_label
    # g++ ./local/c/merge_abx_with_log.cpp -o ./local/c/merge_abx_with_log

    # for file in $(ls data/test_time/test_time_phn/); do
    # 	./local/c/merge_abx_with_log \
    # 	    data/test_time/test_time_phn/$file dpgmm/test/data/mfcc.vtln.deltas/cluster_label/$(basename $file .PHN.abx).clabel.abx \
    # 	    exp/dpgmm/baseline/data > exp/dpgmm/baseline/data/merge_label/$(basename $file .PHN.abx).mlabel.abx
    # done

    # # Get the intersection of the annotation time and the frame time
    # mkdir -p data/test_time/test_abx_time
    # for file in exp/dpgmm/baseline/data/merge_label/*; do
    # 	awk '{print $1}' $file > data/test_time/test_abx_time/$(basename $file .mlabel.abx)
    # done

    # for file in $(ls data/test_time/test_abx_time/*); do
    # 	echo $(basename $file) $(wc -l $file | awk '{print $1}');
    # done > data/test/utt2num_frames_abx_time    
fi

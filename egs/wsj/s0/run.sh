#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail

# Prepare some basic config files of kaldi.
sh local/kaldi_conf.sh
# Note: cmd.sh, path.sh are created by kaldi_conf.sh
. cmd.sh
. path.sh

# general configuration
stage=3  # start from 0 if you need to start from data preparation
mfcc_config=conf/mfcc_hires.conf  # use the high-resolution mfcc for training the neurnal network;
                             # 40 dimensional mfcc feature used by Google and kaldi wsj network training.

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

# data
# wsj0=/project/nakamura-lab01/Share/Corpora/Speech/en/WSJ/wsj0
# wsj1=/project/nakamura-lab01/Share/Corpora/Speech/en/WSJ/wsj1
wsj=/project/nakamura-lab01/Share/Corpora/Speech/en/WSJ/

if [ $stage -le 0 ]; then
    date
    echo "Stage 0: Data Preparation"
    # local/wsj_data_prep.sh ${wsj0}/??-{?,??}.? ${wsj1}/??-{?,??}.?
    local/cstr_wsj_data_prep.sh $wsj || exit 1;
    local/wsj_format_data.sh || exit 1;
    date
fi

# Directory contains mfcc features and cmvn statistics.
mfcc_dir=mfcc
if [ $stage -le 1 ]; then
    date
    echo "Stage 1: Feature Extraction"
    for x in test_eval92 test_eval93 test_dev93 train_si284; do
	steps/make_mfcc.sh --cmd "$train_cmd" --nj 20 --mfcc-config $mfcc_config --write_utt2num_frames true \
			   data/$x exp/make_mfcc/$x $mfcc_dir/$x || exit 1  # 40-dim mfcc suggested by Dan used by Google
	steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfcc_dir/$x || exit 1; # cmvn per speaker suggested by Dan
	utils/fix_data_dir.sh data/$x || exit 1  # Fixing data format; remove segments with problems
    done

    utils/subset_data_dir.sh --first data/train_si284 7138 data/train_si84 || exit 1
    # Now make subset with the shortest 2k utterances from si-84.
    utils/subset_data_dir.sh --shortest data/train_si84 2000 data/train_si84_2kshort || exit 1;
    date
fi

if [ $stage -le 2 ]; then
    date
    echo "Stage 2: Dump Features after CMVN"
    for x in test_eval92 test_eval93 test_dev93 train_si284 train_si84 train_si84_2kshort; do
	cutils/make_cmvn.sh data/$x mfcc/$x
    done
    date
fi

train_set=train_si284
dict=data/lang_1char/${train_set}_units.txt
non_lang_syms=data/lang_1char/${train_set}_non_lang_syms.txt
if [ ${stage} -le 3 ]; then
    echo "Stage 3: Dictionary Preparation"
    # Task dependent. You have to check non-linguistic symbols used in the corpus.
    # These operations have priorities. First replacing string, then deleting characters and then replacing characters.
    [ -f conf/str_rep.txt ] || touch conf/str_rep.txt      # replace the special strings in original text
    [ -f conf/chars_del.txt ] || touch conf/chars_del.txt  # the characters (including non_lang_syms} to be deleted
    [ -f conf/chars_rep.txt ] || touch conf/chars_rep.txt  # the characters (including non_lang_syms} to be replaced

    echo "dictionary: ${dict}"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    cat data/${train_set}/text | \
	tr [A-Z] [a-z] | \
	python cutils/replace_str.py --rep_in=conf/str_rep.txt --sep='#' | \
	cut -f 2-  | tr " " "\n" | sort | uniq | grep "<" > ${non_lang_syms}
    cat ${non_lang_syms}

    echo "make a dictionary"
    # We follow the index convention of torchtext.
    echo "<unk> 0" > ${dict}
    echo "<pad> 1" >> ${dict}
    echo "<sos> 2" >> ${dict}
    echo "<eos> 3" >> ${dict}
    cat data/${train_set}/text | \
	tr [A-Z] [a-z] | \
	python cutils/replace_str.py --rep_in=conf/str_rep.txt --sep='#' | \
	cutils/text2token.py -s 1 -n 1 -l ${non_lang_syms} --chars-delete=conf/chars_del.txt --chars-replace=conf/chars_rep.txt | \
	tr [A-Z] [a-z] | \
	cut -f 2- -d" " | tr " " "\n" | sort | uniq | grep -v -e '^\s*$' | grep -v "<unk>" | awk '{print $0 " " NR + 3}' >> ${dict}
    wc -l ${dict}
fi

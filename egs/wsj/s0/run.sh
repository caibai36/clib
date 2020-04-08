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
mfcc_dir=mfcc # Directory contains mfcc features and cmvn statistics.
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
	local/scripts/make_cmvn.sh data/$x mfcc/$x
    done
    date
fi

train_set=train_si284
dict=data/lang_1char/${train_set}_units.txt
non_ling_syms=data/lang_1char/${train_set}_non_ling_syms.txt
if [ ${stage} -le 3 ]; then
    echo "Stage 3: Dictionary Preparation"

    echo "dictionary: ${dict}"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    # Note that the first column of text is the uttid.
    # assume that speakers not confused between upper and lower case; all non_ling_syms with <...> format
    local/scripts/replace_str.py --text_in data/${train_set}/text \
				--rep_in=conf/str_rep.txt \
				--sep='#' \
				--str2lower | \
	cut -f 2-  | tr " " "\n" | sort | uniq | grep "<" > ${non_ling_syms}
    cat ${non_ling_syms}

    echo "make a dictionary"
    echo -ne "<unk> 0\n<pad> 1\n<sos> 2\n<eos> 3\n" > ${dict} # index convention of torchtext
    # text2token.py converts every sentence of the text as a sequence of characters.
    local/scripts/text2token.py --text data/${train_set}/text \
			       --strs-replace-in=conf/str_rep.txt \
			       --strs-replace-sep='#' \
			       --chars-delete=conf/chars_del.txt \
			       --chars-replace=conf/chars_rep.txt \
			       --non-ling-syms=${non_ling_syms} \
			       --skip-ncols=1 \
			       --str2lower | \
	cut -f 2- -d" " | tr " " "\n" | sort | uniq | grep -v -e '^\s*$' | grep -vE "<unk>|<pad>|<sos>|<eos>" | awk '{print $0 " " NR + 3}' >> ${dict}
    wc -l ${dict}
fi

if [ ${stage} -le 4 ]; then
    echo "make json files"
    # get scp files (each line as 'uttid scp_content'), then merge them into utts.json.
    #     (eg. num_frames.scp, feat_dim.scp, num_tokens.scp, tokenid.scp, vocab_size.scp, feat.scp, token.scp and etc.)
    # If you want to add more information, just create more scp files in data2json.sh

    for x in test_eval92 test_eval93 test_dev93 train_si284 train_si84 train_si84_2kshort; do
	local/scripts/data2json.sh --feat data/$x/feats.scp \
    		     --non-ling-syms ${non_ling_syms} \
    	             --output-utts-json data/$x/utts.json \
    		     --output-dir-of-scps data/$x/scps \
    		     data/$x ${dict}
    done
fi

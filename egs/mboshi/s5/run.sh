###################################################################################
# File: run.sh
####################################################################################
#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -uo pipefail

# Prepare some basic config files of kaldi.
./local/kaldi_conf.sh # CHECKME
# Note: cmd.sh, path.sh are created by kaldi_conf.sh
. cmd.sh
. path.sh

# general configuration
stage=8  # start from 0 if you need to start from data preparation
mfcc_config=conf/mfcc.conf # 13 dimensional mfcc feature of kaldi default setting
# mel_config=clib/conf/feat/taco_mel_f80.json # 80 dimensional mel feature of tacotron tts

# options for training ASR
# Acoustic model parameters
numLeavesTri1=2500
numGaussTri1=15000
numLeavesMLLT=2500
numGaussMLLT=15000
numLeavesSAT=2500
numGaussSAT=15000
numGaussUBM=400
numLeavesSGMM=7000
numGaussSGMM=9000

# number of speakers 3 is less than the number of output .scp files
feats_nj=3
train_nj=3
decode_nj=3

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

# the directory of timit
mboshi=/project/nakamura-lab08/Work/bin-wu/share/data/mboshi/full_corpus_newsplit # CHECKME

# vad_file=data/segments/align_rmsil_vad.txt # created by mboshi_get_segment.py
# if [ ${stage} -le -1 ]; then
#     date
#     echo "Data preparation..."
#     # python local/mboshi_get_segment.py --corpus $mboshi --result data/segments --min_amplitude 0.1
#     # ./local/mboshi_get_wav_scp.sh --corpus $mboshi

#     # See the setting at https://arxiv.org/abs/1809.01431 'Pre-training on high-resource speech recognition improves low-resource speech-to-text translation'
#     # "Since this corpus does not include a designated test set, we randomly sampled and removed 200 utterances from training to use as a development set, and use the designated development data as a test set."
#     for dataset in train_raw_fr train_raw_mb train_fr train_mb dev_fr dev_mb; do # train_raw split into train and dev
# 	./local/mboshi_data_prep.sh --dataset $dataset --corpus $mboshi --dir train --vad_file $vad_file --lang ${dataset:(-2)} # $lang will fr or mb
#     done

#     for dataset in dev_raw_fr dev_raw_mb test_fr test_mb; do # dev_raw as test
# 	./local/mboshi_data_prep.sh --dataset $dataset --corpus $mboshi --dir dev --vad_file $vad_file --lang ${dataset:(-2)} # $lang will fr or mb
#     done
#     date
# fi

# if [ ${stage} -le -1 ]; then
#     date
#     echo "Making 39-dimensional mfcc feature..."

#     for dataset in train_raw_fr train_raw_mb dev_raw_fr dev_raw_mb train_fr train_mb dev_fr dev_mb test_fr test_mb; do
# 	# error: utils/validate_data_dir.sh: file data/train_raw_fr/utt2spk is not in sorted order or has duplicates
# 	# I commented out the utils/fix_data_dir.sh data/${dataset} at local/scripts/feat_extract.sh and local/scripts/make_mfcc.sh
# 	./local/scripts/feat_extract.sh --dataset ${dataset} --cmvn true --mfcc_conf $mfcc_config --delta_order 2 --feat_nj 3 # 39-dim mfcc 3 speakers
#     done
#     date
# fi

if [ ${stage} -le 1 ]; then
    echo ============================================================================
    echo "                Get information for speech corpora                     "
    echo ============================================================================

    mkdir -p data
    cp -r ../s0/data/{train_mb,dev_mb,test_mb} data
    for dataset in train_mb dev_mb test_mb; do
	cp data/$dataset/text data/$dataset/text.bak
	# Add sil at the beginning and the end of each line in text and sort the text. May be helpful in phone recognition, may not needed for word recognition.
	# orignal: sort $text | sed -r "s:(^\w+)\s(.*):\1 sil \2 sil:" > $test_data/text; Current version has no model of sil. (although we'll add dummy sil to dict later)
	# Convert the word representation to phoneme representation.
	# similar to TIMIT, small dataset trained by phoneme might be more robust; For small corpus, dictionary contructed from the training set may miss many words at test set(?)
	# one character means one phoneme(?): see https://arxiv.org/pdf/1710.03501.pdf
	# reomve the blanks at the end of each sentence or it will cause errors in language model training.
	# e.g. Dico11_113 mon ami m&apos => Dico11_113 m o n a m i m a p o s
	cat data/$dataset/text.bak | python local/scripts/text2token.py --skip-ncols 1 --chars-del conf/chars_del.txt | sed 's/<space>//g' | tr -s ' ' | sed -r "s/ +$//g" > data/$dataset/text
    done
fi

if [ ${stage} -le 1 ]; then
    echo ============================================================================
    echo "            Create lexicon and convert it to openfst format               "
    echo ============================================================================

    # Prepare raw dictionary source files,
    # including phone.txt, lexicon.txt, silence_phones.txt, optional_silence.txt, nonsilence_phones.txt and extra_questions.txt.
    dict_src=data/local/dict
    train_data=data/train_mb
    lang_tmp=data/local/lang_tmp
    lang=data/lang
    
    mkdir -p $dict_src
    # at least sil, or it will cause error in validating the data dir.
    echo sil > $dict_src/silence_phones.txt
    echo sil > $dict_src/optional_silence.txt
    cat $train_data/text | cut -d' ' -f2- | tr ' ' '\n' | sort -u > $dict_src/phones.txt
    echo sil >> $dict_src/phones.txt
    paste $dict_src/phones.txt $dict_src/phones.txt > $dict_src/lexicon.txt || exit 1;
    grep -v -F -f $dict_src/silence_phones.txt $dict_src/phones.txt > $dict_src/nonsilence_phones.txt
    touch $dict_src/extra_questions.txt

    # Convert the dictionary to the openfst format for kaldi.
    utils/prepare_lang.sh --sil-prob 0.0 --position-dependent-phones false --num-sil-states 3 $dict_src "sil" $lang_tmp $lang
    # utils/validate_data_dir.sh $train_data || exit 1 # utt2spk not sorted
fi

if [ ${stage} -le 1 ]; then
    echo ============================================================================
    echo "Create the phone bigram LM and coverted it to openfst format for decoding "
    echo ============================================================================

    # Create lanuage model from test text file and convert to fst format along with lexicon files
    train_data=data/train_mb
    lang=data/lang # including openfst format files of dict
    lm_src=data/local/nist_lm
    lm=data/lang_test_bg

    # Check irstlm
    if [ -z $IRSTLM ] ; then
	export IRSTLM=$KALDI_ROOT/tools/irstlm/
    fi

    export PATH=${PATH}:$IRSTLM/bin
    if ! command -v prune-lm >/dev/null 2>&1 ; then
	echo "$0: Error: the IRSTLM is not available or compiled" >&2
	echo "$0: Error: We used to install it by default, but." >&2
	echo "$0: Error: this is no longer the case." >&2
	echo "$0: Error: To install it, go to $KALDI_ROOT/tools" >&2
	echo "$0: Error: and run extras/install_irstlm.sh" >&2
	exit 1
    fi

    # Prepare the language mode files.
    mkdir -p $lm_src
    cut -d' ' -f2- $train_data/text | sed -e 's:^:<s> :' -e 's:$: </s>:' > $lm_src/lm_train.text
    build-lm.sh -i $lm_src/lm_train.text -n 2  -o $lm_src/lm_phone_bg.ilm.gz

    compile-lm $lm_src/lm_phone_bg.ilm.gz -t=yes /dev/stdout | \
	grep -v unk | gzip -c > $lm_src/lm_phone_bg.arpa.gz

    # Convert the the language model to fst format
    mkdir -p $lm
    cp -r $lang/* $lm

    gunzip -c $lm_src/lm_phone_bg.arpa.gz | arpa2fst --disambig-symbol=#0 --read-symbol-table=$lm/words.txt - $lm/G.fst
    fstisstochastic $lm/G.fst

    utils/validate_lang.pl $lm || exit 1
fi

if [ ${stage} -le 2 ]; then
    echo ============================================================================
    echo "              MFCC Feature Extration & CMVN of data set                   "
    echo ============================================================================
    mfccdir=feat/mfcc
    mkdir -p $mfccdir

    for dataset in train_mb dev_mb test_mb; do
	rm -rf data/$dataset/{feats.scp,raw.scp,cmvn.scp}
	./local/scripts/feat_extract.sh --dataset ${dataset} --cmvn true --mfcc_conf $mfcc_config --delta_order 0 --feat_nj 3 # 13-dim mfcc for timit script with 3 speakers
	utils/fix_data_dir.sh data/${dataset} || exit 1
	utils/validate_data_dir.sh data/${dataset} || exit 1
    done

#   
fi
	
if [ ${stage} -le 3 ]; then
    echo ============================================================================
    echo "                     MonoPhone Training & Decoding                        "
    echo ============================================================================

    steps/train_mono.sh  --nj "$train_nj" --cmd "$train_cmd" data/train_mb data/lang exp/mono

    utils/mkgraph.sh data/lang_test_bg exp/mono exp/mono/graph

    # --skip-scoring true enables iterative decoding without local/score.sh
    steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" --skip-scoring true \
		    exp/mono/graph data/dev_mb exp/mono/decode_dev

    steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" --skip-scoring true \
		    exp/mono/graph data/test_mb exp/mono/decode_test

fi

if [ ${stage} -le 4 ]; then
    echo ============================================================================
    echo "           tri1 : Deltas + Delta-Deltas Training & Decoding               "
    echo ============================================================================

    steps/align_si.sh --boost-silence 1.25 --nj "$train_nj" --cmd "$train_cmd" \
		      data/train_mb data/lang exp/mono exp/mono_ali

    # Train tri1, which is deltas + delta-deltas, on train data.
    steps/train_deltas.sh --cmd "$train_cmd" \
			  $numLeavesTri1 $numGaussTri1 data/train_mb data/lang exp/mono_ali exp/tri1

    utils/mkgraph.sh data/lang_test_bg exp/tri1 exp/tri1/graph

    steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" --skip-scoring true \
		    exp/tri1/graph data/dev_mb exp/tri1/decode_dev

    steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" --skip-scoring true \
		    exp/tri1/graph data/test_mb exp/tri1/decode_test
fi

if [ ${stage} -le 5 ]; then
    echo ============================================================================
    echo "                 tri2 : LDA + MLLT Training & Decoding                    "
    echo ============================================================================

    steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" \
		      data/train_mb data/lang exp/tri1 exp/tri1_ali

    steps/train_lda_mllt.sh --cmd "$train_cmd" \
			    --splice-opts "--left-context=3 --right-context=3" \
			    $numLeavesMLLT $numGaussMLLT data/train_mb data/lang exp/tri1_ali exp/tri2

    utils/mkgraph.sh data/lang_test_bg exp/tri2 exp/tri2/graph

    steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" --skip-scoring true \
		    exp/tri2/graph data/dev_mb exp/tri2/decode_dev

    steps/decode.sh --nj "$decode_nj" --cmd "$decode_cmd" --skip-scoring true \
		    exp/tri2/graph data/test_mb exp/tri2/decode_test
fi

if [ ${stage} -le 6 ]; then
    echo ============================================================================
    echo "              tri3 : LDA + MLLT + SAT Training & Decoding                 "
    echo ============================================================================

    # Align tri2 system with train data.
    steps/align_si.sh --nj "$train_nj" --cmd "$train_cmd" \
		      --use-graphs true data/train_mb data/lang exp/tri2 exp/tri2_ali

    # From tri2 system, train tri3 which is LDA + MLLT + SAT.
    steps/train_sat.sh --cmd "$train_cmd" \
		       $numLeavesSAT $numGaussSAT data/train_mb data/lang exp/tri2_ali exp/tri3

    utils/mkgraph.sh data/lang_test_bg exp/tri3 exp/tri3/graph

    steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" --skip-scoring true \
			  exp/tri3/graph data/dev_mb exp/tri3/decode_dev

    steps/decode_fmllr.sh --nj "$decode_nj" --cmd "$decode_cmd" --skip-scoring true \
			  exp/tri3/graph data/test_mb exp/tri3/decode_test

    echo ============================================================================
    echo "                        SGMM2 Training & Decoding                         "
    echo ============================================================================

    steps/align_fmllr.sh --nj "$train_nj" --cmd "$train_cmd" \
			 data/train_mb data/lang exp/tri3 exp/tri3_ali
fi

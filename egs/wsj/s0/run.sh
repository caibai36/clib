#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -uo pipefail

# Prepare some basic config files of kaldi.
sh local/kaldi_conf.sh
# Note: cmd.sh, path.sh are created by kaldi_conf.sh
. cmd.sh
. path.sh

# general configuration
stage=8  # start from 0 if you need to start from data preparation
mfcc_dir=mfcc # Directory contains mfcc features and cmvn statistics.
mfcc_config=conf/mfcc_hires.conf  # use the high-resolution mfcc for training the neurnal network;
                             # 40 dimensional mfcc feature used by Google and kaldi wsj network training.

# Data and model options
run=run0
feat=mfcc40
dataset_name=wsj
# data_name=wsj0  # train:si84;dev:dev93;test:eval92
# data_name=wsj1  # train:si284;dev:dev93;test:eval92
model_name=EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp
exp_dir=exp/tmp

# options for training ASR
gpu=auto
batch_size=32
cutoff=1600 # cut off long sentences
label_smoothing=0.05
lr=0.001
num_epochs=70
grad_clip=5
factor=0.5 # for lr scheduler
patience=3 # for lr scheduler
save_interval=1 # save the model every x epoch

# options for evaluating ASR
set_uttid=None # subset of testing data (e.g. set_uttid=conf/data/test_small/set_uttid.txt)

search=beam
max_target=250 # the maximum length of the decoded sequence
beam_size=10

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

    # Use './local/scripts/analyze_marks.sh --text data/train/text' to analyze the text
    # and then manually set str_rep.txt, char_del.txt and chars_rep.txt
    # which are later processed by text2token.py in the order of replacing strings,
    # deleting characters and replacing characters. By default, these files are empty.
    if [ ! -f conf/str_rep.txt ]; then touch conf/str_rep.txt; fi
    if [ ! -f conf/chars_del.txt ]; then touch conf/chars_del.txt; fi
    if [ ! -f conf/chars_rep.txt ]; then touch conf/chars_rep.txt; fi

    echo "dictionary: ${dict}"
    mkdir -p data/lang_1char/

    echo "make a non-linguistic symbol list"
    # Task dependent.
    # Assume that the non-linguistic symbols of this text are in the format <symbol>.
    # Note that the first column of text is the uttid.
    local/scripts/replace_str.py --text_in data/${train_set}/text \
				--rep_in=conf/str_rep.txt \
				--sep='#' \
				--str2lower | \
	cut -f 2- -d" " | tr " " "\n" | sort | uniq | grep "<" > ${non_ling_syms}
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

# # run one by one
# if [ ${stage} -le 5 ]; then
#     ./local/train_asr_wsj0.sh # train:si84;dev:dev93;test:eval92
#     ./local/eval_asr_wsj0.sh
#     ./local/train_asr_wsj1.sh # train:si284;dev:dev93;test:eval92
#     ./local/eval_asr_wsj1.sh
# fi

if [ ${stage} -le 6 ]; then
    echo "Training ASR..."
    # data_name=wsj0  # train:si84;dev:dev93;test:eval92
    # data_name=wsj1  # train:si284;dev:dev93;test:eval92

    for data_name in wsj0 wsj1; do
	data_config=conf/data/${dataset_name}/${data_name}/asr_tts/data_${feat}.yaml
	model_config=conf/model/asr/seq2seq/${model_name}.yaml

	reducelr={\"factor\":$factor,\"patience\":$patience}

	exp_setting=${feat}_batchsize${batch_size}_cutoff${cutoff}_labelsmoothing${label_smoothing}_lr${lr}_gradclip${grad_clip}_factor${factor}_patience${patience}
	result_dir=${exp_dir}/${dataset_name}/${data_name}/${model_name}-${run}/${exp_setting}/train

	# comment out the --exit option if you are not sure how many epochs to run
	# comment out the --overwrite option when you do not want to overwrite the previous runs
	python local/scripts/train_asr.py \
	       --gpu $gpu \
	       --data_config $data_config \
	       --batch_size $batch_size \
	       --cutoff $cutoff \
	       --model_config $model_config \
	       --label_smoothing $label_smoothing \
	       --lr $lr \
	       --reducelr $reducelr \
	       --num_epochs $num_epochs \
	       --grad_clip $grad_clip \
	       --result $result_dir \
	       --save_interval $save_interval \
	       --exit \
	       --overwrite
    done
fi

if [ ${stage} -le 7 ]; then
    echo "Evaluating ASR..."
    # data_name=wsj0  # train:si84;dev:dev93;test:eval92
    # data_name=wsj1  # train:si284;dev:dev93;test:eval92

    for data_name in wsj0 wsj1; do
	exp_setting=${feat}_batchsize${batch_size}_cutoff${cutoff}_labelsmoothing${label_smoothing}_lr${lr}_gradclip${grad_clip}_factor${factor}_patience${patience}
	model_path=${exp_dir}/${dataset_name}/${data_name}/${model_name}-${run}/${exp_setting}/train/best_model.mdl

	data_config=conf/data/${dataset_name}/${data_name}/asr_tts/data_${feat}.yaml
	result_dir=${model_path%/train/*}/eval/beamsize${beam_size} # ${string%substring} # Deletes shortest match of $substring from back of $string.

	python local/scripts/eval_asr.py \
	       --gpu $gpu \
	       --data_config $data_config \
	       --set_uttid $set_uttid \
	       --batch_size 2 \
	       --model $model_path \
	       --max_target $max_target \
	       --search $search \
	       --beam_size $beam_size \
	       --result $result_dir

	echo
	echo "Computing character error rate (CER)..."
	COMPUTE_WER=/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi/src/bin/compute-wer
	$COMPUTE_WER --mode=present ark,t:${result_dir}/ref_char.txt ark,t:${result_dir}/hypo_char.txt |& tee ${result_dir}/cer.txt
	echo
	echo "Computing word error rate (WER)..."
	$COMPUTE_WER --mode=present ark,t:${result_dir}/ref_word.txt ark,t:${result_dir}/hypo_word.txt |& tee ${result_dir}/wer.txt
    done
fi

######################################################################################################
# Additional scripts
if [ ${stage} -le 0 ]; then
    date
    echo "Making 80-dimensional mfcc feature"
    for dataset in test_eval92 train_si284 train_si84 test_dev93; do
	x=${dataset}_mfcc80
	cp -rf data/${dataset} data/${x}
	./local/scripts/feat_extract.sh --dataset ${x} --cmvn true --mfcc_conf conf/mfcc_hires80.conf
	local/scripts/data2json.sh --feat data/${x}/feats.scp \
				   --non-ling-syms ${non_ling_syms} \
				   --output-utts-json data/${x}/utts.json \
				   data/${x} ${dict}
    done
    date
fi

if [ ${stage} -le 0 ]; then
    date
    echo "Making 80-dimensional mel feature (tacotron style)"
    for dataset in test_eval92 train_si284 train_si84 test_dev93; do
	x=${dataset}_mel80
	cp -rf data/${dataset} data/${x}
	./local/scripts/feat_extract_taco.sh --dataset ${x} --cmvn true --mel_conf conf/feat/taco_mel_f80.json
	local/scripts/data2json.sh --feat data/${x}/feats.scp \
				   --non-ling-syms ${non_ling_syms} \
				   --output-utts-json data/${x}/utts.json \
				   data/${x} ${dict}
    done
    date
fi

if [ ${stage} -le 0 ]; then
    date
    echo "Making 80-dimensional mel feature without cmvn (tacotron style)"
    for dataset in test_eval92 train_si284 train_si84 test_dev93; do
	x=${dataset}_mel80_raw
	cp -rf data/${dataset} data/${x}
	./local/scripts/feat_extract_taco.sh --dataset ${x} --cmvn false --mel_conf conf/feat/taco_mel_f80.json
	local/scripts/data2json.sh --feat data/${x}/feats.scp \
				   --non-ling-syms ${non_ling_syms} \
				   --output-utts-json data/${x}/utts.json \
				   data/${x} ${dict}
    done
    date
fi

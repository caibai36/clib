#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail

# Prepare some basic config files of kaldi.
./local/kaldi_conf.sh # CHECKME
# Note: cmd.sh, path.sh are created by kaldi_conf.sh
. cmd.sh
. path.sh

# general configuration
stage=16  # start from 0 if you need to start from data preparation

mfcc_config=conf/mfcc.conf # 13 dimensional mfcc feature of kaldi default setting for timit
mfcc_hires=conf/mfcc_hires.conf # 40 dimensional mfcc feature of kaldi default setting for wsj
mel_config=clib/conf/feat/taco_mel_f80.json # 80 dimensional mel feature of tacotron tts

# Data and model options
run=run0
feat=mfcc39 # or mel80
dataset_name=csj
# data_name=wsj0  # train:si84;dev:dev93;test:eval92
# data_name=wsj1  # train:si284;dev:dev93;test:eval92
model_name=EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp
exp_dir=exp/asr_extend_dict

# extend dict
# Extend the dict whose vocab do not exist in the csj training set
# but possibly exist in corpora that use csj as pretrained model)
# Note the vocab of extended dict should not have repetition.
extended_units=conf/extended_dict/extended_units.txt
extended_non_ling_syms=conf/extended_dict/extended_non_ling_syms.txt

# options for training ASR
gpu=auto
batch_size=32
cutoff=1800 # cut off long sentences
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
csj_data=/project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/csj/s5/data # prepared data by running official scripts
datasets="train train_dev train_nodev eval1 eval2 eval3"

if [ ${stage} -le 1 ]; then
    echo "Data preparation..."
    ./local/csj_data_prep.sh --csj_data "$csj_data" --datasets "$datasets" --stage 1
fi

if [ ${stage} -le 2 ]; then
    date
    echo "Making 39-dimensional mfcc feature..."

    for dataset in $datasets; do
	./local/scripts/feat_extract.sh --dataset ${dataset} --cmvn true --mfcc_conf $mfcc_config --delta_order 2 # 39-dim mfcc
    done
    date
fi

if [ ${stage} -le 2 ]; then
    date
    echo "Making 40-dimensional mfcc feature..."

    for dataset in $datasets; do
    	x=${dataset}_mfcc40
	mkdir -p data/$x; cp -r data/${dataset}/* data/${x} # Fail to overwrite with command `cp -r data/${dataset} data/${x}'
	./local/scripts/feat_extract.sh --dataset ${x} --cmvn true --mfcc_conf $mfcc_hires # 40-dim mfcc
    done
    date
fi

if [ ${stage} -le 2 ]; then
    date
    echo "Making 80-dimensional mel feature..." # 10 hours for datasets="train train_dev train_nodev eval1 eval2 eval3"

    for dataset in $datasets; do
    	x=${dataset}_mel80
	mkdir -p data/$x; cp -r data/${dataset}/* data/${x} # Fail to overwrite with command `cp -r data/${dataset} data/${x}'
    	./local/scripts/feat_extract_taco.sh --dataset ${x} --cmvn true --mel_conf $mel_config  # 80-dim mel
    done
    date
fi

# Start from stage 3 by 'cp -r ../s0/data/ .'
train_set="train" # use all to create dict to avoid oov problem of dict
dict=data/lang_1char/${train_set}_units.txt
non_ling_syms=data/lang_1char/${train_set}_non_ling_syms.txt
if [ ${stage} -le 3 ]; then
    echo "Dictionary preparation..."

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
    # # Task dependent.
    # # Assume that the non-linguistic symbols of this text are in the format <symbol>.
    # # Note that the first column of text is the uttid.
    # local/scripts/replace_str.py --text_in data/${train_set}/text \
    # 				--rep_in=conf/str_rep.txt \
    # 				--sep='#' \
    # 				--str2lower | \
    # 	cut -f 2- -d" " | tr " " "\n" | sort | uniq | grep "<" > ${non_ling_syms}
    # cat ${non_ling_syms}
    if [ ! -f ${non_ling_syms} ]; then touch ${non_ling_syms}; fi

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

    echo ""
    if [ ! -z $extended_non_ling_syms ]; then
	echo "Merging current and extended non-linguistic symbol lists..."
	cat $non_ling_syms $extended_non_ling_syms | sort -u > /tmp/tmp
	mv /tmp/tmp $non_ling_syms
	cat $non_ling_syms
    fi

    if [ ! -z $extended_units ]; then
	echo "Merging current and extended dict units (with new vocab from the extended dict)..."
	vocabsize=`tail -n 1 ${dict} | awk '{print $2}'`
	vocab_index=$vocabsize;
	cp $dict /tmp/tmp
	for new_vocab in $(comm -23 <(cat $extended_units | awk '{print $1}' | sort)  <(cat $dict | awk '{print $1}' | sort)); do
	    let vocab_index=$vocab_index+1;
	    echo $new_vocab $vocab_index |& tee -a /tmp/tmp
	done
	mv /tmp/tmp $dict

	echo "dictionary: ${dict}"
	wc -l $dict
    fi
fi

if [ ${stage} -le 4 ]; then
    echo "Making json files"
    # get scp files (each line as 'uttid scp_content'), then merge them into utts.json.
    #     (eg. num_frames.scp, feat_dim.scp, num_tokens.scp, tokenid.scp, vocab_size.scp, feat.scp, token.scp and etc.)
    # If you want to add more information, just create more scp files in data2json.sh
    for dataset in $datasets; do
	x=${dataset}
	local/scripts/data2json.sh --feat data/$x/feats.scp \
    				   --non-ling-syms ${non_ling_syms} \
    				   --output-utts-json data/$x/utts.json \
    				   --output-dir-of-scps data/$x/scps \
    				   data/$x ${dict}
    done
fi

if [ ${stage} -le 4 ]; then
    echo "Making json files"
    # get scp files (each line as 'uttid scp_content'), then merge them into utts.json.
    #     (eg. num_frames.scp, feat_dim.scp, num_tokens.scp, tokenid.scp, vocab_size.scp, feat.scp, token.scp and etc.)
    # If you want to add more information, just create more scp files in data2json.sh
    for dataset in $datasets; do
	x=${dataset}_mfcc40
	local/scripts/data2json.sh --feat data/$x/feats.scp \
    				   --non-ling-syms ${non_ling_syms} \
    				   --output-utts-json data/$x/utts.json \
    				   --output-dir-of-scps data/$x/scps \
    				   data/$x ${dict}
    done
fi

if [ ${stage} -le 4 ]; then
    echo "Making json files"
    # get scp files (each line as 'uttid scp_content'), then merge them into utts.json.
    #     (eg. num_frames.scp, feat_dim.scp, num_tokens.scp, tokenid.scp, vocab_size.scp, feat.scp, token.scp and etc.)
    # If you want to add more information, just create more scp files in data2json.sh
    for dataset in $datasets; do
	x=${dataset}_mel80
	local/scripts/data2json.sh --feat data/$x/feats.scp \
    				   --non-ling-syms ${non_ling_syms} \
    				   --output-utts-json data/$x/utts.json \
    				   --output-dir-of-scps data/$x/scps \
    				   data/$x ${dict}

    done
fi

if [ ${stage} -le 5 ]; then
    echo "Updating data configuration file"
    for file in $(find clib/conf/data/csj/ -name '*yaml'); do echo $file; sed -i 's:csj/s0/:csj/s0.char/:g' $file; done
fi

if [ ${stage} -le 6 ]; then
    echo "Training ASR..."

    for data_name in default; do
	data_config=clib/conf/data/${dataset_name}/${data_name}/asr_tts/data_${feat}.yaml # CHECKME (you can change the data_config to the setting of your prepared dataset)
	model_config=clib/conf/model/asr/seq2seq/${model_name}.yaml

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
    echo "Evaluating ASR for the best epoch..."

    for data_name in default eval1 eval2 eval3; do
	exp_setting=${feat}_batchsize${batch_size}_cutoff${cutoff}_labelsmoothing${label_smoothing}_lr${lr}_gradclip${grad_clip}_factor${factor}_patience${patience}
	# model_path=${exp_dir}/${dataset_name}/${data_name}/${model_name}-${run}/${exp_setting}/train/best_model.mdl
	model_path=${exp_dir}/${dataset_name}/default/${model_name}-${run}/${exp_setting}/train/best_model.mdl

	data_config=clib/conf/data/${dataset_name}/${data_name}/asr_tts/data_${feat}.yaml
	# result_dir=${model_path%/train/*}/eval/beamsize${beam_size} # ${string%substring} # Deletes shortest match of $substring from back of $string.
	result_dir=${exp_dir}/${dataset_name}/${data_name}/${model_name}-${run}/${exp_setting}/eval/beamsize${beam_size} # ${string%substring}

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
	echo "Computing word error rate (WER)..." # Note timit performance should be WER
	$COMPUTE_WER --mode=present ark,t:${result_dir}/ref_word.txt ark,t:${result_dir}/hypo_word.txt |& tee ${result_dir}/wer.txt
    done
fi

if [ ${stage} -le 8 ]; then
    echo "Evaluating ASR for the last epoch..."

    for data_name in default eval1 eval2 eval3; do
	exp_setting=${feat}_batchsize${batch_size}_cutoff${cutoff}_labelsmoothing${label_smoothing}_lr${lr}_gradclip${grad_clip}_factor${factor}_patience${patience}
	# model_path=${exp_dir}/${dataset_name}/${data_name}/${model_name}-${run}/${exp_setting}/train/best_model.mdl
	# model_path=${exp_dir}/${dataset_name}/${data_name}/${model_name}-${run}/${exp_setting}/train/last_model.mdl
	model_path=${exp_dir}/${dataset_name}/default/${model_name}-${run}/${exp_setting}/train/last_model.mdl

	data_config=clib/conf/data/${dataset_name}/${data_name}/asr_tts/data_${feat}.yaml
	# result_dir=${model_path%/train/*}/eval/beamsize${beam_size} # ${string%substring} # Deletes shortest match of $substring from back of $string.
	# result_dir=${model_path%/train/*}/eval_last_epoch/beamsize${beam_size} # ${string%substring} # Deletes shortest match of $substring from back of $string.
	result_dir=${exp_dir}/${dataset_name}/${data_name}/${model_name}-${run}/${exp_setting}/eval_last_epoch/beamsize${beam_size}

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
	echo "Computing word error rate (WER)..." # Note timit performance should be WER
	$COMPUTE_WER --mode=present ark,t:${result_dir}/ref_word.txt ark,t:${result_dir}/hypo_word.txt |& tee ${result_dir}/wer.txt
    done
fi

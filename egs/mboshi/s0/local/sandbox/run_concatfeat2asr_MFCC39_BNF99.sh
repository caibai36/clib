#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -uo pipefail

# Prepare some basic config files of kaldi.
# ./local/kaldi_conf.sh # CHECKME
# Note: cmd.sh, path.sh are created by kaldi_conf.sh
. cmd.sh
. path.sh

# general configuration
stage=8  # start from 0 if you need to start from data preparation

feat1_name=MFCC39
feat2_name=BNF99

feat_train=data/train_mb/feats.scp
feat_dev=data/dev_mb/feats.scp
feat_test=data/test_mb/feats.scp

file=param_bnf;if [[ -L "$file" && -d "$file" ]]; then rm -rf $file; fi # $file is a symlink to a directory; safe to remove
ln -s /project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/mboshi/s5/param_bnf/ param_bnf # a feature stored as param_bnf/raw_bnfeat_train_si284.1.ark
feat_train2=/project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/mboshi/s5/data_bnf/train_bnf/feats.scp
feat_dev2=/project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/mboshi/s5/data_bnf/dev_bnf/feats.scp
feat_test2=/project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/mboshi/s5/data_bnf/test_bnf/feats.scp

feat=concat_${feat1_name}_${feat2_name}
root=exp/concatfeat2asr/asr_mboshi_${feat}

# asr data and model options
run=run0
# feat=mfcc39 # or mel80
dataset_name=mboshi
# data_name=default
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

data_dir=$root/data
if [ $stage -le 1 ]; then
    mkdir -p exp/logs

    # backup feats.scp
    if [ -f $root/data/train/feats.scp ]; then cp $root/data/train/feats.scp $root/train_feats.scp; fi
    if [ -f $root/data/dev/feats.scp ]; then cp $root/data/dev/feats.scp $root/dev_feats.scp; fi
    if [ -f $root/data/test/feats.scp ]; then cp $root/data/test/feats.scp $root/test_feats.scp; fi

    rm -rf $data_dir
    mkdir -p $data_dir
    cp -r data/train_mb $root/data/train
    cp -r data/dev_mb $root/data/dev
    cp -r data/test_mb $root/data/test

    date
    python local/concatfeat2asrdata.py \
	   --feat_train=$feat_train \
	   --feat_dev=$feat_dev \
	   --feat_test=$feat_test \
	   --feat_train2=$feat_train2 \
	   --feat_dev2=$feat_dev2 \
	   --feat_test2=$feat_test2 \
	   --result=$data_dir |& tee exp/logs/run_$(basename ${root}).log
    date
    head -2 ${root}/data/*/feats.scp
fi

train_set=train_mb
dict=data/lang_1char/${train_set}_units.txt
non_ling_syms=data/lang_1char/${train_set}_non_ling_syms.txt
# if [ ${stage} -le 2 ]; then
if [ ${stage} -le -1 ]; then # already done
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
    echo "Making json files"
    # get scp files (each line as 'uttid scp_content'), then merge them into utts.json.
    #     (eg. num_frames.scp, feat_dim.scp, num_tokens.scp, tokenid.scp, vocab_size.scp, feat.scp, token.scp and etc.)
    # If you want to add more information, just create more scp files in data2json.sh
    for dataset in train dev test; do
	x=${dataset}
	local/scripts/data2json.sh --feat ${root}/data/$x/feats.scp \
    				   --non-ling-syms ${non_ling_syms} \
    				   --output-utts-json ${root}/data/$x/utts.json \
    				   --output-dir-of-scps ${root}/data/$x/scps \
    				   ${root}/data/$x ${dict}
    done
fi

if [ $stage -le 5 ]; then
    mkdir -p $root/conf
    if [ -f $root/data/train/utts.json ]; then echo train: \'${PWD}/${root}/data/train/utts.json\'; fi > $root/conf/data_${feat}.yaml
    if [ -f $root/data/dev/utts.json ]; then echo dev: \'${PWD}/${root}/data/dev/utts.json\'; fi >> $root/conf/data_${feat}.yaml
    if [ -f $root/data/test/utts.json ]; then echo test: \'${PWD}/${root}/data/test/utts.json\'; fi >> $root/conf/data_${feat}.yaml
    echo "token2id: '/project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/mboshi/s0/data/lang_1char/train_mb_units.txt'" >> $root/conf/data_${feat}.yaml
fi

if [ ${stage} -le 6 ]; then
    echo "Training ASR..."

    for data_name in default; do
	# exp/dpgmm2asr/asr_dpgmm_seed123_K39/conf/data_dpgmm_seed123_K39.yaml
	data_config=$root/conf/data_${feat}.yaml # CHECKME (you can change the data_config to the setting of your prepared dataset)
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
    echo "Evaluating ASR..."

    for data_name in default; do
	exp_setting=${feat}_batchsize${batch_size}_cutoff${cutoff}_labelsmoothing${label_smoothing}_lr${lr}_gradclip${grad_clip}_factor${factor}_patience${patience}
	model_path=${exp_dir}/${dataset_name}/${data_name}/${model_name}-${run}/${exp_setting}/train/best_model.mdl

	data_config=$root/conf/data_${feat}.yaml
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
	echo "Computing word error rate (WER)..." # Note timit performance should be WER
	$COMPUTE_WER --mode=present ark,t:${result_dir}/ref_word.txt ark,t:${result_dir}/hypo_word.txt |& tee ${result_dir}/wer.txt
    done
fi

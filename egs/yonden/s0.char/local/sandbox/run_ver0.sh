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
stage=1  # start from 0 if you need to start from data preparation

mfcc_config=conf/mfcc.conf # 13 dimensional mfcc feature of kaldi default setting for timit
mfcc_hires=conf/mfcc_hires.conf # 40 dimensional mfcc feature of kaldi default setting for wsj
mel_config=clib/conf/feat/taco_mel_f80.json # 80 dimensional mel feature of tacotron tts

# Data and model options
run=run0
feat=mfcc39 # or mel80
dataset_name=yonden
# data_name=wsj0  # train:si84;dev:dev93;test:eval92
# data_name=wsj1  # train:si284;dev:dev93;test:eval92
model_name=EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp
exp_dir=exp/asr

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
save_interval=100 # save the model every x epoch

# options for evaluating ASR
set_uttid=None # subset of testing data (e.g. set_uttid=conf/data/test_small/set_uttid.txt)

search=beam
max_target=250 # the maximum length of the decoded sequence
beam_size=10

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

# data
yonden=/localwork/asrwork/yonden/data
segmented_recordings=/project/nakamura-lab08/Work/bin-wu/workspace/datasets/yonden/20220913_3to36 # directory that outputs segmented recordings according to wav.scp and segments

if [ ${stage} -le 1 ]; then
    echo "Data preparation..."

    # Start from the stage 2, skipping the stage 1 of the segmentation of long recordings.
    ./local/yonden_data_prep_all.sh --stage 2 --corpus $yonden --data_index "{003..036}*" --segmented_recordings $segmented_recordings --dir data/local/all_3to36
fi

if [ ${stage} -le 1 ]; then
    echo "Data division..."
    # Start from the stage 2, skipping the stage 1 of recovering info.json to original yonden data.
    ./local/yonden_data_division.sh --stage 1 --info data/local/all_3to36/info.json --dir data/local/division
fi

train="train_data3_5_7to17_baseline"
dev="dev_data4_baseline"
test="test_data6_baseline"
if [ ${stage} -le 2 ]; then
    date
    echo "Making 39-dimensional mfcc feature..."

    cp -r data/local/division/{$train,$dev,$test} data
    for dataset in $train $dev $test; do
	./local/scripts/feat_extract.sh --dataset ${dataset} --cmvn true --mfcc_conf $mfcc_config --delta_order 2 # 39-dim mfcc
    done
    date
fi

if [ ${stage} -le 2 ]; then
    date
    echo "Making 40-dimensional mfcc feature..."

    for dataset in $train $dev $test; do
    	x=${dataset}_mfcc40
	mkdir -p data/$x; cp -r data/${dataset}/* data/${x} # Fail to overwrite with command `cp -r data/${dataset} data/${x}'
	./local/scripts/feat_extract.sh --dataset ${x} --cmvn true --mfcc_conf $mfcc_hires # 40-dim mfcc
    done
    date
fi

if [ ${stage} -le 2 ]; then
    date
    echo "Making 80-dimensional mel feature..."

    for dataset in $train $dev $test; do
    	x=${dataset}_mel80
	mkdir -p data/$x; cp -r data/${dataset}/* data/${x} # Fail to overwrite with command `cp -r data/${dataset} data/${x}'
    	./local/scripts/feat_extract_taco.sh --dataset ${x} --cmvn true --mel_conf $mel_config  # 80-dim mel
    done
    date
fi

all="all_data3-28_baseline"
train_set=$all # use all to create dict to avoid oov problem of dict
dict=data/lang_1char/${train_set}_units.txt
non_ling_syms=data/lang_1char/${train_set}_non_ling_syms.txt
if [ ${stage} -le 3 ]; then
    echo "Dictionary preparation..."
    echo "dictionary: ${dict}"
    mkdir -p data/lang_1char/
    cp -r data/local/division/$all data

    echo "make a non-linguistic symbol list"
    # Task dependent.
    # Assume that the non-linguistic symbols of this text are in the format <symbol>.
    # Note that the first column of text is the uttid.
    cat data/${train_set}/kana | cut -f 2- -d" " | tr " " "\n" | sort | uniq | grep "<" > ${non_ling_syms}
    cat ${non_ling_syms}

    echo "make a dictionary"
    echo -ne "<unk> 0\n<pad> 1\n<sos> 2\n<eos> 3\n" > ${dict} # index convention of torchtext
    cat data/${train_set}/kana | cut -f 2- -d" " | tr " " "\n" | sort | uniq | grep -v -e '^\s*$' | grep -vE "<unk>|<pad>|<sos>|<eos>" | awk '{print $0 " " NR + 3}' >> ${dict}
    wc -l ${dict}
fi

if [ ${stage} -le 4 ]; then
    echo "Making json files"
    # get scp files (each line as 'uttid scp_content'), then merge them into utts.json.
    #     (eg. num_frames.scp, feat_dim.scp, num_tokens.scp, tokenid.scp, vocab_size.scp, feat.scp, token.scp and etc.)
    # If you want to add more information, just create more scp files in data2json.sh
    for dataset in $train $dev $test; do
	x=${dataset}
	local/scripts/data2json.sh --feat data/$x/feats.scp \
				   --token_scp data/$x/kana \
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
    for dataset in $train $dev $test; do
	x=${dataset}_mfcc40
	local/scripts/data2json.sh --feat data/$x/feats.scp \
				   --token_scp data/$x/kana \
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
    for dataset in $train $dev $test; do
	x=${dataset}_mel80
	local/scripts/data2json.sh --feat data/$x/feats.scp \
				   --token_scp data/$x/kana \
    				   --non-ling-syms ${non_ling_syms} \
    				   --output-utts-json data/$x/utts.json \
    				   --output-dir-of-scps data/$x/scps \
    				   data/$x ${dict}

    done
fi

# if [ ${stage} -le 6 ]; then
#     echo "Training ASR..."

#     for data_name in default; do
# 	data_config=clib/conf/data/${dataset_name}/${data_name}/asr_tts/data_${feat}.yaml # CHECKME (you can change the data_config to the setting of your prepared dataset)
# 	model_config=clib/conf/model/asr/seq2seq/${model_name}.yaml

# 	reducelr={\"factor\":$factor,\"patience\":$patience}

# 	exp_setting=${feat}_batchsize${batch_size}_cutoff${cutoff}_labelsmoothing${label_smoothing}_lr${lr}_gradclip${grad_clip}_factor${factor}_patience${patience}
# 	result_dir=${exp_dir}/${dataset_name}/${data_name}/${model_name}-${run}/${exp_setting}/train

# 	# comment out the --exit option if you are not sure how many epochs to run
# 	# comment out the --overwrite option when you do not want to overwrite the previous runs
# 	python local/scripts/train_asr.py \
# 	       --gpu $gpu \
# 	       --data_config $data_config \
# 	       --batch_size $batch_size \
# 	       --cutoff $cutoff \
# 	       --model_config $model_config \
# 	       --label_smoothing $label_smoothing \
# 	       --lr $lr \
# 	       --reducelr $reducelr \
# 	       --num_epochs $num_epochs \
# 	       --grad_clip $grad_clip \
# 	       --result $result_dir \
# 	       --save_interval $save_interval \
# 	       --exit \
# 	       --overwrite
#     done
# fi

# if [ ${stage} -le 7 ]; then
#     echo "Evaluating ASR for the best epoch..."

#     for data_name in default; do
# 	exp_setting=${feat}_batchsize${batch_size}_cutoff${cutoff}_labelsmoothing${label_smoothing}_lr${lr}_gradclip${grad_clip}_factor${factor}_patience${patience}
# 	model_path=${exp_dir}/${dataset_name}/${data_name}/${model_name}-${run}/${exp_setting}/train/best_model.mdl

# 	data_config=clib/conf/data/${dataset_name}/${data_name}/asr_tts/data_${feat}.yaml
# 	result_dir=${model_path%/train/*}/eval/beamsize${beam_size} # ${string%substring} # Deletes shortest match of $substring from back of $string.

# 	python local/scripts/eval_asr.py \
# 	       --gpu $gpu \
# 	       --data_config $data_config \
# 	       --set_uttid $set_uttid \
# 	       --batch_size 2 \
# 	       --model $model_path \
# 	       --max_target $max_target \
# 	       --search $search \
# 	       --beam_size $beam_size \
# 	       --result $result_dir

# 	echo
# 	echo "Computing character error rate (CER)..."
# 	COMPUTE_WER=/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi/src/bin/compute-wer
# 	$COMPUTE_WER --mode=present ark,t:${result_dir}/ref_char.txt ark,t:${result_dir}/hypo_char.txt |& tee ${result_dir}/cer.txt
# 	echo
# 	echo "Computing word error rate (WER)..." # Note timit performance should be WER
# 	$COMPUTE_WER --mode=present ark,t:${result_dir}/ref_word.txt ark,t:${result_dir}/hypo_word.txt |& tee ${result_dir}/wer.txt
#     done
# fi

# if [ ${stage} -le 8 ]; then
#     echo "Evaluating ASR for the last epoch..."

#     for data_name in default; do
# 	exp_setting=${feat}_batchsize${batch_size}_cutoff${cutoff}_labelsmoothing${label_smoothing}_lr${lr}_gradclip${grad_clip}_factor${factor}_patience${patience}
# 	# model_path=${exp_dir}/${dataset_name}/${data_name}/${model_name}-${run}/${exp_setting}/train/best_model.mdl
# 	model_path=${exp_dir}/${dataset_name}/${data_name}/${model_name}-${run}/${exp_setting}/train/last_model.mdl

# 	data_config=clib/conf/data/${dataset_name}/${data_name}/asr_tts/data_${feat}.yaml
# 	# result_dir=${model_path%/train/*}/eval/beamsize${beam_size} # ${string%substring} # Deletes shortest match of $substring from back of $string.
# 	result_dir=${model_path%/train/*}/eval_last_epoch/beamsize${beam_size} # ${string%substring} # Deletes shortest match of $substring from back of $string.

# 	python local/scripts/eval_asr.py \
# 	       --gpu $gpu \
# 	       --data_config $data_config \
# 	       --set_uttid $set_uttid \
# 	       --batch_size 2 \
# 	       --model $model_path \
# 	       --max_target $max_target \
# 	       --search $search \
# 	       --beam_size $beam_size \
# 	       --result $result_dir

# 	echo
# 	echo "Computing character error rate (CER)..."
# 	COMPUTE_WER=/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi/src/bin/compute-wer
# 	$COMPUTE_WER --mode=present ark,t:${result_dir}/ref_char.txt ark,t:${result_dir}/hypo_char.txt |& tee ${result_dir}/cer.txt
# 	echo
# 	echo "Computing word error rate (WER)..." # Note timit performance should be WER
# 	$COMPUTE_WER --mode=present ark,t:${result_dir}/ref_word.txt ark,t:${result_dir}/hypo_word.txt |& tee ${result_dir}/wer.txt
#     done
# fi

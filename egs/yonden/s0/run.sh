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
stage=8  # start from 0 if you need to start from data preparation

mfcc_config=conf/mfcc.conf # 13 dimensional mfcc feature of kaldi default setting for timit
mfcc_hires=conf/mfcc_hires.conf # 40 dimensional mfcc feature of kaldi default setting for wsj
mel_config=clib/conf/feat/taco_mel_f80.json # 80 dimensional mel feature of tacotron tts

# Data and model options
run=run0
feat=mel80 # mfcc39 or mfcc40 or mel80 # be careful about the feat consistency with pretrained_model, train, dev, and test
dataset_name=yonden
# data_name=wsj0  # train:si84;dev:dev93;test:eval92
# data_name=wsj1  # train:si284;dev:dev93;test:eval92
model_name=EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp
exp_dir=exp/asr_pretrained_extend_dict

# options for training ASR
asr_seed=2020
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

# options for pretrain model
pretrained_dict=conf/csj/dict/train_units.txt
pretrained_model=conf/csj/model/mel80/best_model.mdl

# options for running ASR (e.g., 'train=train_si284_mel80' for wsj)
# e.g., ./local/sandbox/run_csj_pretrained.sh --stage 5 --test ${dataset}_${feat} --feat ${feat} --pretrained_dict ${pretrained_dict} --pretrained_model ${pretrain_model}
train=train_data3-36_remove_4_6_28_36_mel80 # if train is empty and pretrained_model not empty, use pretrained model directly for test set
dev=dev_data4_28_mel80
test=test_data6_mel80

tag=default_csj_pretrained_with_yonden

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

# root for ASR
if [ ! -z $pretrained_model ];then
    tag=${tag}_pretrained_train.${train}_dev.${dev}_test.${test}
else
    tag=${tag}_train.${train}_dev.${dev}_test.${test}
fi

data_name=$tag
root=exp/conf/asr_pretrained/yonden/conf/asr_${tag}

# data
yonden=/localwork/asrwork/yonden/data
segmented_recordings=/project/nakamura-lab08/Work/bin-wu/workspace/datasets/yonden/20220913_3to36 # directory that outputs segmented recordings according to wav.scp and segments

if [ ${stage} -le -1 ]; then
    echo "Preparing pretrained dict and model files (details at conf/csj/{dict,model})..."
    mkdir -p conf/csj/dict
    mkdir -p conf/csj/model/{mfcc39,mfcc40,mel80}

    echo "Copying the dict of csj..."
    cp /project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/csj/s0/data/lang_1char/train_units.txt conf/csj/dict/

    echo "Copying the pretrained models of mfcc39, mfcc40, and mel80 features..."
    cp /project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/csj/s0/exp/asr_extend_dict/csj/default/EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp-run0/mfcc39_batchsize32_cutoff1800_labelsmoothing0.05_lr0.001_gradclip5_factor0.5_patience3/train/best_model.{conf,mdl} conf/csj/model/mfcc39/
    cp /project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/csj/s0/exp/asr_extend_dict/csj/default/EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp-run0/mfcc40_batchsize32_cutoff1800_labelsmoothing0.05_lr0.001_gradclip5_factor0.5_patience3/train/best_model.{conf,mdl} conf/csj/model/mfcc40/
    cp /project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/csj/s0/exp/asr_extend_dict/csj/default/EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp-run0/mel80_batchsize32_cutoff1800_labelsmoothing0.05_lr0.001_gradclip5_factor0.5_patience3/train/best_model.{conf,mdl} conf/csj/model/mel80/
fi

if [ ${stage} -le 1 ]; then
    echo "Data preparation..."

    # Start from the stage 2, skipping the stage 1 of the segmentation of long recordings.
    ./local/yonden_data_prep_all.sh --stage 2 --corpus $yonden --data_index "{003..036}*" --segmented_recordings $segmented_recordings --dir data/local/all_3to36
fi

if [ ${stage} -le 1 ]; then
    echo "Data division..."

    ./local/yonden_data_division.sh --stage 2 --info data/local/all_3to36/info.json --dir data/local/division
fi

# for dataset in data/local/division/*; do echo $(basename $dataset); done | tr -s '\n' ' '
source_data_dir=data/local/division
# datasets="test_data6 test_data6_ampnorm test_data6_spkinfo test_data6_spkinfo_daily test_data36 test_data36_ampnorm test_data36_spkinfo test_data36_spkinfo_daily dev_data4_28 dev_data4_28_ampnorm dev_data4_28_spkinfo dev_data4_28_spkinfo_daily train_data3-36_remove_4_6_28_36 train_data3-36_remove_4_6_28_36_ampnorm train_data3-36_remove_4_6_28_36_spkinfo train_data3-36_remove_4_6_28_36_spkinfo_daily"
datasets="test_data6 test_data6_ampnorm test_data36 test_data36_ampnorm dev_data4_28 dev_data4_28_ampnorm train_data3-36_remove_4_6_28_36 train_data3-36_remove_4_6_28_36_ampnorm"
if [ ${stage} -le 2 ]; then
    echo "Making mfcc39, mfcc40, and mel80 features..."

    ./local/yonden_feat_extraction.sh --datasets "$datasets" --source_data_dir $source_data_dir --mfcc39 true --mfcc40 true --mel80 true --cmvn true
fi

all="all_data3-36"
train_set=$all # use all to create dict to avoid oov problem of dict
dict=data/lang_1char/${train_set}_units.txt
non_ling_syms=data/lang_1char/${train_set}_non_ling_syms.txt
if [ ${stage} -le 3 ]; then
    echo "Dictionary preparation..."
    echo "dictionary: ${dict}"
    mkdir -p data/lang_1char/
    cp -r data/local/division/$all data

    echo "Make a non-linguistic symbol list"
    # Task dependent.
    # Assume that the non-linguistic symbols of this text are in the format <symbol>.
    # Note that the first column of text is the uttid.
    cat data/${train_set}/kana | cut -f 2- -d" " | tr " " "\n" | sort | uniq | grep "<" > ${non_ling_syms}
    cat ${non_ling_syms}

    echo "Make a dictionary"
    echo -ne "<unk> 0\n<pad> 1\n<sos> 2\n<eos> 3\n" > ${dict} # index convention of torchtext
    cat data/${train_set}/kana | cut -f 2- -d" " | tr " " "\n" | sort | uniq | grep -v -e '^\s*$' | grep -vE "<unk>|<pad>|<sos>|<eos>" | awk '{print $0 " " NR + 3}' >> ${dict}
    wc -l ${dict}

    if [ ! -z $pretrained_dict ]; then
	echo ""
 
	pre_dict=$dict; pre_non_ling_syms=$non_ling_syms
	dict=data/lang_1char/pretrained_units.txt
	non_ling_syms=data/lang_1char/pretrained_non_ling_syms.txt

	echo "Merging current and pretrained non-linguistic symbol lists..."
	cat <(awk '{print $1}' $pretrained_dict | grep "<" | grep -vE "<unk>|<pad>|<sos>|<eos>") <(cat $pre_non_ling_syms) | sort -u > $non_ling_syms
	cat ${non_ling_syms}

	echo "Using pretrained dict (dict from pretrained model)..."
	# csj pretrain label should be consistent with yonden data; so should not merge dict, instead, we replace the dict
	cp $pretrained_dict $dict
	echo "dictionary: ${dict}"
	wc -l $dict
    fi
fi

if [ ! -z $pretrained_dict ]; then dict=data/lang_1char/pretrained_units.txt; non_ling_syms=data/lang_1char/pretrained_non_ling_syms.txt; fi
if [ ${stage} -le 4 ]; then
    echo "Making json files..."

    echo "Using dict $dict..."
    ./local/yonden_data2json.sh --datasets "$datasets" --dict $dict --token_scp_type kana --mfcc39 true --mfcc40 true --mel80 true
fi

if [ $stage -le 5 ]; then
    mkdir -p $root/conf
    echo train: \'${PWD}/data/$train/utts.json\' > $root/conf/data_${feat}.yaml
    echo dev: \'${PWD}/data/$dev/utts.json\' >> $root/conf/data_${feat}.yaml
    echo test: \'${PWD}/data/$test/utts.json\' >> $root/conf/data_${feat}.yaml
    echo "token2id: '$PWD/$dict'" >> $root/conf/data_${feat}.yaml
fi

if [ -z "$train" ] || [ -z "$dev" ] || [ -z "$test" ]; then echo "Warning: training, development, or test sets is not available, skip training ASR..."; fi
echo -e "train: $train\ndev: $dev\ntest: $test\npretrained_model: $pretrained_model\n"
if [ ${stage} -le 6 ] && [ ! -z "$train" ] && [ ! -z "$dev" ] && [ ! -z "$test" ]; then
    echo "Training ASR..."

#    for data_name in wsj1; do
	# exp/dpgmm2asr/asr_dpgmm_seed123_K39/conf/data_dpgmm_seed123_K39.yaml
	data_config=$root/conf/data_${feat}.yaml # CHECKME (you can change the data_config to the setting of your prepared dataset)
	model_config=clib/conf/model/asr/seq2seq/${model_name}.yaml

	reducelr={\"factor\":$factor,\"patience\":$patience}

	exp_setting=batchsize${batch_size}_cutoff${cutoff}_labelsmoothing${label_smoothing}_lr${lr}_gradclip${grad_clip}_factor${factor}_patience${patience}
	result_dir=${exp_dir}/${dataset_name}/${data_name}/${model_name}-${run}/${exp_setting}/train

	# comment out the --exit option if you are not sure how many epochs to run
	# comment out the --overwrite option when you do not want to overwrite the previous runs
	python local/scripts/train_asr.py \
	       --seed $asr_seed \
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
	       --pretrained_model "$pretrained_model" \
	       --exit \
	       --overwrite
#    done
fi


if [ ${stage} -le 7 ]; then
    echo "Evaluating ASR..."

#   for data_name in wsj1; do
	exp_setting=batchsize${batch_size}_cutoff${cutoff}_labelsmoothing${label_smoothing}_lr${lr}_gradclip${grad_clip}_factor${factor}_patience${patience}
	model_path=${exp_dir}/${dataset_name}/${data_name}/${model_name}-${run}/${exp_setting}/train/best_model.mdl

	if [ ! -z $pretrained_model ] && [ -z "$train" ]; then
	    model_path=$pretrained_model
	fi

	data_config=$root/conf/data_${feat}.yaml
	# result_dir=${model_path%/train/*}/eval/beamsize${beam_size} # ${string%substring} # Deletes shortest match of $substring from back of $string.
	result_dir=${exp_dir}/${dataset_name}/${data_name}/${model_name}-${run}/${exp_setting}/eval/beamsize${beam_size} # ${string%substring} # Deletes shortest match of $substring from back of $string.

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
#    done
fi

if [ ! -z $pretrained_model ] && [ -z "$train" ]; then echo "Skip evaluating the last epoch..."; exit 1; fi
if [ ${stage} -le 8 ]; then
    echo "Evaluating ASR from the last epoch..."

#   for data_name in wsj1; do
	exp_setting=batchsize${batch_size}_cutoff${cutoff}_labelsmoothing${label_smoothing}_lr${lr}_gradclip${grad_clip}_factor${factor}_patience${patience}
	# model_path=${exp_dir}/${dataset_name}/${data_name}/${model_name}-${run}/${exp_setting}/train/best_model.mdl
	model_path=${exp_dir}/${dataset_name}/${data_name}/${model_name}-${run}/${exp_setting}/train/last_model.mdl

	data_config=$root/conf/data_${feat}.yaml
	# result_dir=${model_path%/train/*}/eval/beamsize${beam_size} # ${string%substring} # Deletes shortest match of $substring from back of $string.
	result_dir=${exp_dir}/${dataset_name}/${data_name}/${model_name}-${run}/${exp_setting}/eval_last_epoch/beamsize${beam_size} # ${string%substring} # Deletes shortest match of $substring from back of $string.

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
#    done
fi

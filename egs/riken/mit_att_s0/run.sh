#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -uo pipefail

# general conf
stage=8  # start from 0 if you need to start from data preparation
run=run0
# dataset_name=mit_sample # see https://marmosetbehavior.mit.edu
dataset_name=mit_data # see dataset shared by the paper of 'Close range vocal interaction in the common marmoset (Callithrix Jacchus)'
data_name=mit0 # training:pair3-10;dev:pair2;test:pair1

tag=two_stream_asr # tag for the experiment
feat_name="mel"

# data conf
data_div_yaml="conf/data/division.yaml"
feat_yaml="conf/feat/feat.yaml"
token2id_yaml="conf/dict/token2id_marmoset.yaml"
train_chunk_size_ms=500
train_chunk_shift_ms=150
dev_chunk_size_ms=500
dev_chunk_shift_ms=400
test_chunk_size_ms=500
test_chunk_shift_ms=400
feat_dir="feat"

# model conf
model_name=EncRNNDecRNNAtt-enc3_bi256_ds0_drop-dec1_h512_do0.25-att_mlp # EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp
exp_dir=exp/att

# options for training ASR
asr_seed=2020
gpu=auto
batch_size=128 # 32
cutoff=2000 # 1600 # cut off long sentences
label_smoothing=0.05
lr=0.001
num_epochs=70
grad_clip=5
factor=0.5 # for lr scheduler
patience=10 # 3 # for lr scheduler
save_interval=20 # 1 # save the model every x epoch
label_downsampling=1

# options for evaluating ASR
set_uttid=None # subset of testing data (e.g. set_uttid=conf/data/test_small/set_uttid.txt)

search=beam
max_target=2000 # 250 # the maximum length of the decoded sequence
beam_size=1 # 10

# dataset
# mit_sample=/data/share/bin-wu/data/marmoset/vocalization/marmoset_mit_cnn/original/Wave_files
mit_data=/data/share/bin-wu/data/marmoset/vocalization/marmoset_mit

# eval
first="p1a1_toget" # first uttid of a test pair
sec="p1a2_toget" # second uttid of a test pair
eval_epoch=

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. local/scripts/parse_options.sh || exit 1

root=exp/root/asr_${data_name}_${tag}_${feat_name}
feat=${data_name}_${tag}_${feat_name}

if [ ${stage} -le 1 ]; then
    date
    echo "Data preparation..."
    ./local/mit_data_data_prep.sh --mit_data $mit_data --dataset_name $dataset_name --stage 0 # Get info.json
    date
fi

chunk_json_dir="data/${dataset_name}/chunk_info/traincsize${train_chunk_size_ms}cshift${train_chunk_shift_ms}_devcsize${dev_chunk_size_ms}cshift${dev_chunk_shift_ms}_testcsize${test_chunk_size_ms}cshift${test_chunk_shift_ms}"
info_json="data/${dataset_name}/info.json"
if [ ${stage} -le 2 ]; then
    date
    echo "Chunk preparation..."
    echo "Breaking long audios and segments into short chunks..."
    python local/mit_seg_prep.py --data_div_yaml $data_div_yaml \
	   --info_json $info_json \
	   --feat_yaml $feat_yaml \
	   --token2id_yaml $token2id_yaml \
	   --train_chunk_size_ms $train_chunk_size_ms \
	   --train_chunk_shift_ms $train_chunk_shift_ms \
	   --dev_chunk_size_ms $dev_chunk_size_ms \
	   --dev_chunk_shift_ms $dev_chunk_shift_ms \
	   --test_chunk_size_ms $test_chunk_size_ms \
	   --test_chunk_shift_ms $test_chunk_shift_ms \
	   --feat_dir $feat_dir \
	   --chunk_json_dir $chunk_json_dir
    date
fi

if [ $stage -le 5 ]; then
    echo "Creating the data conf file..."
    mkdir -p $root/conf
    if [ -f ${chunk_json_dir}/train_chunk.json ]; then echo "train: '${PWD}/${chunk_json_dir}/train_chunk.json'"; fi > $root/conf/data_${feat}.yaml
    if [ -f ${chunk_json_dir}/dev_chunk.json ]; then echo "dev: '${PWD}/${chunk_json_dir}/dev_chunk.json'"; fi >> $root/conf/data_${feat}.yaml
    if [ -f ${chunk_json_dir}/test_chunk.json ]; then echo "test: '${PWD}/${chunk_json_dir}/test_chunk.json'"; fi >> $root/conf/data_${feat}.yaml
    echo "token2id: '$PWD/${token2id_yaml}'" >> $root/conf/data_${feat}.yaml
fi

if [ ${stage} -le 6 ]; then
    echo "Training ASR..."

    data_config=$root/conf/data_${feat}.yaml # CHECKME (you can change the data_config to the setting of your prepared dataset)
    model_config=clib/conf/model/asr/seq2seq/${model_name}.yaml

    reducelr={\"factor\":$factor,\"patience\":$patience}

    exp_setting=${feat}_batchsize${batch_size}_cutoff${cutoff}_labelsmoothing${label_smoothing}_lr${lr}_gradclip${grad_clip}_factor${factor}_patience${patience}_chunksizeshifttr${train_chunk_size_ms}tr${train_chunk_shift_ms}dev${dev_chunk_size_ms}dev${dev_chunk_shift_ms}test${test_chunk_size_ms}test${test_chunk_shift_ms}_labeldownsample${label_downsampling}

    result_dir=${exp_dir}/${dataset_name}/${data_name}/${model_name}-${run}/${exp_setting}/train

    # comment out the --exit option if you are not sure how many epochs to run
    # comment out the --overwrite option when you do not want to overwrite the previous runs
    python local/scripts/train_asr.py \
	   --seed $asr_seed \
	   --gpu $gpu \
	   --data_config $data_config \
	   --batch_size $batch_size \
	   --cutoff $cutoff \
	   --label_downsampling $label_downsampling \
	   --model_config $model_config \
	   --label_smoothing $label_smoothing \
	   --lr $lr \
	   --reducelr $reducelr \
	   --num_epochs $num_epochs \
	   --grad_clip $grad_clip \
	   --result $result_dir \
	   --save_interval $save_interval \
	   --overwrite
#     	   --exit
fi

if [ ${stage} -le 7 ]; then
    echo "Evaluating ASR..."

    exp_setting=${feat}_batchsize${batch_size}_cutoff${cutoff}_labelsmoothing${label_smoothing}_lr${lr}_gradclip${grad_clip}_factor${factor}_patience${patience}_chunksizeshifttr${train_chunk_size_ms}tr${train_chunk_shift_ms}dev${dev_chunk_size_ms}dev${dev_chunk_shift_ms}test${test_chunk_size_ms}test${test_chunk_shift_ms}_labeldownsample${label_downsampling}
    model_path=${exp_dir}/${dataset_name}/${data_name}/${model_name}-${run}/${exp_setting}/train/best_model.mdl

    data_config=$root/conf/data_${feat}.yaml
    result_dir=${model_path%/train/*}/eval/beamsize${beam_size} # ${string%substring} # Deletes shortest match of $substring from back of $string.
    if [ ! -z "$eval_epoch" ]; then
	model_path=${exp_dir}/${dataset_name}/${data_name}/${model_name}-${run}/${exp_setting}/train/model_e${eval_epoch}.mdl
	result_dir=${model_path%/train/*}/eval/beamsize${beam_size}_epoch${eval_epoch} # ${string%substring} # Deletes shortest match of $substring from back of $string.
    fi

    python local/scripts/eval_asr.py \
	   --gpu $gpu \
	   --data_config $data_config \
	   --set_uttid $set_uttid \
	   --batch_size 128 \
	   --model $model_path \
	   --max_target $max_target \
	   --search $search \
	   --beam_size $beam_size \
	   --result $result_dir

    echo
    echo "Computing character error rate (CER)..."
    KALDI_ROOT=/home/bin-wu/share/tools/kaldi
    COMPUTE_WER=$KALDI_ROOT/src/bin/compute-wer
    $COMPUTE_WER --mode=present ark,t:${result_dir}/ref_char.txt ark,t:${result_dir}/hypo_char.txt |& tee ${result_dir}/cer.txt
    echo
fi

if [ ${stage} -le 8 ]; then
    echo "Evaluating ASR..."
    exp_setting=${feat}_batchsize${batch_size}_cutoff${cutoff}_labelsmoothing${label_smoothing}_lr${lr}_gradclip${grad_clip}_factor${factor}_patience${patience}_chunksizeshifttr${train_chunk_size_ms}tr${train_chunk_shift_ms}dev${dev_chunk_size_ms}dev${dev_chunk_shift_ms}test${test_chunk_size_ms}test${test_chunk_shift_ms}_labeldownsample${label_downsampling}
    model_path=${exp_dir}/${dataset_name}/${data_name}/${model_name}-${run}/${exp_setting}/train/best_model.mdl

    result_dir=${model_path%/train/*}/eval/beamsize${beam_size} # ${string%substring} # Deletes shortest match of $substring from back of $string.
    if [ ! -z "$eval_epoch" ]; then
	model_path=${exp_dir}/${dataset_name}/${data_name}/${model_name}-${run}/${exp_setting}/train/model_e${eval_epoch}.mdl
	result_dir=${model_path%/train/*}/eval/beamsize${beam_size}_epoch${eval_epoch} # ${string%substring} # Deletes shortest match of $substring from back of $string.
    fi

    python local/mit_get_eval_seg.sh --chunk_json_file $chunk_json_dir/test_chunk.json \
	   --labels_file ${result_dir}/hypo_char.txt \
	   --pad_token "pad" \
	   --label_downsampling $label_downsampling

    eval_dir=${result_dir}/seg
    rm -rf $eval_dir/results.txt
    python local/mit_cnn_eval_acc.py --info_json data/$dataset_name/info.json \
	   --hypo_files $eval_dir/test_${first}.txt $eval_dir/test_${sec}.txt \
	   --ref_uttids $first $sec | tee -a $eval_dir/results.txt
    echo | tee -a $eval_dir/results.txt
fi

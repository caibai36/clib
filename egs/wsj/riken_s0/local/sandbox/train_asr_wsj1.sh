run=run0
feat=mfcc40
dataset_name=wsj
data_name=wsj1  # train:si284;dev:dev93;test:eval92
model_name=EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp
exp_dir=exp/tmp

gpu=auto
batch_size=32
cutoff=1600
label_smoothing=0.05
lr=0.001
num_epochs=70
grad_clip=5

# for lr scheduler
factor=0.5
patience=3

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

data_config=conf/data/${dataset_name}/${data_name}/asr_tts/data_${feat}.yaml
model_config=conf/model/asr/seq2seq/${model_name}.yaml

reducelr={\"factor\":$factor,\"patience\":$patience}

exp_setting=${feat}_batchsize${batch_size}_cutoff${cutoff}_labelsmoothing${label_smoothing}_lr${lr}_gradclip${grad_clip}_factor${factor}_patience${patience}
result_dir=${exp_dir}/${dataset_name}/${data_name}/${model_name}-${run}/${exp_setting}/train

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
       --result $result_dir

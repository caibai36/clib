# Implemented by bin-wu at 21:19 on 2 Feb 2021

# general configuration
stage=8  # start from 0 if you need to start from data preparation

# subcommands
run_kana_asr=false

# settings in general
tag=yonden_kana_baseline
dataset_name=yonden # timit # wsj
data_name=yonden
feat_name=MFCC40

train_data=$PWD/data/train_data3_5_7to17_baseline # /project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/wsj/s0/data/train_si84
dev_data=$PWD/data/dev_data4_baseline # /project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/wsj/s0/data/test_dev93
test_data=$PWD/data/test_data6_baseline # /project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/wsj/s0/data/test_eval92
train_set=all_data3-28_baseline # name of the training data set

feat_train=$PWD/data/train_data3_5_7to17_baseline_mfcc40/feats.scp
feat_dev=$PWD/data/dev_data4_baseline_mfcc40/feats.scp
feat_test=$PWD/data/test_data6_baseline_mfcc40/feats.scp

# settings for kana asr
token_scp_name=kana # # e.g., data/train/kana with a line "uttid1 シ キ <space> シャ <space> ヒ ラ オ カ <space> <period>"

exp_dir=exp/tmp_asr_full # folder holding results of attentional ASR system

# the following are the default setting for ASR system
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
save_interval=1 # save the model every x epoch

# options for evaluating ASR
set_uttid=None # subset of testing data (e.g. set_uttid=conf/data/test_small/set_uttid.txt)
search=beam
max_target=250 # the maximum length of the decoded sequence
beam_size=10

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

model_name=EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp
exp_dir=$exp_dir/aseed${asr_seed}.gpu${gpu}.bs${batch_size}.cf${cutoff}.ls${label_smoothing}.lr${lr}.ne${num_epochs}.gp${grad_clip}.factor${factor}.pat${patience}.si${save_interval}.search${search}.mt${max_target}.beamsize.${beam_size} # add full asr settings

if [ $stage == 8 ]; then
    echo "help: ./local/sandbox/asr_general/run_all.sh --run_kana_asr true"
fi

if $run_kana_asr; then
    # Run kana ASR after feature extraction
    root=exp/asr/${data_name}_${tag}_${feat_name}
    ./local/sandbox/asr_general/run_singlefeat2asr.sh --stage 0 --tag ${tag} --dataset_name ${dataset_name} --data_name ${data_name} --feat_name ${feat_name} \
						      --train_data ${train_data} --dev_data ${dev_data} --test_data ${test_data} --train_set ${train_set} \
						      --feat_train ${feat_train} --feat_dev ${feat_dev} --feat_test ${feat_test} --token_scp_name $token_scp_name\
						      --asr_seed ${asr_seed} --gpu ${gpu} --batch_size ${batch_size} --cutoff ${cutoff} --label_smoothing ${label_smoothing} `# config asr`\
						      --lr ${lr} --num_epochs ${num_epochs} --grad_clip ${grad_clip} --factor ${factor} --patience ${patience} --save_interval ${save_interval} `# train asr`\
						      --set_uttid ${set_uttid} --search ${search} --max_target ${max_target} --beam_size ${beam_size} `# eval asr`\
						      --exp_dir ${exp_dir}
fi

# Implemented by bin-wu at 21:19 on 2 Feb 2021

# general configuration
stage=8  # start from 0 if you need to start from data preparation

# subcommands
bnf_feat_extract=false
dpgmm_feat_extract=false
hybrid_feat_extract=false
run_dpgmm_asr=false
run_hybrid_asr=false
run_bnf_asr=false
cat_dpgmm_asr=false # asr with features concatenating mfcc and dpgmm features
cat_bnf_asr=false
cat_hybrid_asr=false

# settings in general

# settings for DPGMM feature extraction
seed=123 # seed for dpgmm
K0=98 # at least K0 + 1 dimension, additional 1 for possible new clusters

# settings for bottleneck feature extraction
mode=1 # different modes of text processing for data of tandem system
bnf_dim=42

# settings for hybird LSTM neural network
nseed=123 # seed for the initalization of weights of neural network
l=8 # number of frames as the left context
r=8 # number of frames as the right context
bs=256 # batch size
hd=512 # hidden dimension
nl=5   # number of layers
ne=20  # number of epochs

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

tag=asr_general
dataset_name=bjava # timit # wsj
data_name=default # wsj0 
feat_name=MFCC39 # MFCC40

train_data=$PWD/data/train # /project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/wsj/s0/data/train_si84
dev_data=$PWD/data/dev # /project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/wsj/s0/data/test_dev93
test_data=$PWD/data/test # /project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/wsj/s0/data/test_eval92
train_set=train # train_si284; name of the training data set

model_name=EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp
exp_dir=$exp_dir/aseed${asr_seed}.gpu${gpu}.bs${batch_size}.cf${cutoff}.ls${label_smoothing}.lr${lr}.ne${num_epochs}.gp${grad_clip}.factor${factor}.pat${patience}.si${save_interval}.search${search}.mt${max_target}.beamsize.${beam_size} # add full asr settings

if [ $stage == 8 ]; then
    echo "help: ./local/sandbox/asr_general/run_all.sh --dpgmm_feat_extract true"
    echo "      ./local/sandbox/asr_general/run_all.sh --bnf_feat_extract true"
    echo "      ./local/sandbox/asr_general/run_all.sh --bnf_feat_extract true --bnf_dim 99"
    echo "      ./local/sandbox/asr_general/run_all.sh --dpgmm_feat_extract true --bnf_feat_extract true"
    echo "      ./local/sandbox/asr_general/run_all.sh --hybrid_feat_extract true"
    echo "      ./local/sandbox/asr_general/run_all.sh --hybrid_feat_extract true --run_hybrid_asr true"
    echo "      ./local/sandbox/asr_general/run_all.sh --run_dpgmm_asr true"
    echo "      ./local/sandbox/asr_general/run_all.sh --run_hybrid_asr true"
    echo "      ./local/sandbox/asr_general/run_all.sh --run_bnf_asr true"
    echo "      ./local/sandbox/asr_general/run_all.sh --run_bnf_asr true --bnf_dim 99"
fi

if $dpgmm_feat_extract; then
    # Extract DPGMM feature
    ./local/sandbox/asr_general/run_dpgmm2asr.sh --stage 0 --tag ${tag} --dataset_name ${dataset_name} --data_name ${data_name} --feat_name ${feat_name} \
					  --train_data ${train_data} --dev_data ${dev_data} --test_data ${test_data} --train_set ${train_set} \
					  --seed ${seed} --K0 ${K0} \
					  --asr_seed ${asr_seed} --gpu ${gpu} --batch_size ${batch_size} --cutoff ${cutoff} --label_smoothing ${label_smoothing} `# config asr`\
					  --lr ${lr} --num_epochs ${num_epochs} --grad_clip ${grad_clip} --factor ${factor} --patience ${patience} --save_interval ${save_interval} `# train asr`\
					  --set_uttid ${set_uttid} --search ${search} --max_target ${max_target} --beam_size ${beam_size} `# eval asr`\
					  --model_name ${model_name} \
					  --exp_dir ${exp_dir} \
					  --run_asr false # extract DPGMM feature without running ASR
fi

if $hybrid_feat_extract; then
    # Extract hybrid feature (after extracting DPGMM feature)
    dpgmm_root=exp/dpgmm2asr/asr_${data_name}_${tag}_dpgmm_seed${seed}_K${K0}_from_${feat_name}
    dpgmm_train=$dpgmm_root/data/train/feats.scp
    dpgmm_dev=$dpgmm_root/data/dev/feats.scp
    dpgmm_test=$dpgmm_root/data/test/feats.scp
    ./local/sandbox/asr_general/run_hybrid2asr_mse.sh --stage 0 --tag ${tag} --dataset_name ${dataset_name} --data_name ${data_name} --feat_name ${feat_name} \
						      --train_data ${train_data} --dev_data ${dev_data} --test_data ${test_data} --train_set ${train_set} \
						      --dpgmm_train ${dpgmm_train} --dpgmm_dev ${dpgmm_dev} --dpgmm_test ${dpgmm_test} \
						      --seed ${seed} --K0 ${K0} `# DPGMM feature setting` \
						      --nseed ${nseed} --num_left_context ${l} --num_right_context ${r} --hybrid_batch_size ${bs} \
						      --hybrid_hidden_dim ${hd} --hybrid_num_layers ${nl} --hybrid_num_epochs ${ne} \
						      --asr_seed ${asr_seed} --gpu ${gpu} --batch_size ${batch_size} --cutoff ${cutoff} --label_smoothing ${label_smoothing} `# config asr`\
						      --lr ${lr} --num_epochs ${num_epochs} --grad_clip ${grad_clip} --factor ${factor} --patience ${patience} --save_interval ${save_interval} `# train asr`\
						      --set_uttid ${set_uttid} --search ${search} --max_target ${max_target} --beam_size ${beam_size} `# eval asr`\
						      --model_name ${model_name} \
						      --exp_dir ${exp_dir} \
						      --run_asr false # extract hybrid feature without running ASR
fi

if $bnf_feat_extract; then
    # Extract BNF feature
    t=${tag};tag=${tag}_mode${mode}
    ./local/sandbox/asr_general/make_bnf.sh --stage 0 --tag ${tag} --dataset_name ${dataset_name} --data_name ${data_name} --feat_name ${feat_name} \
					    --train_data ${train_data} --dev_data ${dev_data} --test_data ${test_data} \
					    --mode ${mode} \
					    --bnf_dim ${bnf_dim}
    tag=${t}
fi

if $run_dpgmm_asr; then
    # Run DPGMM ASR after feature extraction
    ./local/sandbox/asr_general/run_dpgmm2asr.sh --stage 2 `# run DPGMM ASR assuming DPGMM feature extracted` \
                                          --tag ${tag} --dataset_name ${dataset_name} --data_name ${data_name} --feat_name ${feat_name} \
                                          --train_data ${train_data} --dev_data ${dev_data} --test_data ${test_data} --train_set ${train_set} \
					  --seed ${seed} --K0 ${K0} \
					  --asr_seed ${asr_seed} --gpu ${gpu} --batch_size ${batch_size} --cutoff ${cutoff} --label_smoothing ${label_smoothing} `# config asr`\
					  --lr ${lr} --num_epochs ${num_epochs} --grad_clip ${grad_clip} --factor ${factor} --patience ${patience} --save_interval ${save_interval} `# train asr`\
					  --set_uttid ${set_uttid} --search ${search} --max_target ${max_target} --beam_size ${beam_size} `# eval asr`\
					  --exp_dir ${exp_dir}

    dpgmm_root=exp/dpgmm2asr/asr_${data_name}_${tag}_dpgmm_seed${seed}_K${K0}_from_${feat_name}
    dpgmm_train=$dpgmm_root/data/train/feats.scp
    dpgmm_dev=$dpgmm_root/data/dev/feats.scp
    dpgmm_test=$dpgmm_root/data/test/feats.scp
    feat1_name=${feat_name}
    feat2_name=dpgmm_seed${seed}_K${K0}_from_${feat_name}
    ./local/sandbox/asr_general/run_concatfeat2asr.sh --stage 0 --tag ${tag} --dataset_name ${dataset_name} --data_name ${data_name} \
						    --train_data ${train_data} --dev_data ${dev_data} --test_data ${test_data} --train_set ${train_set} \
						    --feat_train ${train_data}/feats.scp --feat_dev ${dev_data}/feats.scp --feat_test ${test_data}/feats.scp \
						    --feat_train2 ${dpgmm_train} --feat_dev2 ${dpgmm_dev} --feat_test2 ${dpgmm_test} \
						    --feat1_name ${feat1_name} --feat2_name ${feat2_name} \
						    --asr_seed ${asr_seed} --gpu ${gpu} --batch_size ${batch_size} --cutoff ${cutoff} --label_smoothing ${label_smoothing} `# config asr`\
						    --lr ${lr} --num_epochs ${num_epochs} --grad_clip ${grad_clip} --factor ${factor} --patience ${patience} --save_interval ${save_interval} `# train asr`\
						    --set_uttid ${set_uttid} --search ${search} --max_target ${max_target} --beam_size ${beam_size} `# eval asr`\
						    --exp_dir ${exp_dir}
fi

if $run_hybrid_asr; then
    # Run hybrid ASR after feature extraction
    dpgmm_root=exp/dpgmm2asr/asr_${data_name}_${tag}_dpgmm_seed${seed}_K${K0}_from_${feat_name}
    dpgmm_train=$dpgmm_root/data/train/feats.scp
    dpgmm_dev=$dpgmm_root/data/dev/feats.scp
    dpgmm_test=$dpgmm_root/data/test/feats.scp

    ./local/sandbox/asr_general/run_hybrid2asr_mse.sh --stage 2 `# run hybrid ASR assuming hybrid feature extracted` \
						      --tag ${tag} --dataset_name ${dataset_name} --data_name ${data_name} --feat_name ${feat_name} \
						      --train_data ${train_data} --dev_data ${dev_data} --test_data ${test_data} --train_set ${train_set} \
						      --dpgmm_train ${dpgmm_train} --dpgmm_dev ${dpgmm_dev} --dpgmm_test ${dpgmm_test} \
						      --seed ${seed} --K0 ${K0} `# DPGMM feature setting` \
						      --nseed ${nseed} --num_left_context ${l} --num_right_context ${r} --hybrid_batch_size ${bs} \
						      --hybrid_hidden_dim ${hd} --hybrid_num_layers ${nl} --hybrid_num_epochs ${ne} \
						      --asr_seed ${asr_seed} --gpu ${gpu} --batch_size ${batch_size} --cutoff ${cutoff} --label_smoothing ${label_smoothing} `# config asr`\
						      --lr ${lr} --num_epochs ${num_epochs} --grad_clip ${grad_clip} --factor ${factor} --patience ${patience} --save_interval ${save_interval} `# train asr`\
						      --set_uttid ${set_uttid} --search ${search} --max_target ${max_target} --beam_size ${beam_size} `# eval asr`\
						      --exp_dir ${exp_dir}

    hybrid_root=exp/hybrid2asr_mse/asr_${data_name}_${tag}_hybrid_mse_nseed${nseed}_l${l}r${r}_bs${bs}_hd${hd}_nl${nl}_ne${ne}_from_dpgmm_seed${seed}_K${K0}_with_${feat_name}
    hybrid_train=$hybrid_root/data/train/feats.scp
    hybrid_dev=$hybrid_root/data/dev/feats.scp
    hybrid_test=$hybrid_root/data/test/feats.scp
    feat1_name=${feat_name}
    feat2_name=hybrid_mse_nseed${nseed}_l${l}r${r}_bs${bs}_hd${hd}_nl${nl}_ne${ne}_from_dpgmm_seed${seed}_K${K0}_with_${feat_name}
    ./local/sandbox/asr_general/run_concatfeat2asr.sh --stage 0 --tag ${tag} --dataset_name ${dataset_name} --data_name ${data_name} \
						    --train_data ${train_data} --dev_data ${dev_data} --test_data ${test_data} --train_set ${train_set} \
						    --feat_train ${train_data}/feats.scp --feat_dev ${dev_data}/feats.scp --feat_test ${test_data}/feats.scp \
						    --feat_train2 ${hybrid_train} --feat_dev2 ${hybrid_dev} --feat_test2 ${hybrid_test} \
						    --feat1_name ${feat1_name} --feat2_name ${feat2_name} \
						    --asr_seed ${asr_seed} --gpu ${gpu} --batch_size ${batch_size} --cutoff ${cutoff} --label_smoothing ${label_smoothing} `# config asr`\
						    --lr ${lr} --num_epochs ${num_epochs} --grad_clip ${grad_clip} --factor ${factor} --patience ${patience} --save_interval ${save_interval} `# train asr`\
						    --set_uttid ${set_uttid} --search ${search} --max_target ${max_target} --beam_size ${beam_size} `# eval asr`\
						    --exp_dir ${exp_dir}
fi

if $run_bnf_asr; then
    # Run BNF ASR after feature extraction
    t=${tag};tag=${tag}_mode${mode}
    bnf_root=exp/bnf2asr/BNF_${data_name}_${tag}_BNF${bnf_dim}_from_${feat_name}
    bnf_train=$bnf_root/bnf/data_bnf/train_bnf/feats.scp
    bnf_dev=$bnf_root/bnf/data_bnf/dev_bnf/feats.scp
    bnf_test=$bnf_root/bnf/data_bnf/test_bnf/feats.scp
    bnf_dir=$bnf_root/bnf/param_bnf
    ./local/sandbox/asr_general/run_singlefeat2asr.sh --stage 0 --tag ${tag} --dataset_name ${dataset_name} --data_name ${data_name} --feat_name BNF${bnf_dim} \
						      --train_data ${train_data} --dev_data ${dev_data} --test_data ${test_data} --train_set ${train_set} \
						      --feat_train ${bnf_train} --feat_dev ${bnf_dev} --feat_test ${bnf_test} \
						      --bnf_dir ${bnf_dir} --bnf_feat true \
						      --asr_seed ${asr_seed} --gpu ${gpu} --batch_size ${batch_size} --cutoff ${cutoff} --label_smoothing ${label_smoothing} `# config asr`\
						      --lr ${lr} --num_epochs ${num_epochs} --grad_clip ${grad_clip} --factor ${factor} --patience ${patience} --save_interval ${save_interval} `# train asr`\
						      --set_uttid ${set_uttid} --search ${search} --max_target ${max_target} --beam_size ${beam_size} `# eval asr`\
						      --exp_dir ${exp_dir}

    bnf_root=exp/bnf2asr/asr_${data_name}_${tag}_single_BNF${bnf_dim}
    bnf_train=$bnf_root/data/train/feats.scp
    bnf_dev=$bnf_root/data/dev/feats.scp
    bnf_test=$bnf_root/data/test/feats.scp
    feat1_name=${feat_name}
    feat2_name=single_BNF${bnf_dim}
    ./local/sandbox/asr_general/run_concatfeat2asr.sh --stage 0 --tag ${tag} --dataset_name ${dataset_name} --data_name ${data_name} \
						    --train_data ${train_data} --dev_data ${dev_data} --test_data ${test_data} --train_set ${train_set} \
						    --feat_train ${train_data}/feats.scp --feat_dev ${dev_data}/feats.scp --feat_test ${test_data}/feats.scp \
						    --feat_train2 ${bnf_train} --feat_dev2 ${bnf_dev} --feat_test2 ${bnf_test} \
						    --feat1_name ${feat1_name} --feat2_name ${feat2_name} \
						    --asr_seed ${asr_seed} --gpu ${gpu} --batch_size ${batch_size} --cutoff ${cutoff} --label_smoothing ${label_smoothing} `# config asr`\
						    --lr ${lr} --num_epochs ${num_epochs} --grad_clip ${grad_clip} --factor ${factor} --patience ${patience} --save_interval ${save_interval} `# train asr`\
						    --set_uttid ${set_uttid} --search ${search} --max_target ${max_target} --beam_size ${beam_size} `# eval asr`\
						    --exp_dir ${exp_dir}
    tag=${t}
fi

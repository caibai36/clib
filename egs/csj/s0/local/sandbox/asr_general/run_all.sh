# general configuration
stage=8  # start from 0 if you need to start from data preparation

# Prepare some basic config files of kaldi.
./local/kaldi_conf.sh # CHECKME

# functions
dpgmm_feat_extract=false
run_dpgmm_asr=false

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

tag=asr_general
dataset_name=csj # timit # wsj
data_name=default1 # wsj0
feat_name=MFCC13 # MFCC39 # MFCC40

data_root=../s5
train_data=$data_root/data/train_nodev # /project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/wsj/s0/data/train_si84
dev_data=$data_root/data/train_dev # /project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/wsj/s0/data/test_dev93
test_data=$data_root/data/eval1 # /project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/wsj/s0/data/test_eval92
train_set=train_nodev # train_si284; name of the training data set

if [ $stage == 8 ]; then
    echo "help: ./local/sandbox/asr_general/run_all.sh --dpgmm_feat_extract true"
fi 

if $dpgmm_feat_extract; then
    # Extract DPGMM feature
    ./local/sandbox/asr_general/run_dpgmm2asr.sh --stage 0 --tag ${tag} --dataset_name ${dataset_name} --data_name ${data_name} --feat_name ${feat_name} \
					  --train_data ${train_data} --dev_data ${dev_data} --test_data ${test_data} --train_set ${train_set} \
					  --seed 123 --K0 98 \
					  --run_asr false # extract DPGMM feature without running ASR
    exit 1
fi

# if $run_dpgmm_asr; then
#     # Run DPGMM ASR after feature extraction
#     ./local/sandbox/asrs/run_dpgmm2asr.sh --stage 2 # run DPGMM ASR assuming DPGMM feature extracted \
# 					  --tag ${tag} --dataset_name ${dataset_name} --data_name ${data_name} --feat_name ${feat_name} \
# 					  --train_data ${train_data} --dev_data ${dev_data} --test_data ${test_data} --train_set ${train_set} \
# 					  --seed 98 --K0 123
#     exit 1
# fi

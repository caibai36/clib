# general configuration
stage=8  # start from 0 if you need to start from data preparation

tag=asr_general # si84_first1000

train_data=/project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/bjava/s0/data/train # /project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/wsj/s0/data/train_si84_first1000
dev_data=/project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/bjava/s0/data/dev # /project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/wsj/s0/data/test_dev93
test_data=/project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/bjava/s0/data/test # /project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/wsj/s0/data/test_eval92

# setting for BNF feature extraction
bnf_dim=42

dataset_name=bjava # wsj
data_name=default # wsj0
feat_name=MFCC39 # MFCC40

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

root=exp/bnf2asr/BNF_${data_name}_${tag}_BNF${bnf_dim}_from_${feat_name}

if [ $stage == 8 ]; then
    echo "help: ./local/sandbox/asr_general/make_bnf.sh --stage 0 --bnf_dim 42"
    exit
fi 

if [ $stage -le 1 ]; then
    # mkdir local/sandbox/asr_general/bnf
    # cp -r ../bnf/bnf42/{local,run.sh} local/sandbox/asr_general/bnf
    rm -rf $root
    mkdir -p $root
    cp -r local/sandbox/asr_general/bnf $root
    cur_dir=$PWD
    
    cd $root/bnf
    echo Enter into dir: $PWD
    mkdir -p exp/logs
    ./run.sh --stage 0 \
	     --train_data ${train_data} --dev_data ${dev_data} --test_data ${test_data} \
	     --bnf_dim ${bnf_dim} 2>&1 | tee exp/logs/make_bnf.log
    cd $cur_dir
    echo Back to dir: $PWD
fi
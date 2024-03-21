# general configuration
stage=8  # start from 0 if you need to start from data preparation

N=1000 # first n utterances

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

tag=si84_first${N}
dname=wsj0
feat_name=MFCC40

train_data=/project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/wsj/s0/data/train_si84_first${N}
dev_data=/project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/wsj/s0/data/test_dev93
test_data=/project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/wsj/s0/data/test_eval92

if [ $stage -le 1 ]; then
     utils/subset_data_dir.sh --first data/train_si84 $N data/train_si84_first${N} || exit 1
fi

if [ $stage -le 2 ]; then
    ./local/sandbox/low_resource_wsj0/run_feat2asr.sh --stage 0 --tag ${tag} --dname ${dname} --feat_name ${feat_name} --train_data ${train_data} --dev_data ${dev_data} --test_data ${test_data}
fi

if [ $stage -le 3 ]; then
    ./local/sandbox/low_resource_wsj0/run_dpgmm2asr.sh --stage 0 --tag ${tag} --dname ${dname} --feat_name ${feat_name} --train_data ${train_data} --dev_data ${dev_data} --test_data ${test_data}
fi

if [ $stage -le 4 ]; then
    local/sandbox/low_resource_wsj0/run_concatfeat2asr_FEAT_DPGMM.sh --stage 0 --tag ${tag} --dname ${dname} --feat_name ${feat_name} --train_data ${train_data} --dev_data ${dev_data} --test_data ${test_data}
fi

#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -o pipefail

# Prepare some basic config files of kaldi.
./local/kaldi_conf.sh # CHECKME
# Note: cmd.sh, path.sh are created by kaldi_conf.sh
. cmd.sh
. path.sh

# general configuration
stage=8  # start from 0 if you need to start from data preparation
feat=mfcc39 # or mfcc39_vtln
mfcc_config=conf/mfcc.conf # 13 dimensional mfcc feature of kaldi default setting

# setting for DPGMM clustering
seed=123
K0=39

num_iterations=1500
alpha=1
lmbda=1

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

# the directory of xitsonga
tso=/project/nakamura-lab01/Work/michael-h/data/nchlt_tso # CHECKME
# the directory contains wav_files and vad_files for test.
test_dir=/project/nakamura-lab08/Work/bin-wu/share/data/zr15 #CHECKME
# The directory contains tools for evaluation with abx test.
abx=/project/nakamura-lab08/Work/bin-wu/share/tools/abx # CHECKME
# evaluation environment
abx_env=/project/nakamura-lab08/Work/bin-wu/.local/miniconda3/envs/zr15 # CHECKME
# avoid conda not found error
. /project/nakamura-lab08/Work/bin-wu/.local/miniconda3/etc/profile.d/conda.sh

if [ ${stage} -le 1 ]; then
    date
    echo "Data preparation..."

    local/tso_data_prep.sh $tso $test_dir || exit 1
    date
fi

if [ ${stage} -le 2 ]; then
    date
    echo "Making 39-dimensional mfcc feature..."

    for dataset in test; do
	./local/scripts/feat_extract.sh --dataset ${dataset} --cmvn true --vtln true --mfcc_conf $mfcc_config --delta_order 2 # 39-dim mfcc
    done
    date
fi

if [ ${feat} = mfcc39 ]; then
    test=test
elif [ ${feat} = mfcc39_vtln ]; then
    test=test_vtln
fi

feat_train=data/$test/feats.scp
feat_test=data/$test/feats.scp
result=eval/abx/embedding/exp/dpgmm/${feat}_dpgmm_seed${seed}_K${K0}
if [ $stage -le 6 ]; then
    echo feat_train: $feat_train
    echo feat_test: $feat_test
    echo feat: $feat seed: $seed K0: $K0
    mkdir -p exp/logs
    rm -rf ${result}_* # ${result}_onehot and ${result}_prob by dpgmm2embedding.py

    date
    python local/dpgmm2embedding.py \
	   --feat_train=$feat_train \
	   --feat_test=$feat_test \
	   --result=$result \
	   --K0=$K0 \
	   --num_iterations=$num_iterations \
	   --seed=$seed \
	   --alpha=$alpha \
	   --lmbda=$lmbda |& tee exp/logs/run_$(basename ${result}).log
    date
fi

if [ $stage -le 7 ]; then
    date
    # result=exp/dpgmm/test2utt_mfcc13
    # task=$(basename $result)

    for task in $(basename ${result}_onehot) $(basename ${result}_prob); do
	root=exp/dpgmm/${task}
	abx_embedding=eval/abx/embedding/${root} # CHECKME
	for file in $(find $abx_embedding -name *.txt);do
	    python local/abx_add_time_col.py --frame_length_ms 25 --frame_shift_ms 10 --file $file --segment_file data/test/segments # abx tool needs to add time info at the first column
	    mv $file $(echo $file | sed -r "s:^(.*/)(.*).txt:\1nchlt_tso_\2.txt:") # add prefix. eg: 146f_0584.pos -> nchlt_tso_146f_0584.pos
	done
    done

    for task in $(basename ${result}_onehot) $(basename ${result}_prob); do
	echo "Evaluating ABX embedding..."
	root=exp/dpgmm/${task}

	# Note zr15 doesn't have edit distance evaluation
	abx_embedding=eval/abx/embedding/${root} # CHECKME
	abx_result_cos=eval/abx/result/${root}/cos
	abx_result_kl=eval/abx/result/${root}/kl

	conda activate $abx_env
	python $abx/ABXpy-zerospeech2015/bin/xitsonga_eval1.py $abx_embedding $abx_result_cos -j 5
	python $abx/ABXpy-zerospeech2015/bin/xitsonga_eval1.py -kl $abx_embedding $abx_result_kl -j 5
	conda deactivate
    done
    date
fi

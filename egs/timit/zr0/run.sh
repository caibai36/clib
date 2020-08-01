#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -o pipefail # without -u here for conda setting

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
K0=98

num_iterations=1500
alpha=1
lmbda=1

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

# the directory of timit
timit=/project/nakamura-lab01/Share/Corpora/Speech/en/TIMIT/TIMIT # CHECKME
# The directory contains tools for evaluation with abx test.
abx=/project/nakamura-lab08/Work/bin-wu/share/tools/abx # CHECKME
# evaluation environment
abx_env=/project/nakamura-lab08/Work/bin-wu/.local/miniconda3/envs/zr15 # CHECKME
# avoid conda not found error
. /project/nakamura-lab08/Work/bin-wu/.local/miniconda3/etc/profile.d/conda.sh

if [ ${stage} -le 1 ]; then
    date
    echo "Data preparation..."

    local/timit_data_prep.sh $timit || exit 1
    local/timit_prepare_dict.sh
    utils/prepare_lang.sh --sil-prob 0.0 --position-dependent-phones false --num-sil-states 3 \
			  data/local/dict "sil" data/local/lang_tmp data/lang
    local/timit_format_data.sh
    date
fi

if [ ${stage} -le 2 ]; then
    date
    echo "Making 39-dimensional mfcc feature..."

    for dataset in train dev test; do
	./local/scripts/feat_extract.sh --dataset ${dataset} --cmvn true --vtln true --mfcc_conf $mfcc_config --delta_order 2 # 39-dim mfcc
    done
    date
fi

if [ ${feat} = mfcc39 ]; then
    test=test
elif [ ${feat} = mfcc39_vtln ]; then
    test=test_vtln
fi

if [ ${stage} -le 3 ]; then
    date
    echo "Feature to Embedding..."
    feat_scp=data/$test/feats.scp
    result=eval/abx/embedding/exp/feat/$feat
    rm -rf $result

    echo "feat: $feat_scp"
    echo "result: $result"
    python local/feat2embedding.py --feat=$feat_scp --result=$result
    date
fi

if [ ${stage} -le 4 ]; then
    date
    echo "Evaluating ABX embedding..."
    abx_embedding=eval/abx/embedding/exp/feat/$feat
    abx_result_cos=eval/abx/result/exp/feat/$feat/cos
    abx_result_kl=eval/abx/result/exp/feat/$feat/kl

    for file in $(find $abx_embedding -name *.txt);do
	python local/abx_add_time_col.py --frame_length_ms 25 --frame_shift_ms 10 --file $file # abx tool needs to add time info at the first column (no segment file for timit)
    done

    conda activate $abx_env
    python $abx/ABXpy-zerospeech2015/bin/timit_eval1.py $abx_embedding $abx_result_cos -j 5 --csv
    python $abx/ABXpy-zerospeech2015/bin/timit_eval1.py -kl $abx_embedding $abx_result_kl -j 5 --csv
    conda deactivate

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
    for task in $(basename ${result}_onehot) $(basename ${result}_prob); do
	# result=exp/dpgmm/test2utt_mfcc13
	# task=$(basename $result)

	echo "Evaluating ABX embedding..."
	root=exp/dpgmm/${task}

	# Note zr15 doesn't have edit distance evaluation
	abx_embedding=eval/abx/embedding/${root} # CHECK ME
	abx_result_cos=eval/abx/result/${root}/cos
	abx_result_kl=eval/abx/result/${root}/kl

	for file in $(find $abx_embedding -name *.txt);do
	    python local/abx_add_time_col.py --frame_length_ms 25 --frame_shift_ms 10 --file $file # abx tool needs to add time info at the first column (no segment file for timit)
	done

	conda activate $abx_env
	python $abx/ABXpy-zerospeech2015/bin/timit_eval1.py $abx_embedding $abx_result_cos -j 5 --csv
	python $abx/ABXpy-zerospeech2015/bin/timit_eval1.py -kl $abx_embedding $abx_result_kl -j 5 --csv
	conda deactivate
    done
    date
fi

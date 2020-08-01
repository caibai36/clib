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

if [ ${feat} = mfcc39 ]; then
    feat_scp=data/test/feats.scp
elif [ ${feat} = mfcc39_vtln ]; then
    feat_scp=data/test_vtln/feats.scp
elif [ ${feat} = dpgmm98 ]; then
    feat_scp=exp/dpgmm2asr/asr_dpgmm_seed123_K98/data/test/feats.scp
elif [ ${feat} = cat_mfcc39_dpgmm98 ]; then
    feat_scp=exp/concatfeat2asr/asr_mfcc39_dpgmm_embedding_seed123_K98/data/test/feats.scp
else
    echo "feat should be mfcc39, mfcc39_vtln, dpgmm98 or cat_mfcc39_dpgmm98"
    exit
fi

if [ ${stage} -le 3 ]; then
    date
    echo "Feature to Embedding..."
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

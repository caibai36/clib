#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -o pipefail

# Prepare some basic config files of kaldi.
bash local/kaldi_conf.sh
# Note: cmd.sh, path.sh are created by kaldi_conf.sh
. cmd.sh
. path.sh

stage=8
seed=123
K0=60
# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

# Database for zerospeech 2019
db=/project/nakamura-lab08/Work/bin-wu/share/data/zr19/db # CHECKME downloaded by local/download.sh
# Evaluation environment
# see /project/nakamura-lab08/Work/bin-wu/share/tools/abx_2019/system/deploy/set_up_eval.sh
# or go to http://zerospeech.com/2020/instructions.html
abx_env=/project/nakamura-lab08/Work/bin-wu/.local/miniconda3/envs/eval # CHECKME

if [ $stage -le 1 ]; then
    date
    echo "Data preparation..."
    ./local/zr19_data_prep.sh --dataset test_en --audio_dir $db/english/test --vad_file $db/english/vads.txt
    ./local/zr19_data_prep.sh --dataset train_en --audio_dir $db/english/train/unit --vad_file $db/english/vads.txt
    date
fi

if [ $stage -le 2 ]; then
    date
    echo "Feature extraction..."
    ./local/scripts/feat_extract.sh --dataset test_en --cmvn true --vtln true --delta_order 2 --mfcc_conf conf/mfcc.conf --min_segment_length 0.001 # takes 3 hours
    ./local/scripts/feat_extract.sh --dataset train_en --cmvn true --vtln true --delta_order 2 --mfcc_conf conf/mfcc.conf --min_segment_length 0.001 # takes 2 hours
    date
fi

echo seed: $seed K0: $K0;
#K0=10
num_iterations=1500
alpha=1
lmbda=1
#seed=2020

feat_train=data/train_en/feats.scp
feat_test=data/test_en/feats.scp
result=eval/abx/embedding/exp/dpgmm/mfcc39_dpgmm_seed${seed}_K${K0}
if [ $stage -le 6 ]; then
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
    for task in $(basename ${result}_onehot) $(basename ${result}_prob); do
	# result=exp/dpgmm/test2utt_mfcc13
	# task=$(basename $result)
	root=exp/dpgmm/${task}

	abx_embedding=eval/abx/embedding/${root} # CHECK ME
	abx_result_cos=eval/abx/result/${root}/cos
	abx_result_kl=eval/abx/result/${root}/kl
	abx_result_edit=eval/abx/result/${root}/edit

	conda activate $abx_env
	./local/eval.sh --DIST 'cos' --EMB $abx_embedding --RES $abx_result_cos
	./local/eval.sh --DIST 'kl' --EMB $abx_embedding --RES $abx_result_kl
	./local/eval.sh --DIST 'edit' --EMB $abx_embedding --RES $abx_result_edit
    done
    date
fi

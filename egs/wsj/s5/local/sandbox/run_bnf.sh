#!/bin/bash
# Reference: ~/egs/wsj/s5/local/run_bnf.sh
# Modification: bottleneck-dim => 99
# Be careful that this script is experimental.

# Note: In order to run BNF, run run_bnf.sh
. ./path.sh
. ./cmd.sh

set -e
set -o pipefail
set -u

bnf_train_stage=-100
run_cmd="$train_cmd" # "$cuda_cmd" for gpu, "" for empty spaces in the variable

train_data_dir=data/train_si284
dev_data_dir=data/test_dev93
test_data_dir=data/test_eval92
align_dir=exp/tri4b_ali_si284 # transformed with lda, and etc.
dev_align_dir=exp/tri4b/decode_bd_tgpr_dev93 # transformed with lda, and etc.
test_align_dir=exp/tri4b/decode_bd_tgpr_eval92 # transformed with lda, and etc.
data_train_bnf=data_bnf/train_bnf
data_dev_bnf=data_bnf/dev93_bnf
data_test_bnf=data_bnf/eval92_bnf
exp_dir=exp_bnf
bnf_exp_dir=$exp_dir/tri6_bnf

. utils/parse_options.sh

if [ ! -f $bnf_exp_dir/.done ]; then
  mkdir -p $exp_dir
  mkdir -p $bnf_exp_dir
  echo ---------------------------------------------------------------------
  echo "Starting training the bottleneck network"
  echo ---------------------------------------------------------------------
  steps/nnet2/train_tanh_bottleneck.sh \
    --stage $bnf_train_stage --num-jobs-nnet 4 \
    --num-threads 1 --mix-up 5000 --max-change 40 \
    --minibatch-size 512 \
    --initial-learning-rate 0.005 \
    --final-learning-rate 0.0005 \
    --num-hidden-layers 5 \
    --bottleneck-dim 99 --hidden-layer-dim 1024 --cmd "$run_cmd" \
    $train_data_dir data/lang $align_dir $bnf_exp_dir || exit 1 
  touch $bnf_exp_dir/.done
fi

[ ! -d param_bnf ] && mkdir -p param_bnf
if [ ! -f $data_train_bnf/.done ]; then
  mkdir -p data_bnf
  # put the archives in param_bnf/.
  steps/nnet2/dump_bottleneck_features.sh --cmd "$train_cmd" \
    --transform-dir $align_dir $train_data_dir $data_train_bnf $bnf_exp_dir param_bnf $exp_dir/dump_bnf
  touch $data_train_bnf/.done
fi 

[ ! -d $test_data_dir ] && echo "No such directory $test_data_dir" && exit 1;
[ ! -d $dev_data_dir ] && echo "No such directory $dev_data_dir" && exit 1;
[ ! -d $test_align_dir ] && echo "No such directory $test_align_dir" && exit 1;
[ ! -d $dev_align_dir ] && echo "No such directory $dev_align_dir" && exit 1;
# put the archives in param_bnf/.
steps/nnet2/dump_bottleneck_features.sh --nj 8 \
  --transform-dir $test_align_dir $test_data_dir $data_test_bnf $bnf_exp_dir param_bnf $exp_dir/dump_bnf

steps/nnet2/dump_bottleneck_features.sh --nj 10 \
  --transform-dir $dev_align_dir $dev_data_dir $data_dev_bnf $bnf_exp_dir param_bnf $exp_dir/dump_bnf



# if [ ! data_bnf/train/.done -nt $data_train_bnf/.done ]; then
#   steps/nnet/make_fmllr_feats.sh --cmd "$train_cmd --max-jobs-run 10" \
#      --transform-dir $align_dir  data_bnf/train_sat $train_data_dir \
#     exp/tri4b $exp_dir/make_fmllr_feats/log param_bnf/ 

#   steps/append_feats.sh --cmd "$train_cmd" --nj 4 \
#     $data_train_bnf data_bnf/train_sat data_bnf/train \
#     $exp_dir/append_feats/log param_bnf/ 
#   steps/compute_cmvn_stats.sh --fake data_bnf/train $exp_dir/make_fmllr_feats param_bnf
#   rm -r data_bnf/train_sat

#   touch data_bnf/train/.done
# fi
# ## preparing Bottleneck features for eval92 and dev93
# steps/nnet/make_fmllr_feats.sh \
#   --nj 8 --transform-dir $test_align_dir data_bnf/eval92_sat $test_data_dir \
#   $align_dir $exp_dir/make_fmllr_feats/log param_bnf/ 
# steps/nnet/make_fmllr_feats.sh \
#   --nj 10 --transform-dir $dev_align_dir data_bnf/dev93_sat $dev_data_dir \
#   $align_dir $exp_dir/make_fmllr_feats/log param_bnf/ 

# steps/append_feats.sh --nj 4 \
#   $data_test_bnf data_bnf/eval92_sat data_bnf/eval92 \
#   $exp_dir/append_feats/log param_bnf/ 
# steps/append_feats.sh --nj 4 \
#   $data_dev_bnf data_bnf/dev93_sat data_bnf/dev93 \
#   $exp_dir/append_feats/log param_bnf/ 
  
# steps/compute_cmvn_stats.sh --fake data_bnf/eval92 $exp_dir/make_fmllr_feats param_bnf
# steps/compute_cmvn_stats.sh --fake data_bnf/dev93 $exp_dir/make_fmllr_feats param_bnf

# rm -r data_bnf/eval92_sat
# rm -r data_bnf/dev93_sat

exit 0;

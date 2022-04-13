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

stage=0
bnf_train_stage=-100
run_cmd="$train_cmd" # "$cuda_cmd" for gpu, "" for empty spaces in the variable

train_data_dir=data/train_mb
dev_data_dir=data/dev_mb
test_data_dir=data/test_mb
align_dir=exp/tri3_ali
dev_align_dir=exp/tri3/decode_dev
test_align_dir=exp/tri3/decode_test
model_dir=exp/tri3
data_train_bnf=data_bnf/train_bnf
data_dev_bnf=data_bnf/dev_bnf
data_test_bnf=data_bnf/test_bnf
exp_dir=exp_bnf
bnf_exp_dir=$exp_dir/tri6_bnf
nj=3 # number of speakers is 3

. utils/parse_options.sh
if [ $stage -le 1 ]; then
    if [ ! -f $bnf_exp_dir/.done ]; then
	mkdir -p $exp_dir
	mkdir -p $bnf_exp_dir
	echo ---------------------------------------------------------------------
	echo "Starting training the bottleneck network"
	echo ---------------------------------------------------------------------
	steps/nnet2/train_tanh_bottleneck.sh \
	    --stage $bnf_train_stage --num-jobs-nnet $nj \
	    --num-threads 10 --mix-up 5000 --max-change 40 \
	    --minibatch-size 512 \
	    --initial-learning-rate 0.005 \
	    --final-learning-rate 0.0005 \
	    --num-hidden-layers 5 \
	    --bottleneck-dim 42 --hidden-layer-dim 1024 --cmd "$run_cmd" \
	    $train_data_dir data/lang $align_dir $bnf_exp_dir || exit 1
	touch $bnf_exp_dir/.done
    fi
fi

if [ $stage -le 2 ]; then
    [ ! -d param_bnf ] && mkdir -p param_bnf
    if [ ! -f $data_train_bnf/.done ]; then
	mkdir -p data_bnf
	# put the archives in param_bnf/.
	steps/nnet2/dump_bottleneck_features.sh --cmd "$train_cmd" \
	      --nj $nj --transform-dir $align_dir $train_data_dir $data_train_bnf $bnf_exp_dir param_bnf $exp_dir/dump_bnf
	touch $data_train_bnf/.done
    fi

    steps/nnet2/dump_bottleneck_features.sh --nj $nj \
	      --transform-dir $test_align_dir $test_data_dir $data_test_bnf $bnf_exp_dir param_bnf $exp_dir/dump_bnf

    steps/nnet2/dump_bottleneck_features.sh --nj $nj \
	      --transform-dir $dev_align_dir $dev_data_dir $data_dev_bnf $bnf_exp_dir param_bnf $exp_dir/dump_bnf

# [ ! -d data/test_eval92 ] && echo "No such directory data/test_eval92" && exit 1;
# [ ! -d data/test_dev93 ] && echo "No such directory data/test_dev93" && exit 1;
# [ ! -d exp/tri4b/decode_bd_tgpr_eval92 ] && echo "No such directory exp/tri4b/decode_bd_tgpr_eval92" && exit 1;
# [ ! -d exp/tri4b/decode_bd_tgpr_dev93 ] && echo "No such directory exp/tri4b/decode_bd_tgpr_dev93" && exit 1;
# # put the archives in param_bnf/.
# steps/nnet2/dump_bottleneck_features.sh --nj 8 \
#   --transform-dir exp/tri4b/decode_bd_tgpr_eval92 data/test_eval92 data_bnf/eval92_bnf $bnf_exp_dir param_bnf $exp_dir/dump_bnf

# steps/nnet2/dump_bottleneck_features.sh --nj 10 \
#   --transform-dir exp/tri4b/decode_bd_tgpr_dev93 data/test_dev93 data_bnf/dev93_bnf $bnf_exp_dir param_bnf $exp_dir/dump_bnf
fi

if [ $stage -le 3 ]; then
    if [ ! data_bnf/train/.done -nt $data_train_bnf/.done ]; then
	steps/nnet/make_fmllr_feats.sh --cmd "$train_cmd --max-jobs-run 10" \
	      --nj $nj --transform-dir $align_dir  data_bnf/train_sat $train_data_dir \
	      $model_dir $exp_dir/make_fmllr_feats/log param_bnf/

	steps/append_feats.sh --cmd "$train_cmd" --nj $nj \
	      $data_train_bnf data_bnf/train_sat data_bnf/train \
	      $exp_dir/append_feats/log param_bnf/
	steps/compute_cmvn_stats.sh --fake data_bnf/train $exp_dir/make_fmllr_feats param_bnf
	rm -r data_bnf/train_sat

	touch data_bnf/train/.done
    fi
    # preparing Bottleneck features for test and dev
    steps/nnet/make_fmllr_feats.sh \
	--nj $nj --transform-dir $test_align_dir data_bnf/test_sat $test_data_dir \
	$align_dir $exp_dir/make_fmllr_feats/log param_bnf/
    steps/nnet/make_fmllr_feats.sh \
	--nj $nj --transform-dir $dev_align_dir data_bnf/dev_sat $dev_data_dir \
	$align_dir $exp_dir/make_fmllr_feats/log param_bnf/

    steps/append_feats.sh --nj $nj \
			  $data_test_bnf data_bnf/test_sat data_bnf/test \
			  $exp_dir/append_feats/log param_bnf/
    steps/append_feats.sh --nj $nj \
			  $data_dev_bnf data_bnf/dev_sat data_bnf/dev \
			  $exp_dir/append_feats/log param_bnf/

    steps/compute_cmvn_stats.sh --fake data_bnf/test $exp_dir/make_fmllr_feats param_bnf
    steps/compute_cmvn_stats.sh --fake data_bnf/dev $exp_dir/make_fmllr_feats param_bnf

    rm -r data_bnf/test_sat
    rm -r data_bnf/dev_sat
fi

# if [ ! data_bnf/train/.done -nt data_bnf/train_bnf/.done ]; then
#   steps/nnet/make_fmllr_feats.sh --cmd "$train_cmd --max-jobs-run 10" \
#      --transform-dir $align_dir  data_bnf/train_sat $train_data_dir \
#     exp/tri4b $exp_dir/make_fmllr_feats/log param_bnf/

#   steps/append_feats.sh --cmd "$train_cmd" --nj 4 \
#     data_bnf/train_bnf data_bnf/train_sat data_bnf/train \
#     $exp_dir/append_feats/log param_bnf/
#   steps/compute_cmvn_stats.sh --fake data_bnf/train $exp_dir/make_fmllr_feats param_bnf
#   rm -r data_bnf/train_sat

#   touch data_bnf/train/.done
# fi
# ## preparing Bottleneck features for eval92 and dev93
# steps/nnet/make_fmllr_feats.sh \
#   --nj 8 --transform-dir exp/tri4b/decode_bd_tgpr_eval92 data_bnf/eval92_sat data/test_eval92 \
#   $align_dir $exp_dir/make_fmllr_feats/log param_bnf/
# steps/nnet/make_fmllr_feats.sh \
#   --nj 10 --transform-dir exp/tri4b/decode_bd_tgpr_dev93 data_bnf/dev93_sat data/test_dev93 \
#   $align_dir $exp_dir/make_fmllr_feats/log param_bnf/

# steps/append_feats.sh --nj 4 \
#   data_bnf/eval92_bnf data_bnf/eval92_sat data_bnf/eval92 \
#   $exp_dir/append_feats/log param_bnf/
# steps/append_feats.sh --nj 4 \
#   data_bnf/dev93_bnf data_bnf/dev93_sat data_bnf/dev93 \
#   $exp_dir/append_feats/log param_bnf/

# steps/compute_cmvn_stats.sh --fake data_bnf/eval92 $exp_dir/make_fmllr_feats param_bnf
# steps/compute_cmvn_stats.sh --fake data_bnf/dev93 $exp_dir/make_fmllr_feats param_bnf

# rm -r data_bnf/eval92_sat
# rm -r data_bnf/dev93_sat

exit 0;

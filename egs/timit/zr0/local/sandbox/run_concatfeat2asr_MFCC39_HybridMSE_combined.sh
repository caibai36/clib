# setting for hybird LSTM neural network
l=0
r=0
nseed=123
exp_dir=exp/tmp_hybrid_asr # for extra experiments

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

./local/sandbox/run_hybrid2asr_mse2.sh --stage 0 --nseed ${nseed} --num_left_context ${l} --num_right_context ${r} --exp_dir ${exp_dir} 2>&1 |  tee exp/logs/run_hybrid2asr_mse_l8r8.log
./local/sandbox/run_concatfeat2asr_MFCC39_HybridMSElXrX.sh --stage 0 --nseed ${nseed} --num_left_context ${l} --num_right_context ${r} --exp_dir ${exp_dir} 2>&1 |  tee exp/logs/run_concatfeat2asr_mfcc39_hybridmsel8r8.log

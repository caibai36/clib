# setting for hybird LSTM neural network
l=0
r=0
seed=123
nseed=123
bs=256
hd=512
nl=3
ne=20
exp_dir=exp/tmp_hybrid_asr_general_wsj1 # for extra experiments

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

./local/sandbox/hybrid_general_wsj1/run_hybrid2asr_mse_wsj1_general.sh --stage 0 --seed ${seed} --nseed ${nseed} --num_left_context ${l} --num_right_context ${r} --hybrid_batch_size ${bs} --hybrid_hidden_dim ${hd} --hybrid_num_layers ${nl} --hybrid_num_epochs ${ne} --exp_dir ${exp_dir} 2>&1 |  tee exp/logs/run_hybrid2asr_mse_wsj1_l${l}r${r}s${seed}ns${nseed}_bs${bs}_hd${hd}_nl${nl}_ne${ne}.log
./local/sandbox/hybrid_general_wsj1/run_concatfeat2asr_wsj1_MFCC40_HybridMSElXrX_general.sh --stage 0 --seed ${seed} --nseed ${nseed} --num_left_context ${l} --num_right_context ${r} --hybrid_batch_size ${bs} --hybrid_hidden_dim ${hd} --hybrid_num_layers ${nl} --hybrid_num_epochs ${ne} --exp_dir ${exp_dir} 2>&1 |  tee exp/logs/run_concatfeat2asr_wsj1_mfcc40_hybridmse_l${l}r${r}s${seed}ns${nseed}_bs${bs}_hd${hd}_nl${nl}_ne${ne}.log

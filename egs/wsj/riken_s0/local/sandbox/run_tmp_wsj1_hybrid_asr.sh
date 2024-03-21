./local/sandbox/run_hybrid2asr_mse_wsj1.sh --stage 0 2>&1 | tee exp/logs/run_hybrid2asr_wsj1_mse_l0r0.log
./local/sandbox/run_hybrid2asr_mse_wsj1.sh --stage 0 --num_left_context 16 2>&1 | tee exp/logs/run_hybrid2asr_wsj1_mse_l16r0.log
#./local/sandbox/run_hybrid2asr_mse_wsj1.sh --stage 0 --num_left_context 8 --num_right_context 8 2>&1 | tee exp/logs/run_hybrid2asr_wsj1_mse_l8r8.log
#./local/sandbox/run_hybrid2asr_mse_wsj1.sh --stage 0 --num_left_context 4 --num_right_context 4 2>&1 | tee exp/logs/run_hybrid2asr_wsj1_mse_l4r4.log
./local/sandbox/run_concatfeat2asr_wsj1_MFCC40_HybridMSEl0r0.sh --stage 0  2>&1 | tee exp/logs/run_concatfeat2asr_wsj1_mfcc40_hybridmsel0r0.log
./local/sandbox/run_concatfeat2asr_wsj1_MFCC40_HybridMSEl16r0.sh --stage 0  2>&1 | tee exp/logs/run_concatfeat2asr_wsj1_mfcc40_hybridmsel16r0.log
#./local/sandbox/run_concatfeat2asr_wsj1_MFCC40_HybridMSEl8r8.sh --stage 0  2>&1 | tee exp/logs/run_concatfeat2asr_wsj1_mfcc40_hybridmsel8r8.log
#./local/sandbox/run_concatfeat2asr_wsj1_MFCC40_HybridMSEl4r4.sh --stage 0  2>&1 | tee exp/logs/run_concatfeat2asr_wsj1_mfcc40_hybridmsel4r4.log

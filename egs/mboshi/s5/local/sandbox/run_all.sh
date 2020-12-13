mkdir -p exp/logs
# ./run.sh --stage 0 2>&1 | tee exp/logs/run_asr.log
./local/sandbox/run_bnf.sh --stage 0 2>&1 | tee exp/logs/run_bnf.log

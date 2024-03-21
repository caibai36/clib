for N in 3000 3500; do
    date
    echo "dpgmm2asr training with first $N utterances"
    ./local/sandbox/low_resource_wsj0/tmp_run/run_low_resource_dpgmm2asr.sh --N $N --stage 0 2>&1 | tee exp/logs/run_low_resource_dpgmm2asr_${N}.log
done

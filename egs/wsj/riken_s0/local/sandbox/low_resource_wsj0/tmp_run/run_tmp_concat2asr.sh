for N in 3000 3500 4000 5000 6000; do
    date
    echo "concat2asr training with first $N utterances"
    ./local/sandbox/low_resource_wsj0/tmp_run/run_low_resource_concat2asr.sh --N $N --stage 0 2>&1 | tee exp/logs/run_low_resource_concat2asr_${N}.log
done

dir=

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

if [ -z $dir ]; then
    echo "./local/eval_nopunct.sh --dir \$dir"
    echo "Remove the punctuations of <space> <comma> <period> and <unk>"
    echo
    echo "e.g., ./local/eval_nopunct.sh --dir exp/asr_pretrained/yonden/default_csj_pretrained_with_yonden_pretrained_train.train_data3-36_remove_4_6_28_36_ampnorm_mel80_dev.dev_data4_28_ampnorm_mel80_test.test_data6_ampnorm_mel80/EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp-run0/batchsize32_cutoff1600_labelsmoothing0.05_lr0.001_gradclip5_factor0.5_patience3/eval/beamsize10"
    echo
    exit 1
fi

pushd .
cd $dir
cat hypo_char.txt | sed 's/ <space> / /g' | sed 's/ <comma> / /g' | sed 's/ <period>$//g' | sed 's/ <unk> / /g' | sed 's/ <period> / /g' > hypo_char_nopunct.txt
cat ref_char.txt | sed 's/ <space> / /g' | sed 's/ <comma> / /g' | sed 's/ <period>$//g'  | sed 's/ <unk> / /g' | sed 's/ <period> / /g' > ref_char_nopunct.txt
cmd=$(head cer.txt | sed -e 's/hypo_char.txt/hypo_char_nopunct.txt/g' -e 's/ref_char.txt/ref_char_nopunct.txt/g'|head -1)
popd
eval $cmd

COMPUTE_WER=/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi/src/bin/compute-wer

feat=mfcc40
dataset_name=wsj
data_name=wsj1  # train:si284;dev:dev93;test:eval92
set_uttid=None # subset of testing data (e.g. set_uttid=conf/data/test_small/set_uttid.txt)

# model_path=exp/tmp/wsj/wsj1/EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp-run0/mfcc40_batchsize32_cutoff1600_labelsmoothing0.05_lr0.001_gradclip5_factor0.5_patience3/train/model_e30.mdl
model_path=exp/tmp/wsj/wsj1/EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp-run0/mfcc40_batchsize32_cutoff1600_labelsmoothing0.05_lr0.001_gradclip5_factor0.5_patience3/train/best_model.mdl

gpu=auto
batch_size=2 # result is independent of batch_size

search=beam
max_target=250 # the maximum length of the decoded sequence
beam_size=10

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

data_config=conf/data/${dataset_name}/${data_name}/asr_tts/data_${feat}.yaml
result_dir=${model_path%/train/*}/eval/beamsize${beam_size} # ${string%substring} # Deletes shortest match of $substring from back of $string.

python local/scripts/eval_asr.py \
       --gpu $gpu \
       --data_config $data_config \
       --set_uttid $set_uttid \
       --batch_size $batch_size \
       --model $model_path \
       --max_target $max_target \
       --search $search \
       --beam_size $beam_size \
       --result $result_dir

echo
echo "Computing character error rate (CER)..."
$COMPUTE_WER --mode=present ark,t:${result_dir}/ref_char.txt ark,t:${result_dir}/hypo_char.txt |& tee ${result_dir}/cer.txt
echo
echo "Computing word error rate (WER)..."
$COMPUTE_WER --mode=present ark,t:${result_dir}/ref_word.txt ark,t:${result_dir}/hypo_word.txt |& tee ${result_dir}/wer.txt

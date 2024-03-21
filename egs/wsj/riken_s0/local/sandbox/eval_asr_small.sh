exp_dir=exp/tmp
model_name=test_small_att
result_dir=${exp_dir}/${model_name}/eval

data_config=conf/data/test_small/data.yaml
# set_uttid=conf/data/test_small/set_uttid.txt
set_uttid=None

# model_path=${exp_dir}/${model_name}/train/best_model.mdl
model_path=conf/data/test_small/pretrained_model/model_e2000.mdl

python local/sandbox/eval_asr.py \
       --gpu 0 \
       --data_config $data_config \
       --set_uttid $set_uttid \
       --batch_size 3 \
       --model $model_path \
       --search beam --beam_size 2 \
       --max_target 4 \
       --result $result_dir

. path.sh # /project/nakamura-lab08/Work/bin-wu/share/tools/kaldi/src/bin/compute-wer
echo
echo "Computing character error rate (CER)..."
compute-wer --mode=present ark,t:${result_dir}/ref_char.txt ark,t:${result_dir}/hypo_char.txt
echo
echo "Computing word error rate (WER)..."
compute-wer --mode=present ark,t:${result_dir}/ref_word.txt ark,t:${result_dir}/hypo_word.txt


exp_dir=exp/tmp
model_name=test_small_att
result_dir=${exp_dir}/${model_name}/train

# python local/sandbox/train_asr.py \
#        --data_config conf/data/test_small/data.yaml \
#        --model_config conf/data/test_small/model.yaml \
#        --num_epochs 10 \
#        --batch_size 2 \
#        --lr 0.001 \
#        --reducelr '{"factor":0.5, "patience":3}' \
#        --gpu 0 \
#        --cutoff -1 \
#        --label_smoothing 0 \
#        --grad_clip 10 \
#        --result $result_dir

python local/sandbox/train_asr.py \
       --gpu 0 \
       --data_config conf/data/test_small/data.yaml \
       --batch_size 2 \
       --cutoff -1 \
       --model_config conf/data/test_small/model.yaml \
       --label_smoothing 0 \
       --lr 0.001 \
       --reducelr '{"factor":0.5, "patience":3}' \
       --num_epochs 10 \
       --grad_clip 10 \
       --result $result_dir

# b1_f1
python local/sandbox/filter_15w_raw_spec.py \
    --input /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data/division_nas5_b1_f1_mae_vit_labels_for_mae_pretrained_all_days/train_input.npy \
    --mask exp/mae_feat_dim_reduction/python_script_nas5/b1_f1_15weeks/mask_b1_f1_15weeks.txt \
    --output /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data_15w/b1_f1_15weeks/spec_raw.npy |& tee logs/filter_15w_raw_spec_nas5_b1_f1.log

# b1_f2
python local/sandbox/filter_15w_raw_spec.py \
    --input /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data/division_nas5_b1_f2_mae_vit_labels_for_mae_pretrained_all_days/train_input.npy \
    --mask exp/mae_feat_dim_reduction/python_script_nas5/b1_f2_15weeks/mask_b1_f2_15weeks.txt \
    --output /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data_15w/b1_f2_15weeks/spec_raw.npy |& tee logs/filter_15w_raw_spec_nas5_b1_f2.log

# b2_f1 (default)
python local/sandbox/filter_15w_raw_spec.py \
    --input /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data/division_nas5_b2_f1_mae_vit_labels_for_mae_pretrained_all_days/train_input.npy \
    --mask exp/mae_feat_dim_reduction/python_script_nas5/b2_f1_15weeks/mask_b2_f1_15weeks.txt \
    --output /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data_15w/b2_f1_15weeks/spec_raw.npy |& tee logs/filter_15w_raw_spec_nas5_b2_f1.log

# b2_f2
python local/sandbox/filter_15w_raw_spec.py \
    --input /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data/division_nas5_b2_f2_mae_vit_labels_for_mae_pretrained_all_days/train_input.npy \
    --mask exp/mae_feat_dim_reduction/python_script_nas5/b2_f2_15weeks/mask_b2_f2_15weeks.txt \
    --output /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data_15w/b2_f2_15weeks/spec_raw.npy |& tee logs/filter_15w_raw_spec_nas5_b2_f2.log

# b3_f1
python local/sandbox/filter_15w_raw_spec.py \
    --input /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data/division_nas5_b3_f1_mae_vit_labels_for_mae_pretrained_all_days/train_input.npy \
    --mask exp/mae_feat_dim_reduction/python_script_nas5/b3_f1_15weeks/mask_b3_f1_15weeks.txt \
    --output /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data_15w/b3_f1_15weeks/spec_raw.npy |& tee logs/filter_15w_raw_spec_nas5_b3_f1.log

# b3_f2
python local/sandbox/filter_15w_raw_spec.py \
    --input /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data/division_nas5_b3_f2_mae_vit_labels_for_mae_pretrained_all_days/train_input.npy \
    --mask exp/mae_feat_dim_reduction/python_script_nas5/b3_f2_15weeks/mask_b3_f2_15weeks.txt \
    --output /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data_15w/b3_f2_15weeks/spec_raw.npy |& tee logs/filter_15w_raw_spec_nas5_b3_f2.log

# b4_f1
python local/sandbox/filter_15w_raw_spec.py \
    --input /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data/division_nas5_b4_f1_mae_vit_labels_for_mae_pretrained_all_days/train_input.npy \
    --mask exp/mae_feat_dim_reduction/python_script_nas5/b4_f1_15weeks/mask_b4_f1_15weeks.txt \
    --output /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data_15w/b4_f1_15weeks/spec_raw.npy |& tee logs/filter_15w_raw_spec_nas5_b4_f1.log

# b4_f2
python local/sandbox/filter_15w_raw_spec.py \
    --input /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data/division_nas5_b4_f2_mae_vit_labels_for_mae_pretrained_all_days/train_input.npy \
    --mask exp/mae_feat_dim_reduction/python_script_nas5/b4_f2_15weeks/mask_b4_f2_15weeks.txt \
    --output /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data_15w/b4_f2_15weeks/spec_raw.npy |& tee logs/filter_15w_raw_spec_nas5_b4_f2.log

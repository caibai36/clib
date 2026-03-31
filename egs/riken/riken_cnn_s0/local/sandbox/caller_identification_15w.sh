# b1_f1
python local/sandbox/sandbox/caller_identification_run_b1_infant_b3_adult.py \
    --gpu 1 \
    --eval_feature /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data_15w/b1_f1_15weeks/spec_raw.npy \
    --eval_config exp/mae_feat_dim_reduction/python_script_nas5/b1_f1_15weeks/info_mae.csv \
    --eval_output_config exp/mae_feat_dim_reduction/python_script_nas5/b1_f1_15weeks/info_mae_predicted_speaker.csv \
    --eval_model exp/caller_identification_exp/train_b1_infant_b3_adult_ver2/model_best_dev_backup.ckpt |& tee logs/extract_predicted_speaker_b1_f1.log

# b1_f2
python local/sandbox/sandbox/caller_identification_run_b1_infant_b3_adult.py \
    --gpu 1 \
    --eval_feature /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data_15w/b1_f2_15weeks/spec_raw.npy \
    --eval_config exp/mae_feat_dim_reduction/python_script_nas5/b1_f2_15weeks/info_mae.csv \
    --eval_output_config exp/mae_feat_dim_reduction/python_script_nas5/b1_f2_15weeks/info_mae_predicted_speaker.csv \
    --eval_model exp/caller_identification_exp/train_b1_infant_b3_adult_ver2/model_best_dev_backup.ckpt |& tee logs/extract_predicted_speaker_b1_f2.log

# b2_f1
python local/sandbox/sandbox/caller_identification_run_b1_infant_b3_adult.py \
    --gpu 1 \
    --eval_feature /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data_15w/b2_f1_15weeks/spec_raw.npy \
    --eval_config exp/mae_feat_dim_reduction/python_script_nas5/b2_f1_15weeks/info_mae.csv \
    --eval_output_config exp/mae_feat_dim_reduction/python_script_nas5/b2_f1_15weeks/info_mae_predicted_speaker.csv \
    --eval_model exp/caller_identification_exp/train_b1_infant_b3_adult_ver2/model_best_dev_backup.ckpt |& tee logs/extract_predicted_speaker_b2_f1.log

# b2_f2
python local/sandbox/sandbox/caller_identification_run_b1_infant_b3_adult.py \
    --gpu 1 \
    --eval_feature /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data_15w/b2_f2_15weeks/spec_raw.npy \
    --eval_config exp/mae_feat_dim_reduction/python_script_nas5/b2_f2_15weeks/info_mae.csv \
    --eval_output_config exp/mae_feat_dim_reduction/python_script_nas5/b2_f2_15weeks/info_mae_predicted_speaker.csv \
    --eval_model exp/caller_identification_exp/train_b1_infant_b3_adult_ver2/model_best_dev_backup.ckpt |& tee logs/extract_predicted_speaker_b2_f2.log

# b3_f1
python local/sandbox/sandbox/caller_identification_run_b1_infant_b3_adult.py \
    --gpu 1 \
    --eval_feature /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data_15w/b3_f1_15weeks/spec_raw.npy \
    --eval_config exp/mae_feat_dim_reduction/python_script_nas5/b3_f1_15weeks/info_mae.csv \
    --eval_output_config exp/mae_feat_dim_reduction/python_script_nas5/b3_f1_15weeks/info_mae_predicted_speaker.csv \
    --eval_model exp/caller_identification_exp/train_b1_infant_b3_adult_ver2/model_best_dev_backup.ckpt |& tee logs/extract_predicted_speaker_b3_f1.log

# b3_f2
python local/sandbox/sandbox/caller_identification_run_b1_infant_b3_adult.py \
    --gpu 1 \
    --eval_feature /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data_15w/b3_f2_15weeks/spec_raw.npy \
    --eval_config exp/mae_feat_dim_reduction/python_script_nas5/b3_f2_15weeks/info_mae.csv \
    --eval_output_config exp/mae_feat_dim_reduction/python_script_nas5/b3_f2_15weeks/info_mae_predicted_speaker.csv \
    --eval_model exp/caller_identification_exp/train_b1_infant_b3_adult_ver2/model_best_dev_backup.ckpt |& tee logs/extract_predicted_speaker_b3_f2.log

# b4_f1
python local/sandbox/sandbox/caller_identification_run_b1_infant_b3_adult.py \
    --gpu 1 \
    --eval_feature /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data_15w/b4_f1_15weeks/spec_raw.npy \
    --eval_config exp/mae_feat_dim_reduction/python_script_nas5/b4_f1_15weeks/info_mae.csv \
    --eval_output_config exp/mae_feat_dim_reduction/python_script_nas5/b4_f1_15weeks/info_mae_predicted_speaker.csv \
    --eval_model exp/caller_identification_exp/train_b1_infant_b3_adult_ver2/model_best_dev_backup.ckpt |& tee logs/extract_predicted_speaker_b4_f1.log

# b4_f2
python local/sandbox/sandbox/caller_identification_run_b1_infant_b3_adult.py \
    --gpu 1 \
    --eval_feature /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data_15w/b4_f2_15weeks/spec_raw.npy \
    --eval_config exp/mae_feat_dim_reduction/python_script_nas5/b4_f2_15weeks/info_mae.csv \
    --eval_output_config exp/mae_feat_dim_reduction/python_script_nas5/b4_f2_15weeks/info_mae_predicted_speaker.csv \
    --eval_model exp/caller_identification_exp/train_b1_infant_b3_adult_ver2/model_best_dev_backup.ckpt |& tee logs/extract_predicted_speaker_b4_f2.log

# b1_f1
python local/riken_mae_pretrained_feature_extraction.py --gpu 0 --input_file /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data/division_nas5_b1_f1_mae_vit_labels_for_mae_pretrained_all_days/train_input.npy --pretrained_path exp/mae_pretraining/nas5_b2_mae_pretrained_48days/base/model_epoch_400_base_48days.pt --output_dir exp/mae_feat/division_nas5_b1_f1_mae_vit_labels_for_mae_pretrained_all_days |& tee logs/b1_f1_mae_feat_extraction.log

# b2_f1
python local/riken_mae_pretrained_feature_extraction.py --gpu 0 --input_file /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data/division_nas5_b2_f1_mae_vit_labels_for_mae_pretrained_all_days/train_input.npy --pretrained_path exp/mae_pretraining/nas5_b2_mae_pretrained_48days/base/model_epoch_400_base_48days.pt --output_dir exp/mae_feat/division_nas5_b2_f1_mae_vit_labels_for_mae_pretrained_all_days |& tee logs/b2_f1_mae_feat_extraction.log

# b3_f1
python local/riken_mae_pretrained_feature_extraction.py --gpu 0 --input_file /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data/division_nas5_b3_f1_mae_vit_labels_for_mae_pretrained_all_days/train_input.npy --pretrained_path exp/mae_pretraining/nas5_b2_mae_pretrained_48days/base/model_epoch_400_base_48days.pt --output_dir exp/mae_feat/division_nas5_b3_f1_mae_vit_labels_for_mae_pretrained_all_days |& tee logs/b3_f1_mae_feat_extraction.log

# b4_f1
python local/riken_mae_pretrained_feature_extraction.py --gpu 0 --input_file /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data/division_nas5_b4_f1_mae_vit_labels_for_mae_pretrained_all_days/train_input.npy --pretrained_path exp/mae_pretraining/nas5_b2_mae_pretrained_48days/base/model_epoch_400_base_48days.pt --output_dir exp/mae_feat/division_nas5_b4_f1_mae_vit_labels_for_mae_pretrained_all_days |& tee logs/b4_f1_mae_feat_extraction.log

# b1_f2
python local/riken_mae_pretrained_feature_extraction.py --gpu 0 --input_file /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data/division_nas5_b1_f2_mae_vit_labels_for_mae_pretrained_all_days/train_input.npy --pretrained_path exp/mae_pretraining/nas5_b2_mae_pretrained_48days/base/model_epoch_400_base_48days.pt --output_dir exp/mae_feat/division_nas5_b1_f2_mae_vit_labels_for_mae_pretrained_all_days |& tee logs/b1_f2_mae_feat_extraction.log

# b2_f2
python local/riken_mae_pretrained_feature_extraction.py --gpu 0 --input_file /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data/division_nas5_b2_f2_mae_vit_labels_for_mae_pretrained_all_days/train_input.npy --pretrained_path exp/mae_pretraining/nas5_b2_mae_pretrained_48days/base/model_epoch_400_base_48days.pt --output_dir exp/mae_feat/division_nas5_b2_f2_mae_vit_labels_for_mae_pretrained_all_days |& tee logs/b2_f2_mae_feat_extraction.log

# b3_f2
python local/riken_mae_pretrained_feature_extraction.py --gpu 0 --input_file /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data/division_nas5_b3_f2_mae_vit_labels_for_mae_pretrained_all_days/train_input.npy --pretrained_path exp/mae_pretraining/nas5_b2_mae_pretrained_48days/base/model_epoch_400_base_48days.pt --output_dir exp/mae_feat/division_nas5_b3_f2_mae_vit_labels_for_mae_pretrained_all_days |& tee logs/b3_f2_mae_feat_extraction.log

# b4_f2
python local/riken_mae_pretrained_feature_extraction.py --gpu 0 --input_file /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data/division_nas5_b4_f2_mae_vit_labels_for_mae_pretrained_all_days/train_input.npy --pretrained_path exp/mae_pretraining/nas5_b2_mae_pretrained_48days/base/model_epoch_400_base_48days.pt --output_dir exp/mae_feat/division_nas5_b4_f2_mae_vit_labels_for_mae_pretrained_all_days |& tee logs/b4_f2_mae_feat_extraction.log

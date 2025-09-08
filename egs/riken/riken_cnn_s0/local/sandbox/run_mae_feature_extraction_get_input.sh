# b1_f1
python local/nas5_b2_data_prep_mae_pretrained_batch_gpu.py --gpu 0 --csv_path data/nas5_b1_f1/b1_f1_mae_vit.csv --audio_root /data01/share/bin-wu/data/marmoset/vocalization/riken_long/nas5/b1_906F_1302M_3153M --output_dir /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data/division_nas5_b1_f1_mae_vit_labels_for_mae_pretrained_all_days --train_start 0 --train_end 1000 --dev_start -2 --dev_end -1 --test_start -1 |& tee logs/b1_f1_mae_pretrained_all.log

# b2_f1
python local/nas5_b2_data_prep_mae_pretrained_batch_gpu.py  --gpu 0 --csv_path data/nas5_b2_f1/b2_f1_mae_vit.csv --audio_root /data01/share/bin-wu/data/marmoset/vocalization/riken_long/nas5/b2_1305F_759M_3162F --output_dir /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data/division_nas5_b2_f1_mae_vit_labels_for_mae_pretrained_all_days --train_start 0 --train_end 1000 --dev_start -2 --dev_end -1 --test_start -1 |& tee logs/b2_f1_mae_pretrained_all.log

# b3_f1
python local/nas5_b2_data_prep_mae_pretrained_batch_gpu.py  --gpu 0 --csv_path data/nas5_b3_f1/b3_f1_mae_vit.csv --audio_root /data01/share/bin-wu/data/marmoset/vocalization/riken_long/nas5/b3_762F_763M_3121F --output_dir /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data/division_nas5_b3_f1_mae_vit_labels_for_mae_pretrained_all_days --train_start 0 --train_end 1000 --dev_start -2 --dev_end -1 --test_start -1 |& tee logs/b3_f1_mae_pretrained_all.log

# b4_f1
python local/nas5_b2_data_prep_mae_pretrained_batch_gpu.py  --gpu 0 --csv_path data/nas5_b4_f1/b4_f1_mae_vit.csv --audio_root /data01/share/bin-wu/data/marmoset/vocalization/riken_long/nas5/b4_1372F_1169M_3117F --output_dir /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data/division_nas5_b4_f1_mae_vit_labels_for_mae_pretrained_all_days --train_start 0 --train_end 1000 --dev_start -2 --dev_end -1 --test_start -1 |& tee logs/b4_f1_mae_pretrained_all.log

# b1_f2
python local/nas5_b2_data_prep_mae_pretrained_batch_gpu.py  --gpu 0 --csv_path data/nas5_b1_f2/b1_f2_mae_vit.csv --audio_root /data03/share/bin-wu/data/marmoset/vocalization/riken_long/nas5_f2/b1_906F_1302M_3211F --output_dir /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data/division_nas5_b1_f2_mae_vit_labels_for_mae_pretrained_all_days --train_start 0 --train_end 1000 --dev_start -2 --dev_end -1 --test_start -1 |& tee logs/b1_f2_mae_pretrained_all.log

# b2_f2
python local/nas5_b2_data_prep_mae_pretrained_batch_gpu.py  --gpu 0 --csv_path data/nas5_b2_f2/b2_f2_mae_vit.csv --audio_root /data03/share/bin-wu/data/marmoset/vocalization/riken_long/nas5_f2/b2_1305F_759M_3222M --output_dir /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data/division_nas5_b2_f2_mae_vit_labels_for_mae_pretrained_all_days --train_start 0 --train_end 1000 --dev_start -2 --dev_end -1 --test_start -1 |& tee logs/b2_f2_mae_pretrained_all.log

# b3_f2
python local/nas5_b2_data_prep_mae_pretrained_batch_gpu.py  --gpu 0 --csv_path data/nas5_b3_f2/b3_f2_mae_vit.csv --audio_root /data03/share/bin-wu/data/marmoset/vocalization/riken_long/nas5_f2/b3_762F_763M_3201M --output_dir /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data/division_nas5_b3_f2_mae_vit_labels_for_mae_pretrained_all_days --train_start 0 --train_end 1000 --dev_start -2 --dev_end -1 --test_start -1 |& tee logs/b3_f2_mae_pretrained_all.log

# b4_f2
python local/nas5_b2_data_prep_mae_pretrained_batch_gpu.py  --gpu 0 --csv_path data/nas5_b4_f2/b4_f2_mae_vit.csv --audio_root /data03/share/bin-wu/data/marmoset/vocalization/riken_long/nas5_f2/b4_1372F_1169M_3196M --output_dir /data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data/division_nas5_b4_f2_mae_vit_labels_for_mae_pretrained_all_days --train_start 0 --train_end 1000 --dev_start -2 --dev_end -1 --test_start -1 |& tee logs/b4_f2_mae_pretrained_all.log

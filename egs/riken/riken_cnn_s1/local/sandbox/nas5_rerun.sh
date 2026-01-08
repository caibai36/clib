# B1_F1
bash local/sandbox/run_jay_nas5_b2_f1.sh \
    --stage 1 \
    --dataset b1_f1_mae_vit \
    --wav_dir /data01/share/bin-wu/data/marmoset/vocalization/riken_long/nas5/b1_906F_1302M_3153M \
    --info_csv /work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/data/nas5_b1_f1/b1_f1_mae_vit.csv \
    --exp_dir /data02/share/bin-wu/exp/sandbox/nas5/b1_f1 \
    --dim_reduction_dir /data02/share/bin-wu/exp/sandbox/nas5/b1_f1/dim_reduction_15w \
    --fig_dir /data02/share/bin-wu/exp/sandbox/nas5/b1_f1/dim_reduction_15w/figures \
    |& tee logs/run_jay_b1_f1.log

# B2_F1
bash local/sandbox/run_jay_nas5_b2_f1.sh \
    --stage 1 \
    --dataset b2_f1_mae_vit \
    --wav_dir /data01/share/bin-wu/data/marmoset/vocalization/riken_long/nas5/b2_1305F_759M_3162F \
    --info_csv /work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/data/nas5_b2_f1/b2_f1_mae_vit.csv \
    --exp_dir /data02/share/bin-wu/exp/sandbox/nas5/b2_f1 \
    --dim_reduction_dir /data02/share/bin-wu/exp/sandbox/nas5/b2_f1/dim_reduction_15w \
    --fig_dir /data02/share/bin-wu/exp/sandbox/nas5/b2_f1/dim_reduction_15w/figures \
    |& tee logs/run_jay_b2_f1.log

# B3_F1
bash local/sandbox/run_jay_nas5_b2_f1.sh \
    --stage 1 \
    --dataset b3_f1_mae_vit \
    --wav_dir /data01/share/bin-wu/data/marmoset/vocalization/riken_long/nas5/b3_762F_763M_3121F \
    --info_csv /work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/data/nas5_b3_f1/b3_f1_mae_vit.csv \
    --exp_dir /data02/share/bin-wu/exp/sandbox/nas5/b3_f1 \
    --dim_reduction_dir /data02/share/bin-wu/exp/sandbox/nas5/b3_f1/dim_reduction_15w \
    --fig_dir /data02/share/bin-wu/exp/sandbox/nas5/b3_f1/dim_reduction_15w/figures \
    |& tee logs/run_jay_b3_f1.log

# B4_F1
bash local/sandbox/run_jay_nas5_b2_f1.sh \
    --stage 1 \
    --dataset b4_f1_mae_vit \
    --wav_dir /data01/share/bin-wu/data/marmoset/vocalization/riken_long/nas5/b4_1372F_1169M_3117F \
    --info_csv /work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/data/nas5_b4_f1/b4_f1_mae_vit.csv \
    --exp_dir /data02/share/bin-wu/exp/sandbox/nas5/b4_f1 \
    --dim_reduction_dir /data02/share/bin-wu/exp/sandbox/nas5/b4_f1/dim_reduction_15w \
    --fig_dir /data02/share/bin-wu/exp/sandbox/nas5/b4_f1/dim_reduction_15w/figures \
    |& tee logs/run_jay_b4_f1.log

# B1_F2
bash local/sandbox/run_jay_nas5_b2_f1.sh \
    --stage 1 \
    --dataset b1_f2_mae_vit \
    --wav_dir /data03/share/bin-wu/data/marmoset/vocalization/riken_long/nas5_f2/b1_906F_1302M_3211F \
    --info_csv /work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/data/nas5_b1_f2/b1_f2_mae_vit.csv \
    --exp_dir /data02/share/bin-wu/exp/sandbox/nas5/b1_f2 \
    --dim_reduction_dir /data02/share/bin-wu/exp/sandbox/nas5/b1_f2/dim_reduction_15w \
    --fig_dir /data02/share/bin-wu/exp/sandbox/nas5/b1_f2/dim_reduction_15w/figures \
    --sort_stage min_age \
    |& tee logs/run_jay_b1_f2.log

# B2_F2
bash local/sandbox/run_jay_nas5_b2_f1.sh \
    --stage 1 \
    --dataset b2_f2_mae_vit \
    --wav_dir /data03/share/bin-wu/data/marmoset/vocalization/riken_long/nas5_f2/b2_1305F_759M_3222M \
    --info_csv /work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/data/nas5_b2_f2/b2_f2_mae_vit.csv \
    --exp_dir /data02/share/bin-wu/exp/sandbox/nas5/b2_f2 \
    --dim_reduction_dir /data02/share/bin-wu/exp/sandbox/nas5/b2_f2/dim_reduction_15w \
    --fig_dir /data02/share/bin-wu/exp/sandbox/nas5/b2_f2/dim_reduction_15w/figures \
    |& tee logs/run_jay_b2_f2.log

# B3_F2
bash local/sandbox/run_jay_nas5_b2_f1.sh \
    --stage 1 \
    --dataset b3_f2_mae_vit \
    --wav_dir /data03/share/bin-wu/data/marmoset/vocalization/riken_long/nas5_f2/b3_762F_763M_3201M \
    --info_csv /work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/data/nas5_b3_f2/b3_f2_mae_vit.csv \
    --exp_dir /data02/share/bin-wu/exp/sandbox/nas5/b3_f2 \
    --dim_reduction_dir /data02/share/bin-wu/exp/sandbox/nas5/b3_f2/dim_reduction_15w \
    --fig_dir /data02/share/bin-wu/exp/sandbox/nas5/b3_f2/dim_reduction_15w/figures \
    |& tee logs/run_jay_b3_f2.log

# B4_F2
bash local/sandbox/run_jay_nas5_b2_f1.sh \
    --stage 1 \
    --dataset b4_f2_mae_vit \
    --wav_dir /data03/share/bin-wu/data/marmoset/vocalization/riken_long/nas5_f2/b4_1372F_1169M_3196M \
    --info_csv /work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/data/nas5_b4_f2/b4_f2_mae_vit.csv \
    --exp_dir /data02/share/bin-wu/exp/sandbox/nas5/b4_f2 \
    --dim_reduction_dir /data02/share/bin-wu/exp/sandbox/nas5/b4_f2/dim_reduction_15w \
    --fig_dir /data02/share/bin-wu/exp/sandbox/nas5/b4_f2/dim_reduction_15w/figures \
    |& tee logs/run_jay_b4_f2.log

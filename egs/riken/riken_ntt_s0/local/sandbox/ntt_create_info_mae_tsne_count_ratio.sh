# Original version
# kk
python -u local/sandbox/ntt_create_info_mae_tsne_count_ratio.py \
    --config_df exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa_data_kk/info.csv \
    --tsne_result exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa_data_kk/mae_tsne.npy \
    --output_dir exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa_data_kk \
    --option only_kana |& tee logs/create_kk_mae_tsne_only_kana.log

# ma
python -u local/sandbox/ntt_create_info_mae_tsne_count_ratio.py \
    --config_df exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa_data_ma/info.csv \
    --tsne_result exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa_data_ma/mae_tsne.npy \
    --output_dir exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa_data_ma \
    --option only_kana |& tee logs/create_ma_mae_tsne_only_kana.log

# mk
python -u local/sandbox/ntt_create_info_mae_tsne_count_ratio.py \
    --config_df exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa_data_mk/info.csv \
    --tsne_result exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa_data_mk/mae_tsne.npy \
    --output_dir exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa_data_mk \
    --option only_kana |& tee logs/create_mk_mae_tsne_only_kana.log

# sa
python -u local/sandbox/ntt_create_info_mae_tsne_count_ratio.py \
    --config_df exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa_data_sa/info.csv \
    --tsne_result exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa_data_sa/mae_tsne.npy \
    --output_dir exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa_data_sa \
    --option only_kana |& tee logs/create_sa_mae_tsne_only_kana.log

# sk
python -u local/sandbox/ntt_create_info_mae_tsne_count_ratio.py \
    --config_df exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa_data_sk/info.csv \
    --tsne_result exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa_data_sk/mae_tsne.npy \
    --output_dir exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa_data_sk \
    --option only_kana |& tee logs/create_sk_mae_tsne_only_kana.log

# rerun version
# kk
python -u local/sandbox/ntt_create_info_mae_tsne_count_ratio.py \
    --config_df /data02/share/bin-wu/exp/sandbox/ntt/kk/filter_begin0_end-2/info_filtered.csv \
    --tsne_result /data02/share/bin-wu/exp/sandbox/ntt/kk/filter_begin0_end-2/mae_tsne.npy \
    --output_dir /data02/share/bin-wu/exp/sandbox/ntt/kk/filter_begin0_end-2 \
    --option only_kana |& tee logs/create_kk_mae_tsne_only_kana_rerun.log

# ma
python -u local/sandbox/ntt_create_info_mae_tsne_count_ratio.py \
    --config_df /data02/share/bin-wu/exp/sandbox/ntt/ma/filter_begin0_end-2/info_filtered.csv \
    --tsne_result /data02/share/bin-wu/exp/sandbox/ntt/ma/filter_begin0_end-2/mae_tsne.npy \
    --output_dir /data02/share/bin-wu/exp/sandbox/ntt/ma/filter_begin0_end-2 \
    --option only_kana |& tee logs/create_ma_mae_tsne_only_kana_rerun.log

# mk
python -u local/sandbox/ntt_create_info_mae_tsne_count_ratio.py \
    --config_df /data02/share/bin-wu/exp/sandbox/ntt/mk/filter_begin0_end-2/info_filtered.csv \
    --tsne_result /data02/share/bin-wu/exp/sandbox/ntt/mk/filter_begin0_end-2/mae_tsne.npy \
    --output_dir /data02/share/bin-wu/exp/sandbox/ntt/mk/filter_begin0_end-2 \
    --option only_kana |& tee logs/create_mk_mae_tsne_only_kana_rerun.log

# sa
python -u local/sandbox/ntt_create_info_mae_tsne_count_ratio.py \
    --config_df /data02/share/bin-wu/exp/sandbox/ntt/sa/filter_begin0_end-2/info_filtered.csv \
    --tsne_result /data02/share/bin-wu/exp/sandbox/ntt/sa/filter_begin0_end-2/mae_tsne.npy \
    --output_dir /data02/share/bin-wu/exp/sandbox/ntt/sa/filter_begin0_end-2 \
    --option only_kana |& tee logs/create_sa_mae_tsne_only_kana_rerun.log

# sk
python -u local/sandbox/ntt_create_info_mae_tsne_count_ratio.py \
    --config_df /data02/share/bin-wu/exp/sandbox/ntt/sk/filter_begin0_end-2/info_filtered.csv \
    --tsne_result /data02/share/bin-wu/exp/sandbox/ntt/sk/filter_begin0_end-2/mae_tsne.npy \
    --output_dir /data02/share/bin-wu/exp/sandbox/ntt/sk/filter_begin0_end-2 \
    --option only_kana |& tee logs/create_sk_mae_tsne_only_kana_rerun.log

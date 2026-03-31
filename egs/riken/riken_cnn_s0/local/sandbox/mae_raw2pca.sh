# Process all files in a loop
for file in /work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/exp/mae_feat_dim_reduction/python_script_nas5/*_15weeks/mae_feature_*_15weeks.npy; do
    python local/sandbox/mae_raw2pca.py --input_file "$file"
done

# Process all files
for file in /work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_ntt_s0/exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa*/mae_feature_mae_pretrain_*.npy; do
    python local/sandbox/mae_raw2pca.py --input_file "$file"
done

#!/bin/bash
# Copy mae_feature files to mae_raw.npy

echo "=== Copying riken_cnn_s0 files ==="
for file in /work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/exp/mae_feat_dim_reduction/python_script_nas5/*_15weeks/mae_feature_*_15weeks.npy; do
    dir=$(dirname "$file")
    echo "Copying: $file"
    echo "     To: $dir/mae_raw.npy"
    cp "$file" "$dir/mae_raw.npy"
done

echo ""
echo "=== Copying riken_ntt_s0 files ==="
for file in /work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_ntt_s0/exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa*/mae_feature_mae_pretrain_*.npy; do
    dir=$(dirname "$file")
    echo "Copying: $file"
    echo "     To: $dir/mae_raw.npy"
    cp "$file" "$dir/mae_raw.npy"
done

echo ""
echo "=== Done! ==="

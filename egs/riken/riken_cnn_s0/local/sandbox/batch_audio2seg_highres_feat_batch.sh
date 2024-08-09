arch="cnn"
model="conf/model/b0family3010_best_dev_high_res_0.01sec.ckpt"
# out_dir="nas5/familybooth_1594F_1449M_3010" # actual out_dir: "exp/out/${arch}_${model_base}/$out_dir/$file_dir/${file_base}_model_${arch}_${model_base}.txt"
# data_dir="/data01/share/bin-wu/data/marmoset/vocalization/riken_long/nas5/familybooth_1594F_1449M_3010"
out_dir="nas5"  # actual out_dir: "exp/out/${arch}_${model_base}/$out_dir/$file_dir/${file_base}_model_${arch}_${model_base}.txt"
data_dir="/data01/share/bin-wu/data/marmoset/vocalization/riken_long/nas5"
feat_batch_size=20480

# Parse the options. (e.g., ./run.sh --stage 1)
# Note that the options should be defined as shell variables before parsing
. local/scripts/parse_options.sh || exit 1

IFS=$'\n' # Set the Internal Field Separator to newline (This ensures that filenames with spaces are handled correctly)

for file in $(cd "$data_dir"; find . -type f -name "*.wav" | sort); do
    file_base="$(basename "$file" .wav)"
    file_dir="$(dirname "$file")"
    model_base=$(basename "$model" .ckpt)
    out_seg_file="exp/tests/${arch}_model_${model_base}/$out_dir/$file_dir/${file_base}_${arch}_model_${model_base}.txt"

    # Check if the output segmentation file already exists
    if [ -f "$out_seg_file" ]; then
        echo "Skipping $file as $out_seg_file already exists."
        continue
    fi

    # Run the python script if the output file does not exist
    python local/sandbox/riken_cnn_audio2seg_highres_feat_batch.py \
	--feat_batch_size "$feat_batch_size" \
        --eval_model "$model" \
        --wav_file "$data_dir/$file" \
        --out_seg_file "$out_seg_file"
done

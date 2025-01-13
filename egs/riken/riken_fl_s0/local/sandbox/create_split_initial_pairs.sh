phoneme_file="conf/data/Initials.txt"
readarray -t phonemes < "${phoneme_file}"

for ((i=0; i<${#phonemes[@]}; i++)); do
    for ((j=i+1; j<${#phonemes[@]}; j++)); do
        # Get phoneme pairs and clean them
        phoneme1=$(echo "${phonemes[i]}" | tr -d '\n\r')
        phoneme2=$(echo "${phonemes[j]}" | tr -d '\n\r')
        echo $phoneme1 $phoneme2
    done
done |& tee conf/dict/labels/initial_pairs.txt

# divide into n groups
num_lines_each_group=30
input=conf/dict/labels/initial_pairs.txt
output_dir=conf/dict/labels/initial_pairs_groups

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Split into exactly 30 lines per file
split -l $num_lines_each_group "$input" --numeric-suffixes=1 --additional-suffix=.txt \
      --suffix-length=2 "$output_dir/initial_pairs_group"

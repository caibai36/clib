#!/bin/bash

# Default values
start_gram=1
end_gram=6
phoneme_file="conf/data/Initials.txt"
corpus_file="exp/data/tv07_phoneme.txt"
output_dir="exp/sandbox/fl_ce_chinese/all_pairs/csvs/n-grams"
syllable_fl=true

# Parse the options
. local/scripts/parse_options.sh || exit 1

# Validate inputs
if ! [[ "$start_gram" =~ ^[0-9]+$ ]] || ! [[ "$end_gram" =~ ^[0-9]+$ ]]; then
    echo "Error: start_gram and end_gram must be integers"
    exit 1
fi

if [ "$start_gram" -gt "$end_gram" ]; then
    echo "Error: start_gram must be less than or equal to end_gram"
    exit 1
fi

if [ "$start_gram" -lt 1 ]; then
    echo "Error: start_gram must be at least 1"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Process each n-gram size
for n in $(seq $start_gram $end_gram); do
    gram_name="${n}-gram"
    output_file="${output_dir}/fl_ent_tv07_syllable_${gram_name}.csv"

    echo "Computing functional load for ${n}-gram..."

    # Run the FL computation script with appropriate parameters
    ./local/sandbox/fl_ent.sh \
        --ngram $n \
        --phoneme_file "$phoneme_file" \
        --corpus_file "$corpus_file" \
        --output_file "$output_file" \
        --syllable_fl "$syllable_fl"

    echo "Completed ${n}-gram. Output saved to $output_file"
done

echo "All n-gram computations completed successfully."

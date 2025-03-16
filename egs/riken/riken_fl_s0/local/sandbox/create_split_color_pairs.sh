#!/bin/bash

# Configuration
color_file="conf/dict/labels/colors.txt"
output_pairs="conf/dict/labels/color_pairs.txt"
output_dir="conf/dict/labels/color_pairs_groups"

# Create output directories
mkdir -p "$output_dir"

# Read colors into array
readarray -t colors < "${color_file}"

# Generate all possible pairs and save to file
for ((i=0; i<${#colors[@]}; i++)); do
    for ((j=i+1; j<${#colors[@]}; j++)); do
        # Get color pairs and clean them
        color1=$(echo "${colors[i]}" | tr -d '\n\r')
        color2=$(echo "${colors[j]}" | tr -d '\n\r')
        echo $color1 $color2
    done
done |& tee "$output_pairs"

# Split into groups of 5 pairs each
num_lines_each_group=5
split -l $num_lines_each_group "$output_pairs" --numeric-suffixes=1 --additional-suffix=.txt \
      --suffix-length=2 "$output_dir/color_pairs_group"

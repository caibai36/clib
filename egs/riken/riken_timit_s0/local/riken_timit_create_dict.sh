#!/bin/bash

# Define paths
seg_scp="data/timit/seg.scp"
output_file="conf/dict/timit_label2id.yaml"

# Create output directory if it doesn't exist
mkdir -p "$(dirname $output_file)"

# Add default labels
cat > "$output_file" << EOF
<unk>: 0
<pad>: 1
<sos>: 2
<eos>: 3
<period>: 4
<space>: 5
noise: 6
EOF

# Extract all unique phone labels from PHN files
id=7
for phn_file in $(cut -d' ' -f2 "$seg_scp"); do
    awk '{print $3}' "$phn_file" >> /tmp/all_phones.txt
done

# Sort and get unique labels, then append to output file
sort /tmp/all_phones.txt | uniq | while read phone; do
    echo "$phone: $id" >> "$output_file"
    id=$((id+1))
done

# Clean up
rm /tmp/all_phones.txt

echo "Created dict at $output_file"

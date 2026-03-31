#!/bin/bash
# Split file into multiple parts for parallel processing
set -euo pipefail

# Default configuration
input=""
n_splits=3
split_dir=""

# Parse options
. local/scripts/parse_options.sh || exit 1

# Show help if input not provided
if [[ -z "${input}" ]]; then
    cat << EOF
Usage: $0 --input <file> [options]

Required:
  --input <file>          Input file to split

Options:
  --n-splits <num>        Number of splits (default: 3)
  --split-dir <dir>       Output directory (default: same as input file directory + /splits)

Example:
  $0 --input conf/mri/ids.txt --n-splits 5
  Output: conf/mri/splits/ids_part00.txt, ids_part01.txt, ...
EOF
    exit 1
fi

# Validate inputs
[[ ! -f "${input}" ]] && echo "Error: File not found: ${input}" >&2 && exit 1
[[ ! "${n_splits}" =~ ^[0-9]+$ ]] && echo "Error: n_splits must be positive integer" >&2 && exit 1

# Set split_dir to input directory if not specified
if [[ -z "${split_dir}" ]]; then
    split_dir="$(dirname "${input}")/splits"
fi

# Extract filename and extension
filename=$(basename "${input}")
basename_no_ext="${filename%.*}"
extension="${filename##*.}"
[[ "${filename}" == "${extension}" ]] && extension=""  # No extension case

# Setup
mkdir -p "${split_dir}"
total_lines=$(wc -l < "${input}")
lines_per_split=$(( (total_lines + n_splits - 1) / n_splits ))

echo "Splitting ${total_lines} lines into ${n_splits} parts (~${lines_per_split} lines each)"
echo "Output directory: ${split_dir}"

# Split file
rm -f "${split_dir}/${basename_no_ext}_part"*
split -l "${lines_per_split}" -d -a 2 "${input}" "${split_dir}/${basename_no_ext}_part"

# Add original extension
if [[ -n "${extension}" ]]; then
    for file in "${split_dir}/${basename_no_ext}_part"*; do
        [[ -e "$file" ]] && mv "$file" "${file}.${extension}"
    done
    echo "Split files created:"
    ls -lh "${split_dir}/${basename_no_ext}_part"*.${extension}
else
    echo "Split files created:"
    ls -lh "${split_dir}/${basename_no_ext}_part"*
fi

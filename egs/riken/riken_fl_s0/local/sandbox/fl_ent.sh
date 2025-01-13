#!/bin/bash

# Path to the Python script
python_script="local/fl_ent.py"
# Path to the phoneme file
phoneme_file="conf/data/Initials.txt"
# Path to the corpus file
corpus_file="exp/data/sample_tv07_phoneme.txt"
# Output file for results
output_file="exp/fl_ent/fl_sample_tv07_phoneme.txt"
# Compute the syllable fl
syllable_fl=false
# Compute the ngram
ngram=2

# Parse the options. (e.g., ./run.sh --stage 1)
# Note that the options should be defined as shell variables before parsing
. local/scripts/parse_options.sh || exit 1

# Create output directory if it doesn't exist
mkdir -p exp/fl_ent

# Clear previous results
> ${output_file}

echo "phone1,phone2,fl,fl_diff,original_entropy,merged_entropy" |& tee -a ${output_file}

# Read all phonemes into an array
readarray -t phonemes < ${phoneme_file}

# Iterate through unique pairs without repetition
for ((i=0; i<${#phonemes[@]}; i++)); do
    for ((j=i+1; j<${#phonemes[@]}; j++)); do
        phoneme1="${phonemes[i]}"
        phoneme2="${phonemes[j]}"

        # Remove any trailing whitespace/newlines
        phoneme1=$(echo "$phoneme1" | tr -d '\n\r')
        phoneme2=$(echo "$phoneme2" | tr -d '\n\r')

	# Run Python script and append results to output file
	        # Run Python script and append results to output file
        if $syllable_fl; then
            python ${python_script} \
                --corpus "${corpus_file}" \
                --phoneme1 "${phoneme1}" \
                --phoneme2 "${phoneme2}" \
		--ngram "${ngram}" \
                --syllable_fl \
                |& tee -a ${output_file}
        else
            python ${python_script} \
                --corpus "${corpus_file}" \
                --phoneme1 "${phoneme1}" \
                --phoneme2 "${phoneme2}" \
		--ngram "${ngram}" \
                |& tee -a ${output_file}
        fi
    done
done

echo "Completed! Results saved in ${output_file}"

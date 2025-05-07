#!/bin/bash

# Check if correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <timestamp_file> <audio_file> <output_directory>"
    exit 1
fi

timestamp_file="$1"
audio_file="$2"
output_dir="$3"

# Check if input files exist
if [ ! -f "$timestamp_file" ]; then
    echo "Error: Timestamp file '$timestamp_file' does not exist."
    exit 1
fi

if [ ! -f "$audio_file" ]; then
    echo "Error: Audio file '$audio_file' does not exist."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$output_dir"

# Read the lines of timestamp file into an array
times=( $(awk '{print $2}' "$timestamp_file") )

# Get the total number of segments
num_segments=${#times[@]}

# Get the total duration of the audio file
total_duration=$(sox --i -D "$audio_file")

# Extract base name of audio file (without extension)
base_name=$(basename "$audio_file" .wav)

# Loop through each segment
for i in $(seq 0 $(($num_segments - 1))); do
  # Start time of the current segment
  start=${times[$i]}
  
  # Output file name
  output="${output_dir}/${base_name}_$((i+1))_sess.wav"
  
  if [ $i -eq $(($num_segments - 1)) ]; then
    # For the last segment, use the total duration as the end time
    sox "$audio_file" "$output" trim "$start"
  else
    # End time is the start of the next segment
    end=${times[$(($i + 1))]}
    
    # Duration of the current segment
    duration=$(awk "BEGIN {print $end - $start}")
    
    # Use sox to extract the segment
    sox "$audio_file" "$output" trim "$start" "$duration"
  fi
done

echo "Segmentation complete. Output files are in $output_dir/${base_name}"
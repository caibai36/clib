#!/usr/bin/env python3
"""
Script to convert Kaldi force alignments to text format.
Implemented by bin-wu at 10:38 on 2025/05/07

This script does two things:
1. It extracts raw Kaldi alignments and writes them to the output directory
2. It creates fixed alignments by extending Kaldi alignments to match original segment boundaries
   and writes them to a 'fixed' subdirectory

The fixed alignments algorithm:
1. First token's start time is set to the original segment's start time
2. Last token's end time is set to the original segment's end time
3. Boundaries between tokens are set to the midpoint between adjacent tokens
4. For segments with missing alignments, evenly spaced tokens are created within the segment boundaries
5. The script handles overlapped annotations and missing alignments

Fixed alignments ensure that:
- Token boundaries align perfectly with original segment boundaries
- All tokens from the original transcripts are accounted for
- Even when Kaldi fails to provide alignments, reasonable estimates are created
- Overlapping segments and other edge cases are handled properly
"""

import os
import json
import argparse
import subprocess
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(
        description="""
        Convert Kaldi alignments to text format and create fixed alignments.
        Two sets of files are generated:
        1. Raw Kaldi alignments in the output directory
        2. Fixed alignments in the 'fixed' subdirectory that extend Kaldi alignments
        to match original segment boundaries

        The fixed alignment algorithm:
        - First token's start time is set to the original segment's start time
        - Last token's end time is set to the original segment's end time
        - Boundaries between tokens are set to the midpoint between adjacent tokens
        - If Kaldi alignments are not available, falls back to even spacing
        - Handles overlapped annotations and missing alignments
        - Only processes recordings included in the current dataset (test or all)

        Debug files are created to track alignment decisions and identify potential issues.
        # e.g., check the warnings by
        # grep -i warning exp/out/all/fixed/debug/* | wc -l
        # grep -i warning exp/out/test/fixed/debug/* | wc -l
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ali_dir", type=str, required=True,
                        help="Directory containing Kaldi alignments")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing Kaldi data (segments, etc.)")
    parser.add_argument("--data_info_json", type=str, required=True,
                        help="Path to the data info JSON file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for alignment text files")
    return parser.parse_args()

def read_segments_file(segments_path):
    """Read segments file to map utterance IDs to recordings and time ranges"""
    segments = {}
    with open(segments_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                utt_id, rec_id, start_time, end_time = parts
                segments[utt_id] = (rec_id, float(start_time), float(end_time))
    return segments

def read_text_file(text_path):
    """Read text file to get tokens for each utterance"""
    text_dict = {}
    with open(text_path, "r") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                utt_id, text = parts
                tokens = text.split()
                text_dict[utt_id] = tokens
    return text_dict

def read_original_tokens(token_file):
    """Read original token file to get token sequences with timestamps"""
    original_segments = []
    try:
        with open(token_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    start_time = float(parts[0])
                    end_time = float(parts[1])
                    tokens = parts[2:]
                    original_segments.append((start_time, end_time, tokens))
    except Exception as e:
        print("Error reading token file {}: {}".format(token_file, e))

    return original_segments

def extract_phone_ctm(ali_dir):
    """Extract phone alignments in CTM format with timing information"""
    model_path = os.path.join(ali_dir, "final.mdl")

    # Run ali-to-phones with CTM output
    process = subprocess.Popen(
        "ali-to-phones --ctm-output=true {} \"ark:gunzip -c {}/ali.*.gz |\" -".format(
            model_path, ali_dir),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print("Error extracting CTM: {}".format(stderr))
        return {}

    # Parse CTM output (format: utt_id channel start_time duration phone)
    phone_ctm = defaultdict(list)
    for line in stdout.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) >= 5:
            utt_id, channel, start_time, duration, phone = parts[:5]
            phone_ctm[utt_id].append({
                'start': float(start_time),
                'duration': float(duration),
                'end': float(start_time) + float(duration),
                'phone': int(phone)
            })

    return phone_ctm

def convert_alignments(ali_dir, data_dir, data_info_json, output_dir):
    """
    Convert Kaldi alignments to token-level timings and create fixed alignments.

    This function is the main processing pipeline for alignment conversion:
    1. Reads Kaldi segment information and transcript text
    2. Extracts phone-level alignments from Kaldi model
    3. Maps phones to tokens and computes token-level timings
    4. Creates two sets of alignment files:
       - Raw Kaldi alignments (directly in output_dir)
       - Fixed alignments (in output_dir/fixed)

    The raw alignments preserve Kaldi's exact timings, while fixed alignments
    adjust token boundaries to match original segment start/end times.

    For phone-to-token mapping, two approaches are used based on relative counts:
    - When there are more phones than tokens: Multiple phones are grouped per token
    - When there are more tokens than phones: Tokens are distributed within phone durations

    This handles various Japanese speech patterns in infant recordings including:
    - Variable speech rates
    - Pronunciation variations
    - Potentially incomplete alignments

    Parameters:
    -----------
    ali_dir : str
        Directory containing Kaldi alignments (ali.*.gz files and final.mdl)
    data_dir : str
        Directory containing Kaldi data (segments, text files)
    data_info_json : str
        Path to JSON file mapping utterance IDs to original data files
    output_dir : str
        Directory where alignment files will be written

    Returns:
    --------
    str
        Path to the output directory containing the alignments
    """
    os.makedirs(output_dir, exist_ok=True)

    # Read segments file
    segments = read_segments_file(os.path.join(data_dir, "segments"))

    # Read text file to get tokens for each utterance
    text_dict = read_text_file(os.path.join(data_dir, "text"))

    # Extract phone alignments in CTM format
    phone_ctm = extract_phone_ctm(ali_dir)

    # Store token alignments by utterance ID and recording ID
    utt_alignments = {}  # Map from utt_id to list of (start, end, token)
    rec_alignments = defaultdict(list)  # Map from rec_id to list of (utt_id, start, end, token)

    # Group segments by recording ID
    recording_segments = defaultdict(list)
    for utt_id, (rec_id, start_time, end_time) in segments.items():
        recording_segments[rec_id].append((utt_id, start_time, end_time))

    # Keep track of which recordings were processed
    processed_recordings = set()

    # For each recording, create an alignment file
    for rec_id in recording_segments:
        output_path = os.path.join(output_dir, "{}.txt".format(rec_id))
        processed_recordings.add(rec_id)  # Mark this recording as processed

        with open(output_path, "w") as out_f:
            # Sort segments by start time
            sorted_segments = sorted(recording_segments[rec_id], key=lambda x: x[1])

            for utt_id, start_time, end_time in sorted_segments:
                # Skip if no alignment for this utterance
                if utt_id not in phone_ctm:
                    print("Warning: No alignment for {}".format(utt_id))
                    continue

                # Get tokens for this utterance
                tokens = text_dict.get(utt_id, [])
                if not tokens:
                    continue

                # Get phone alignments
                phones = phone_ctm[utt_id]

                # Initialize alignment list for this utterance
                utt_alignments[utt_id] = []

                # Filter out silence phones (usually have phone IDs 1 or 2)
                non_silence_phones = [p for p in phones if p['phone'] > 2]

                # If we have tokens and phones, map them
                if tokens and non_silence_phones:
                    # Handle the alignment case - more or equal phones than tokens
                    if len(tokens) <= len(non_silence_phones):
                        # Group phones to match tokens
                        phones_per_token = len(non_silence_phones) // len(tokens)
                        remainder = len(non_silence_phones) % len(tokens)

                        # Assign phones to tokens
                        for i, token in enumerate(tokens):
                            # Calculate how many phones for this token
                            num_phones = phones_per_token + (1 if i < remainder else 0)

                            # Get the phones for this token
                            start_idx = i * phones_per_token + min(i, remainder)
                            end_idx = start_idx + num_phones
                            token_phones = non_silence_phones[start_idx:end_idx]

                            if token_phones:
                                # Calculate token start and end times
                                token_start = start_time + token_phones[0]['start']
                                token_end = start_time + token_phones[-1]['end']

                                # Store alignment for this token
                                utt_alignments[utt_id].append((token_start, token_end, token))
                                rec_alignments[rec_id].append((utt_id, token_start, token_end, token))

                                # Write token with its timing
                                out_f.write("{:.6f}\t{:.6f}\t{}\n".format(token_start, token_end, token))
                    else:
                        # More tokens than phones
                        tokens_per_phone = len(tokens) // len(non_silence_phones)
                        remainder = len(tokens) % len(non_silence_phones)

                        # Process each phone
                        token_idx = 0
                        for i, phone in enumerate(non_silence_phones):
                            # Calculate how many tokens for this phone
                            num_tokens = tokens_per_phone + (1 if i < remainder else 0)

                            # Calculate phone timing
                            phone_start = start_time + phone['start']
                            phone_end = start_time + phone['end']
                            phone_duration = phone['duration']

                            # Distribute tokens evenly within this phone's duration
                            for j in range(num_tokens):
                                if token_idx < len(tokens):
                                    token = tokens[token_idx]
                                    token_duration = phone_duration / num_tokens
                                    token_start = phone_start + j * token_duration
                                    token_end = token_start + token_duration

                                    # Store alignment for this token
                                    utt_alignments[utt_id].append((token_start, token_end, token))
                                    rec_alignments[rec_id].append((utt_id, token_start, token_end, token))

                                    # Write token with its timing
                                    out_f.write("{:.6f}\t{:.6f}\t{}\n".format(token_start, token_end, token))
                                    token_idx += 1

    # Create debug output with all phone alignments
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    with open(os.path.join(debug_dir, "phone_ctm_parsed.txt"), "w") as f:
        for utt_id, phones in phone_ctm.items():
            f.write("{}:\n".format(utt_id))
            for p in phones:
                f.write("  {:.3f} - {:.3f} ({:.3f}): {}\n".format(
                    p['start'], p['end'], p['duration'], p['phone']))
            f.write("\n")

    # Create fixed alignments
    create_fixed_alignments(output_dir, data_dir, segments, text_dict,
                           utt_alignments, rec_alignments, processed_recordings, data_info_json)

    return output_dir

def create_fixed_alignments(output_dir, data_dir, segments, text_dict,
                           utt_alignments, rec_alignments, processed_recordings, data_info_json=None):
    """
    Create fixed alignments by extending Kaldi alignments to match original segment boundaries.

    Only processes recordings that were included in the current dataset (test or all).

    Algorithm:
    1. First token's start time is set to the original segment's start time
    2. Last token's end time is set to the original segment's end time
    3. Boundaries between tokens are set to the midpoint between adjacent tokens
    4. For segments with missing alignments, evenly spaced tokens are created within the segment boundaries

    This approach:
    - The script handles overlapped annotations and missing alignments
    - Process at utt_id level
    - Add debug information

    Parameters:
    -----------
    output_dir : str
        Directory containing raw Kaldi alignments
    data_dir : str
        Directory containing Kaldi data (segments, text files)
    segments : dict
        Mapping of utterance IDs to recordings and time ranges
    text_dict : dict
        Mapping of utterance IDs to token sequences
    utt_alignments : dict
        Mapping of utterance IDs to token alignments
    rec_alignments : dict
        Mapping of recording IDs to token alignments
    processed_recordings : set
        Set of recording IDs that were processed in this run
    data_info_json : str, optional
        Path to data info JSON file
    """
    # Create output directory for fixed alignments
    fixed_dir = os.path.join(output_dir, "fixed")
    os.makedirs(fixed_dir, exist_ok=True)

    # Debug directory for fixed alignments
    fixed_debug_dir = os.path.join(fixed_dir, "debug")
    os.makedirs(fixed_debug_dir, exist_ok=True)

    # Group segments by recording ID
    recording_segments = defaultdict(list)
    for utt_id, (rec_id, start_time, end_time) in segments.items():
        recording_segments[rec_id].append((utt_id, start_time, end_time))

    # Process each recording
    for rec_id in processed_recordings:
        # Skip if no segments for this recording
        if rec_id not in recording_segments:
            continue

        # Debug file for this recording
        debug_file = os.path.join(fixed_debug_dir, "{}_debug.txt".format(rec_id))

        # Store all fixed alignments for this recording
        rec_fixed_alignments = []

        with open(debug_file, "w") as debug_f:
            debug_f.write("Recording: {}\n\n".format(rec_id))

            # Sort segments by start time
            sorted_segments = sorted(recording_segments[rec_id], key=lambda x: x[1])

            # Write original segments to debug file
            debug_f.write("Original segments:\n")
            for i, (utt_id, start, end) in enumerate(sorted_segments):
                tokens = text_dict.get(utt_id, [])
                h = int(start // 3600)
                m = int((start % 3600) // 60)
                s = int(start % 60)
                ms = int((start % 1) * 1000)
                time_str = f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
                debug_f.write(f"  {i} - {utt_id} ({time_str}): {start:.6f} - {end:.6f}: {' '.join(tokens)}\n")
            debug_f.write("\n")

            # Process each utterance for this recording
            for utt_idx, (utt_id, seg_start, seg_end) in enumerate(sorted_segments):
                tokens = text_dict.get(utt_id, [])
                if not tokens:
                    continue

                # Get Kaldi alignments for this utterance
                utt_tokens = utt_alignments.get(utt_id, [])

                # Format time for debug output
                h = int(seg_start // 3600)
                m = int((seg_start % 3600) // 60)
                s = int(seg_start % 60)
                ms = int((seg_start % 1) * 1000)
                time_str = f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

                if utt_tokens:
                    # Write Kaldi alignments to debug file
                    debug_f.write(f"Kaldi alignments for utterance {utt_id} ({time_str}):\n")
                    for i, (start, end, token) in enumerate(utt_tokens):
                        debug_f.write("  {}: {:.6f} - {:.6f}: {}\n".format(
                            i, start, end, token))
                    debug_f.write("\n")

                    # Check if tokens match
                    if len(utt_tokens) == len(tokens):
                        kaldi_tokens = [t[2] for t in utt_tokens]
                        if kaldi_tokens != tokens:
                            debug_f.write(f"Warning: Token mismatch: Kaldi: {kaldi_tokens} vs Ref: {tokens}\n")

                    # Check if token counts match
                    if len(utt_tokens) != len(tokens):
                        debug_f.write(f"Warning: Token count mismatch for utterance {utt_id} ({time_str}), using even spacing\n")
                        debug_f.write(f"  Kaldi tokens: {len(utt_tokens)}, Segment tokens: {len(tokens)}\n\n")

                        # Use even spacing for this utterance
                        token_duration = (seg_end - seg_start) / len(tokens)
                        utt_fixed_tokens = []

                        for i, token in enumerate(tokens):
                            token_start = seg_start + i * token_duration
                            token_end = token_start + token_duration
                            utt_fixed_tokens.append((token_start, token_end, token))

                        rec_fixed_alignments.extend(utt_fixed_tokens)

                        # Write evenly spaced alignments to debug file
                        debug_f.write(f"Evenly spaced alignments for utterance {utt_id}:\n")
                        for i, (start, end, token) in enumerate(utt_fixed_tokens):
                            debug_f.write("  {}: {:.6f} - {:.6f}: {}\n".format(
                                i, start, end, token))
                        debug_f.write("\n")

                        continue

                    # Apply fixed alignment algorithm
                    utt_fixed_tokens = []

                    for i, (orig_start, orig_end, token) in enumerate(utt_tokens):
                        if i == 0:
                            # First token - match start time to original segment start
                            token_start = seg_start
                        else:
                            # Middle token - use midpoint between previous token's end and this token's start
                            prev_end = utt_tokens[i-1][1]
                            curr_start = orig_start
                            token_start = (prev_end + curr_start) / 2

                        if i == len(utt_tokens) - 1:
                            # Last token - match end time to original segment end
                            token_end = seg_end
                        else:
                            # Middle token - use midpoint between this token's end and next token's start
                            curr_end = orig_end
                            next_start = utt_tokens[i+1][0]
                            token_end = (curr_end + next_start) / 2

                        # Add fixed alignment
                        utt_fixed_tokens.append((token_start, token_end, token))

                    # Add to recording's fixed alignments
                    rec_fixed_alignments.extend(utt_fixed_tokens)

                    # Write fixed alignments to debug file
                    debug_f.write(f"Fixed alignments for utterance {utt_id}:\n")
                    for i, (start, end, token) in enumerate(utt_fixed_tokens):
                        debug_f.write("  {}: {:.6f} - {:.6f}: {}\n".format(
                            i, start, end, token))
                    debug_f.write("\n")
                else:
                    debug_f.write(f"Warning: No Kaldi alignments for utterance {utt_id} ({time_str}), using even spacing\n\n")

                    # Use even spacing for this utterance
                    token_duration = (seg_end - seg_start) / len(tokens)
                    utt_fixed_tokens = []

                    for i, token in enumerate(tokens):
                        token_start = seg_start + i * token_duration
                        token_end = token_start + token_duration
                        utt_fixed_tokens.append((token_start, token_end, token))

                    rec_fixed_alignments.extend(utt_fixed_tokens)

                    # Write evenly spaced alignments to debug file
                    debug_f.write(f"Evenly spaced alignments for utterance {utt_id}:\n")
                    for i, (start, end, token) in enumerate(utt_fixed_tokens):
                        debug_f.write("  {}: {:.6f} - {:.6f}: {}\n".format(
                            i, start, end, token))
                    debug_f.write("\n")

            # Sort all fixed alignments by start time
            rec_fixed_alignments.sort(key=lambda x: x[0])

            # Write all fixed alignments for this recording to debug file
            debug_f.write("All fixed alignments for recording {}:\n".format(rec_id))
            for i, (start, end, token) in enumerate(rec_fixed_alignments):
                debug_f.write("  {}: {:.6f} - {:.6f}: {}\n".format(
                    i, start, end, token))

        # Write fixed alignments to output file
        output_file = os.path.join(fixed_dir, "{}.txt".format(rec_id))
        with open(output_file, "w") as out_f:
            for start, end, token in rec_fixed_alignments:
                out_f.write("{:.6f}\t{:.6f}\t{}\n".format(start, end, token))

def main():
    args = parse_args()
    convert_alignments(args.ali_dir, args.data_dir, args.data_info_json, args.output_dir)
    print("Converted alignments written to {}".format(args.output_dir))
    print("Fixed alignments written to {}/fixed".format(args.output_dir))

if __name__ == "__main__":
    main()

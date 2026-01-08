#!/usr/bin/env python3
"""
Script to extract and process NTT infant segment information and create a CSV dataset.
This script reads audio segment files from Kaldi debug files, processes them with meta information,
and creates a CSV file containing segment information including timing, labels, and demographics.

The script processes audio segments and creates extended timing information for feature extraction:
- For segments shorter than 0.5s: extends them equally on both sides to reach 0.5s
- For segments longer than 0.5s: keeps original extent but adds cut points to extract middle 0.5s

Main features:
- Processes Kaldi debug files containing segment file paths
- Handles both fixed alignments and evenly spaced alignments from Kaldi debug files
- Merges with meta CSV file for demographic information
- Calculates age information from meta data
- Filters segments by maximum duration
- Creates extended timing information for consistent feature extraction
- Outputs results to CSV format
"""

import os
import re
import argparse
from datetime import datetime
import pandas as pd
import warnings

class Segment:
    """
    A class to represent an audio segment in NTT infant format.

    Attributes:
        begin_sec (float): Start time of the segment in seconds
        end_sec (float): End time of the segment in seconds
        label (str): Label of the segment (phoneme)
        utt_id (str): Utterance ID
        utt_id_index (int): Utterance ID index
        low_freq (float): Lower frequency bound (optional, not used in NTT infant)
        high_freq (float): Upper frequency bound (optional, not used in NTT infant)

    Note:
        Frequencies are not used in NTT infant dataset but kept for compatibility
    """
    def __init__(self, begin_sec, end_sec, label, utt_id, utt_id_index, low_freq=None, high_freq=None):
        self.begin_sec = begin_sec
        self.end_sec = end_sec
        self.label = label
        self.utt_id = utt_id
        self.utt_id_index = utt_id_index
        self.low_freq = low_freq
        self.high_freq = high_freq

def read_kaldi_debug_segments(debug_file):
    """
    Read and parse a Kaldi debug file to extract fixed alignment and evenly spaced alignment segments.

    Args:
        debug_file (str): Path to the Kaldi debug file

    Returns:
        list: List of Segment objects with utterance information

    Note:
        Processes both "Fixed alignments for utterance" and "Evenly spaced alignments for utterance" sections.
        Ignores "Original segments", "Kaldi alignments", and "All fixed alignments" sections.

    Example format:
        Fixed alignments for utterance ma028_1_0296:
          0: 621.862000 - 621.957000: ãƒ•
          1: 621.957000 - 622.161000: ãƒ³
        Evenly spaced alignments for utterance ma028_1_0297:
          0: 624.809000 - 625.095393: ã‚±
          1: 625.095393 - 625.381786: ãƒƒ
          ...
    """
    segments = []
    try:
        with open(debug_file, encoding='utf8') as f:
            current_utt_id = None
            in_target_alignment_section = False

            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Check for different section headers
                if line.startswith('Original segments:'):
                    in_target_alignment_section = False
                    current_utt_id = None
                    continue
                elif line.startswith('Kaldi alignments for utterance'):
                    in_target_alignment_section = False
                    current_utt_id = None
                    continue
                elif line.startswith('All fixed alignments for recording'):
                    in_target_alignment_section = False
                    current_utt_id = None
                    continue
                elif line.startswith('Fixed alignments for utterance'):
                    # Extract utterance ID from "Fixed alignments for utterance ma028_1_0296:"
                    utt_match = re.match(r'Fixed alignments for utterance (.+):', line)
                    if utt_match:
                        current_utt_id = utt_match.group(1)
                        in_target_alignment_section = True
                    continue
                elif line.startswith('Evenly spaced alignments for utterance'):
                    # Extract utterance ID from "Evenly spaced alignments for utterance ma028_1_0297:"
                    utt_match = re.match(r'Evenly spaced alignments for utterance (.+):', line)
                    if utt_match:
                        current_utt_id = utt_match.group(1)
                        in_target_alignment_section = True
                    continue
                elif line.startswith('Warning:'):
                    # Skip warning lines like "Warning: No Kaldi alignments for utterance..."
                    continue

                # Only process segment lines when in target alignment section
                if in_target_alignment_section and current_utt_id:
                    # Parse segment line: "  0: 621.862000 - 621.957000: ãƒ•"
                    seg_match = re.match(r'\s*(\d+):\s*(\d+\.\d+)\s*-\s*(\d+\.\d+):\s*(.+)', line)
                    if seg_match:
                        utt_id_index = int(seg_match.group(1))
                        begin_sec = float(seg_match.group(2))
                        end_sec = float(seg_match.group(3))
                        label = seg_match.group(4).strip()

                        s = Segment(begin_sec, end_sec, label, current_utt_id, utt_id_index)
                        segments.append(s)

    except FileNotFoundError:
        print(f"Warning: Kaldi debug file not found: {debug_file}")
    except Exception as e:
        print(f"Warning: Error reading {debug_file}: {e}")

    return segments

def load_meta_csv(meta_csv_path):
    """
    Load the meta CSV file containing demographic information.

    Args:
        meta_csv_path (str): Path to the meta CSV file

    Returns:
        pd.DataFrame: DataFrame containing meta information

    Note:
        Expected columns: session_id, session, subject, month, session_begin_sec, session_end_sec, speaker
        The 'session_id' column is used as the shared identifier
    """
    try:
        return pd.read_csv(meta_csv_path)
    except FileNotFoundError:
        print(f"Error: Meta CSV file not found: {meta_csv_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error: Failed to load meta CSV {meta_csv_path}: {e}")
        return pd.DataFrame()

def extend_and_cut_times(row):
    """
    Calculate extended and cut times for each segment.

    Args:
        row: DataFrame row containing segment information with 'begin_sec', 'end_sec', and 'duration'

    Returns:
        pd.Series: New columns with extended and cut timings:
            - ext_cut_begin_sec: Begin time for 0.5s segment extraction
            - ext_cut_end_sec: End time for 0.5s segment extraction

    Note:
        For segments < 0.5s: extends equally on both sides
        For segments >= 0.5s: extracts middle 0.5s portion
        Unlike the original script, we only calculate cut times as requested
    """
    target_duration = 0.5
    current_duration = row['duration']

    if current_duration < target_duration:
        # For short segments: extend equally on both sides to reach 0.5s
        extra_time_needed = target_duration - current_duration
        time_per_side = extra_time_needed / 2
        return pd.Series({
            'ext_cut_begin_sec': row['begin_sec'] - time_per_side,
            'ext_cut_end_sec': row['end_sec'] + time_per_side
        })
    else:
        # For long segments: extract middle 0.5s
        mid_point = (row['begin_sec'] + row['end_sec']) / 2
        return pd.Series({
            'ext_cut_begin_sec': mid_point - target_duration/2,
            'ext_cut_end_sec': mid_point + target_duration/2
        })

def main():
    """
    Main function to process NTT infant data and create CSV file.

    This function:
    1. Processes command line arguments
    2. Loads Kaldi debug files and meta CSV file
    3. Processes segment files for each session
    4. Merges segment data with meta information including speaker
    5. Creates extended timing information for feature extraction
    6. Saves results to CSV file
    """
    parser = argparse.ArgumentParser(description='Extract NTT infant segment information from Kaldi debug files to CSV. Segments longer than max_duration are removed (default 1000s effectively keeps all segments).')
    parser.add_argument('--meta_csv',
                       default='/data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/processed/all_kana_comment_token.csv',
                       help='CSV file containing meta information')
    parser.add_argument('--output_csv',
                       default='data/ntt_infant_phone/meta.csv',
                       help='Output CSV file path')
    parser.add_argument('--max_duration',
                       type=float,
                       default=1000.0,
                       help='Maximum allowed duration in seconds for segments (default: 1000.0)')
    parser.add_argument('--kaldi_debug_dir',
                       default='/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_ntt_s0/exp/out/all/fixed/debug/',
                       help='Directory containing Kaldi debug files')

    args = parser.parse_args()

    # Load meta CSV
    print("Loading meta CSV file...")
    meta_df = load_meta_csv(args.meta_csv)

    if meta_df.empty:
        print("Error: No data loaded from meta CSV file")
        return

    # Check if speaker column exists
    if 'speaker' not in meta_df.columns:
        print("Warning: 'speaker' column not found in meta CSV. Proceeding without speaker information.")
        has_speaker = False
    else:
        has_speaker = True
        print("Speaker column found in meta CSV")

    # Get list of session IDs from meta CSV
    session_ids = meta_df['session_id'].unique()

    # Create table for data
    table = []
    if has_speaker:
        table_header = ['dataid', 'audioid', 'age_months', 'begin_sec', 'end_sec', 'label', 'utt_id', 'utt_id_index', 'speaker']
    else:
        table_header = ['dataid', 'audioid', 'age_months', 'begin_sec', 'end_sec', 'label', 'utt_id', 'utt_id_index']

    print(f"Using Kaldi debug files from: {args.kaldi_debug_dir}")
    print("Processing both fixed alignments and evenly spaced alignments")

    # Create a mapping from utt_id to speaker for efficient lookup
    if has_speaker:
        # The session column in meta_csv contains the utt_id
        utt_speaker_map = dict(zip(meta_df['session'], meta_df['speaker']))
        print(f"Created speaker mapping for {len(utt_speaker_map)} utterances")

    # Process each session
    print(f"Processing {len(session_ids)} sessions...")
    processed_sessions = 0

    for session_id in session_ids:
        # Find matching meta information
        meta_row = meta_df[meta_df['session_id'] == session_id]
        if meta_row.empty:
            print(f"Warning: No meta information found for session {session_id}")
            continue

        # Extract meta information
        meta_info = meta_row.iloc[0]
        dataid = meta_info['subject']
        audioid = session_id
        age_months = int(meta_info['month'])

        # Read segments from Kaldi debug file
        debug_file = os.path.join(args.kaldi_debug_dir, f"{session_id}_debug.txt")
        segments = read_kaldi_debug_segments(debug_file)

        if not segments:
            print(f"Warning: No segments found for session {session_id}")
            continue

        # Add segments to table
        for seg in segments:
            row_data = [dataid, audioid, age_months,
                       seg.begin_sec, seg.end_sec, seg.label,
                       seg.utt_id, seg.utt_id_index]

            # Add speaker information if available
            if has_speaker:
                speaker = utt_speaker_map.get(seg.utt_id, 'unknown')
                row_data.append(speaker)

            table.append(row_data)

        processed_sessions += 1
        if processed_sessions % 100 == 0:
            print(f"Processed {processed_sessions} sessions...")

    print(f"Finished processing {processed_sessions} sessions")

    if not table:
        print("Error: No segments processed")
        return

    # Create DataFrame
    print("Creating DataFrame and applying filters...")
    df = pd.DataFrame(data=table, columns=table_header)

    # Calculate duration and filter
    df['duration'] = df['end_sec'] - df['begin_sec']
    initial_count = len(df)
    df = df[df['duration'] <= args.max_duration]  # Remove segments longer than max_duration
    df = df[df['duration'] > 0]  # Remove zero or negative duration segments
    filtered_count = len(df)

    print(f"Filtered out {initial_count - filtered_count} segments (duration > {args.max_duration}s or <= 0s)")

    # Add extended timing columns for feature extraction
    print("Adding extended timing information...")
    df[['ext_cut_begin_sec', 'ext_cut_end_sec']] = df.apply(extend_and_cut_times, axis=1)

    # Round all time columns before saving
    time_columns = ['begin_sec', 'end_sec', 'duration', 'ext_cut_begin_sec', 'ext_cut_end_sec']
    for col in time_columns:
        df[col] = df[col].round(6)

    # Save to CSV
    print("Saving to CSV...")
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"Created CSV file: {args.output_csv}")
    print(f"Total segments processed: {len(df)}")
    print(f"Segments longer than {args.max_duration}s were removed")

    # Print some statistics
    print("\nDataset statistics:")
    print(f"Number of subjects: {df['dataid'].nunique()}")
    print(f"Number of sessions: {df['audioid'].nunique()}")
    print(f"Number of utterances: {df['utt_id'].nunique()}")
    print(f"Age range: {df['age_months'].min()}-{df['age_months'].max()} months")
    print(f"Duration range: {df['duration'].min():.3f}-{df['duration'].max():.3f} seconds")
    print(f"Most frequent labels: {df['label'].value_counts().head().to_dict()}")

    if has_speaker:
        print(f"\nSpeaker distribution:")
        print(df['speaker'].value_counts().to_dict())

if __name__ == "__main__":
    main()

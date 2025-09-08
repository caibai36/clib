#!/usr/bin/env python3
"""
Script to extract and process NTT infant segment information and create a CSV dataset.
This script reads audio segment files from JSON info file, processes them with meta information,
and creates a CSV file containing segment information including timing, labels, and demographics.

The script processes audio segments and creates extended timing information for feature extraction:
- For segments shorter than 0.5s: extends them equally on both sides to reach 0.5s
- For segments longer than 0.5s: keeps original extent but adds cut points to extract middle 0.5s

Main features:
- Processes JSON info file containing segment file paths
- Merges with meta CSV file for demographic information
- Calculates age information from meta data
- Filters segments by maximum duration
- Creates extended timing information for consistent feature extraction
- Outputs results to CSV format
"""

import os
import re
import json
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
        low_freq (float): Lower frequency bound (optional, not used in NTT infant)
        high_freq (float): Upper frequency bound (optional, not used in NTT infant)

    Note:
        Frequencies are not used in NTT infant dataset but kept for compatibility
    """
    def __init__(self, begin_sec, end_sec, label, low_freq=None, high_freq=None):
        self.begin_sec = begin_sec
        self.end_sec = end_sec
        self.label = label
        self.low_freq = low_freq
        self.high_freq = high_freq

def read_ntt_segments(segment_file):
    """
    Read and parse a NTT infant segment file.

    Args:
        segment_file (str): Path to the NTT infant segment file

    Returns:
        list: List of Segment objects

    Example format:
        11.401000       11.491000       ア
        11.491000       11.681000       ク
        11.681000       11.891000       モ
        11.891000       12.001000       ッ
    """
    segments = []
    try:
        with open(segment_file, encoding='utf8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                elem = re.split(r"\s+", line)
                if len(elem) != 3:
                    print(f"Warning: Invalid format in {segment_file}\nLine: '{line}'")
                    continue

                begin_sec, end_sec, label = elem
                s = Segment(float(begin_sec), float(end_sec), str(label))
                segments.append(s)
    except FileNotFoundError:
        print(f"Warning: Segment file not found: {segment_file}")
    except Exception as e:
        print(f"Warning: Error reading {segment_file}: {e}")

    return segments

def load_info_json(info_json_path):
    """
    Load the JSON info file containing segment file paths.

    Args:
        info_json_path (str): Path to the JSON info file

    Returns:
        dict: Dictionary containing info data with session_id as keys

    Note:
        The JSON file contains paths to segment files and other metadata
        Each entry has 'seg' key pointing to the segment file path
    """
    try:
        with open(info_json_path, 'r', encoding='utf8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Info JSON file not found: {info_json_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {info_json_path}: {e}")
        return {}

def load_meta_csv(meta_csv_path):
    """
    Load the meta CSV file containing demographic information.

    Args:
        meta_csv_path (str): Path to the meta CSV file

    Returns:
        pd.DataFrame: DataFrame containing meta information

    Note:
        Expected columns: session_id, session, subject, month, session_begin_sec, session_end_sec
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
    2. Loads JSON info file and meta CSV file
    3. Processes segment files for each session
    4. Merges segment data with meta information
    5. Creates extended timing information for feature extraction
    6. Saves results to CSV file
    """
    parser = argparse.ArgumentParser(description='Extract NTT infant segment information to CSV. Segments longer than max_duration are removed (default 1000s effectively keeps all segments).')
    parser.add_argument('--info_json',
                       default='data/ntt_infant_phone/info.json',
                       help='JSON file containing segment file paths')
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

    args = parser.parse_args()

    # Load info JSON and meta CSV
    print("Loading info JSON and meta CSV files...")
    info_data = load_info_json(args.info_json)
    meta_df = load_meta_csv(args.meta_csv)

    if not info_data:
        print("Error: No data loaded from info JSON file")
        return

    if meta_df.empty:
        print("Error: No data loaded from meta CSV file")
        return

    # Create table for data
    table = []
    table_header = ['dataid', 'audioid', 'age_months',
                    'begin_sec', 'end_sec', 'label']

    # Process each session in info_data
    print(f"Processing {len(info_data)} sessions...")
    processed_sessions = 0

    for session_id, session_info in info_data.items():
        # Get segment file path
        seg_file_path = session_info.get('seg')
        if not seg_file_path:
            print(f"Warning: No segment file path for session {session_id}")
            continue

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

        # Read segments from file
        segments = read_ntt_segments(seg_file_path)
        if not segments:
            print(f"Warning: No segments found for session {session_id}")
            continue

        # Add segments to table
        for seg in segments:
            table.append([dataid, audioid, age_months,
                         seg.begin_sec, seg.end_sec, seg.label])

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
    print(f"Age range: {df['age_months'].min()}-{df['age_months'].max()} months")
    print(f"Duration range: {df['duration'].min():.3f}-{df['duration'].max():.3f} seconds")
    print(f"Most frequent labels: {df['label'].value_counts().head().to_dict()}")

if __name__ == "__main__":
    main()

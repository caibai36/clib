#!/usr/bin/env python3
"""
Script to extract and process audio segment information from jay_das_annotation dataset.
This script reads audio segment files, processes them with age information from CSV,
and creates a CSV dataset containing segment information including timing, labels, and frequencies.

Key differences from original:
- Age information comes from an external CSV file (not calculated from filename dates)
- Configurable column names for age and audio ID in the info CSV
- Validates that WAV files exist and gives warnings for missing segments or CSV entries
- Handles segment files that may have malformed lines (2 items instead of 3)
- Sorts segments by begin_sec time
"""

import os
import re
import argparse
import pandas as pd
import warnings

class Segment:
    """
    A class to represent an audio segment in Audacity format.

    Attributes:
        begin_sec (float): Start time of the segment in seconds
        end_sec (float): End time of the segment in seconds
        label (str): Label of the segment
        low_freq (float): Lower frequency bound (optional)
        high_freq (float): Upper frequency bound (optional)
    """
    def __init__(self, begin_sec, end_sec, label, low_freq=None, high_freq=None):
        self.begin_sec = begin_sec
        self.end_sec = end_sec
        self.label = label
        self.low_freq = low_freq
        self.high_freq = high_freq

def read_audacity_segments(segment_file):
    """
    Read and parse an Audacity segment file.

    Args:
        segment_file (str): Path to the Audacity segment file

    Returns:
        list: List of Segment objects sorted by begin_sec

    Note:
        - Skips lines that don't have exactly 3 elements (filters out malformed lines)
        - Handles frequency lines starting with backslash
        - Sorts segments by begin time before returning
    """
    with open(segment_file, encoding='utf8') as f:
        segments = []
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            elem = re.split(r"\s+", line)

            # Skip lines that don't have 3 elements
            if len(elem) != 3:
                continue

            first, second, third = elem
            if first != "\\":
                try:
                    s = Segment(float(first), float(second), str(third))
                    segments.append(s)
                except ValueError:
                    # Skip lines where first/second can't be converted to float
                    continue
            else:
                # Frequency line - attach to last segment if exists
                if segments:
                    try:
                        segments[-1].low_freq = float(second)
                        segments[-1].high_freq = float(third)
                    except ValueError:
                        pass

        # Sort segments by begin_sec
        segments.sort(key=lambda x: x.begin_sec)
        return segments

def extend_and_cut_times(row):
    """
    Calculate extended and cut times for each segment.

    Args:
        row: DataFrame row containing segment information with 'begin_sec', 'end_sec', and 'duration'

    Returns:
        pd.Series: New columns with extended and cut timings

    Note:
        For segments < 0.5s: extends equally on both sides
        For segments >= 0.5s: keeps original extent but adds cut points for middle 0.5s
    """
    target_duration = 0.5
    current_duration = row['duration']

    if current_duration < target_duration:
        extra_time_needed = target_duration - current_duration
        time_per_side = extra_time_needed / 2
        return pd.Series({
            'ext_begin_sec': row['begin_sec'] - time_per_side,
            'ext_end_sec': row['end_sec'] + time_per_side,
            'ext_cut_begin_sec': row['begin_sec'] - time_per_side,
            'ext_cut_end_sec': row['end_sec'] + time_per_side
        })
    else:
        mid_point = (row['begin_sec'] + row['end_sec']) / 2
        return pd.Series({
            'ext_begin_sec': row['begin_sec'],
            'ext_end_sec': row['end_sec'],
            'ext_cut_begin_sec': mid_point - target_duration/2,
            'ext_cut_end_sec': mid_point + target_duration/2
        })

def main():
    """
    Main function to process jay_das_annotation data and create CSV file.
    """
    parser = argparse.ArgumentParser(
        description='Extract audio segment information to CSV for jay_das_annotation dataset. '
                    'Segments longer than max_duration are removed.')
    parser.add_argument('--wav_dir',
                       default="/data04/home/bin-wu/workspace/projects/virtualbox/share/data/jay_das_annotation/b3_762F_763M_3201M/wav/",
                       help='Directory containing WAV files')
    parser.add_argument('--seg_dir',
                       default="/data04/home/bin-wu/workspace/projects/virtualbox/share/data/jay_das_annotation/b3_762F_763M_3201M/seg/",
                       help='Directory containing segment annotation files (Audacity format)')
    parser.add_argument('--info_csv',
                       default="/data04/home/bin-wu/workspace/projects/virtualbox/share/data/jay_das_annotation/b3_762F_763M_3201M/b3_2025-12-19outlineAGEid.csv",
                       help='CSV file containing age and ID information')
    parser.add_argument('--age_days_col',
                       default='Age',
                       help='Column name for age in days in the info CSV (default: Age)')
    parser.add_argument('--audio_id_col',
                       default='id',
                       help='Column name for audio ID in the info CSV (default: id)')
    parser.add_argument('--dataid',
                       default='b3_762F_763M_3201M',
                       help='Data ID for this dataset')
    parser.add_argument('--output_csv',
                       default="data/b3_762F_763M_3201M/b3_762F_763M_3201M.csv",
                       help='Output CSV file path')
    parser.add_argument('--max_duration',
                       type=float,
                       default=3.0,
                       help='Maximum allowed duration in seconds for segments (default: 3.0)')

    args = parser.parse_args()

    # Load the info CSV file
    print(f"Loading age information from: {args.info_csv}")
    info_df = pd.read_csv(args.info_csv)

    # Verify required columns exist
    if args.audio_id_col not in info_df.columns:
        raise ValueError(f"Column '{args.audio_id_col}' not found in {args.info_csv}. "
                        f"Available columns: {list(info_df.columns)}")
    if args.age_days_col not in info_df.columns:
        raise ValueError(f"Column '{args.age_days_col}' not found in {args.info_csv}. "
                        f"Available columns: {list(info_df.columns)}")

    # Create a mapping from id to age
    id_to_age = dict(zip(info_df[args.audio_id_col], info_df[args.age_days_col]))

    print(f"Loaded age information for {len(id_to_age)} audio files")
    print(f"Using columns: audio_id='{args.audio_id_col}', age_days='{args.age_days_col}'")

    # Get all WAV files
    wav_files = []
    if os.path.exists(args.wav_dir):
        wav_files = [f for f in os.listdir(args.wav_dir) if f.endswith('.wav')]
    else:
        raise FileNotFoundError(f"WAV directory not found: {args.wav_dir}")

    print(f"Found {len(wav_files)} WAV files")

    # Create table for data
    table = []
    table_header = ['dataid', 'audioid', 'age_days', 'age_weeks',
                    'begin_sec', 'end_sec', 'label', 'low_freq', 'high_freq']

    # Track warnings
    missing_seg_files = []
    missing_csv_entries = []
    processed_count = 0

    # Process each WAV file
    for wav_file in sorted(wav_files):
        audioid = os.path.splitext(wav_file)[0]  # Remove .wav extension

        # Check if seg file exists
        seg_file = os.path.join(args.seg_dir, audioid + '.txt')
        if not os.path.exists(seg_file):
            missing_seg_files.append(audioid)
            continue

        # Check if audioid is in CSV
        if audioid not in id_to_age:
            missing_csv_entries.append(audioid)
            continue

        # Get age information
        age_days = id_to_age[audioid]
        age_weeks = age_days // 7

        # Read segments
        segments = read_audacity_segments(seg_file)

        # Add segments to table
        for seg in segments:
            table.append([args.dataid, audioid, age_days, age_weeks,
                         seg.begin_sec, seg.end_sec, seg.label,
                         seg.low_freq, seg.high_freq])

        processed_count += 1

    # Print warnings
    if missing_seg_files:
        print(f"\nWARNING: {len(missing_seg_files)} WAV files have no corresponding segment file:")
        for audioid in missing_seg_files[:5]:  # Show first 5
            print(f"  - {audioid}")
        if len(missing_seg_files) > 5:
            print(f"  ... and {len(missing_seg_files) - 5} more")

    if missing_csv_entries:
        print(f"\nWARNING: {len(missing_csv_entries)} WAV files not found in info CSV:")
        for audioid in missing_csv_entries[:5]:  # Show first 5
            print(f"  - {audioid}")
        if len(missing_csv_entries) > 5:
            print(f"  ... and {len(missing_csv_entries) - 5} more")

    # Create DataFrame
    df = pd.DataFrame(data=table, columns=table_header)

    print(f"\nProcessed {processed_count} audio files successfully")
    print(f"Total segments before filtering: {len(df)}")

    # Filter out negative ages (if any)
    df = df[df['age_days'] >= 0]

    # Calculate duration and filter
    df['duration'] = df['end_sec'] - df['begin_sec']
    initial_count = len(df)
    df = df[df['duration'] <= args.max_duration]
    removed_count = initial_count - len(df)

    if removed_count > 0:
        print(f"Removed {removed_count} segments longer than {args.max_duration}s")

    # Add extended timing columns for feature extraction
    df[['ext_begin_sec', 'ext_end_sec', 'ext_cut_begin_sec', 'ext_cut_end_sec']] = \
        df.apply(extend_and_cut_times, axis=1)

    # Round all time columns before saving
    time_columns = ['begin_sec', 'end_sec', 'duration',
                   'ext_begin_sec', 'ext_end_sec',
                   'ext_cut_begin_sec', 'ext_cut_end_sec']
    for col in time_columns:
        df[col] = df[col].round(6)

    # Save to CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"\nCreated CSV file: {args.output_csv}")
    print(f"Total segments in output: {len(df)}")

if __name__ == "__main__":
    main()

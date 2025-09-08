#!/usr/bin/env python3
"""
Script to extract and process NAS5 segment information and create a CSV dataset.
This script reads audio segment files, processes them, and creates a CSV file
containing segment information including timing, labels, and frequencies.

The script processes audio segments and creates extended timing information for feature extraction:
- For segments shorter than 0.5s: extends them equally on both sides to reach 0.5s
- For segments longer than 0.5s: keeps original extent but adds cut points to extract middle 0.5s

Main features:
- Processes Audacity format segment files
- Calculates age information based on file dates
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
    A class to represent an audio segment in Audacity format.

    Attributes:
        begin_sec (float): Start time of the segment in seconds
        end_sec (float): End time of the segment in seconds
        label (str): Label of the segment
        low_freq (float): Lower frequency bound (optional)
        high_freq (float): Upper frequency bound (optional)

    Note:
        Frequencies are optional and may be None if not specified in the source file
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
        list: List of Segment objects

    Example format:
        0.108084        0.153355        tr
        \\       5468.574219     10856.178711
        0.675958        1.387807        ph
    """
    with open(segment_file, encoding='utf8') as f:
        segments = []
        for line in f:
            line = line.strip()
            elem = re.split(r"\s+", line)
            if len(elem) != 3:
                print(f"Warning: Invalid format in {segment_file}\nLine: '{line}'")
                continue

            first, second, third = elem
            if first != "\\":
                s = Segment(float(first), float(second), str(third))
                segments.append(s)
            else:
                segments[-1].low_freq = float(second)
                segments[-1].high_freq = float(third)
    return segments

def get_family_birth_date(pathname):
    """
    Get birth date for a family based on pathname.

    Args:
        pathname (str): Path containing family identifier

    Returns:
        datetime: Birth date of the family or None if family not found

    Note:
        Birth dates are hardcoded for known families. Update this dictionary
        when adding new families to the dataset.
    """
    family_birth_dates = {
        'b1_906F_1302M_3153M': datetime(2024, 3, 2),
        'b2_1305F_759M_3162F': datetime(2024, 3, 9),
        'b3_762F_763M_3121F': datetime(2024, 2, 2),
        'b4_1372F_1169M_3117F': datetime(2024, 1, 30),
        'familybooth_1594F_1449M_3010': datetime(2023, 7, 22),
        'jay_family': datetime(2023, 7, 22),
        # F2 generation birth dates from your table
        'b1_906F_1302M_3211F': datetime(2024, 8, 5),   # booth1_f2
        'b2_1305F_759M_3222M': datetime(2024, 8, 10),  # booth2_f2
        'b3_762F_763M_3201M': datetime(2024, 7, 6),    # booth3_f2
        'b4_1372F_1169M_3196M': datetime(2024, 7, 4),  # booth4_f2
        # mecp2 mutant marmoset wara
        'mecp2_wara': datetime(2023, 6, 4),
    }

    for family, birth_date in family_birth_dates.items():
        if family in pathname:
            return birth_date
    return None

def get_age(segments_file):
    """
    Calculate age information from segment filename.

    Args:
        segments_file (str): Path to segment file

    Returns:
        tuple: (total_days, weeks, days) or (None, None, None) if date cannot be parsed

    Note:
        Expects filename to start with either YYMMDD or YYYYMMDD format
    """
    filename = os.path.basename(segments_file)
    pathname = os.path.dirname(segments_file)

    if re.match(r'^20\d{6}', filename):
        file_date = datetime.strptime(filename[:8], "%Y%m%d")
    elif re.match(r'^\d{6}', filename):
        file_date = datetime.strptime(filename[:6], "%y%m%d")
    else:
        warnings.warn(f"Invalid date format in filename: {filename}")
        return None, None, None

    birth_date = get_family_birth_date(pathname)
    if not birth_date:
        return None, None, None

    age = file_date - birth_date
    weeks = age.days // 7
    days = age.days % 7
    return age.days, weeks, days

def extend_and_cut_times(row):
    """
    Calculate extended and cut times for each segment.

    Args:
        row: DataFrame row containing segment information with 'begin_sec', 'end_sec', and 'duration'

    Returns:
        pd.Series: New columns with extended and cut timings:
            - ext_begin_sec: Extended begin time
            - ext_end_sec: Extended end time
            - ext_cut_begin_sec: Begin time for 0.5s segment extraction
            - ext_cut_end_sec: End time for 0.5s segment extraction

    Note:
        For segments < 0.5s: extends equally on both sides
        For segments > 0.5s: keeps original extent but adds cut points for middle 0.5s
    """
    target_duration = 0.5
    current_duration = row['duration']

    if current_duration < target_duration:
        # For short segments: extend equally on both sides to reach 0.5s
        extra_time_needed = target_duration - current_duration
        time_per_side = extra_time_needed / 2
        return pd.Series({
            'ext_begin_sec': row['begin_sec'] - time_per_side,
            'ext_end_sec': row['end_sec'] + time_per_side,
            'ext_cut_begin_sec': row['begin_sec'] - time_per_side,
            'ext_cut_end_sec': row['end_sec'] + time_per_side
        })
    else:
        # For long segments: maintain original extent but add cut points to extract middle 0.5s
        mid_point = (row['begin_sec'] + row['end_sec']) / 2
        return pd.Series({
            'ext_begin_sec': row['begin_sec'],
            'ext_end_sec': row['end_sec'],
            'ext_cut_begin_sec': mid_point - target_duration/2,
            'ext_cut_end_sec': mid_point + target_duration/2
        })

def main():
    """
    Main function to process NAS5 data and create CSV file.

    This function:
    1. Processes command line arguments
    2. Walks through segment files in specified directory
    3. Extracts segment information and calculates ages
    4. Creates extended timing information for feature extraction
    5. Saves results to CSV file
    """
    parser = argparse.ArgumentParser(description='Extract NAS5 segment information to CSV. Segments longer than max_duration are removed.')
    parser.add_argument('--root',
                       default="/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/exp/out/cnn_model_b0family3010_best_dev/",
                       help='Root directory path')
    parser.add_argument('--dataid',
                       default='b2_f1',
                       help='Data ID')
    parser.add_argument('--seg_paths',
                       default="nas5/b2_1305F_759M_3162F",
                       help='Segment paths under the root. Only include segment files in audacity format in directories or sub-directories.')
    parser.add_argument('--output_csv',
                       default="data/nas5_b2_f1/b2_f1_cnn.csv",
                       help='Output CSV file path')
    parser.add_argument('--max_duration',
                       type=float,
                       default=3.0,
                       help='Maximum allowed duration in seconds for segments (default: 3.0)')

    args = parser.parse_args()

    # Get all segment files
    seg_path = os.path.join(args.root, args.seg_paths)
    seg_files = []
    for dirpath, _, filenames in os.walk(seg_path):
        for filename in filenames:
            seg_files.append(os.path.join(dirpath, filename))

    # Create table for data
    table = []
    table_header = ['dataid', 'audioid', 'age_days', 'age_weeks',
                    'begin_sec', 'end_sec', 'label', 'low_freq', 'high_freq']

    # Process each segment file
    for path in sorted(seg_files):
        file_base = os.path.basename(path)
        # Extract audioid from filename (assumes format ending with 'ch1')
        audioid = file_base[:file_base.find("ch1") + len("ch1")] if "ch1" in file_base else None

        segments = read_audacity_segments(path)
        total_days, weeks, days = get_age(path)

        for seg in segments:
            table.append([args.dataid, audioid, total_days, weeks,
                         seg.begin_sec, seg.end_sec, seg.label,
                         seg.low_freq, seg.high_freq])

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data=table, columns=table_header)
    df = df[df.age_days >= 0]  # Filter out negative ages

    # Calculate duration and filter
    df['duration'] = df['end_sec'] - df['begin_sec']
    df = df[df['duration'] <= args.max_duration]  # Remove segments longer than max_duration

    # Add extended timing columns for feature extraction
    df[['ext_begin_sec', 'ext_end_sec', 'ext_cut_begin_sec', 'ext_cut_end_sec']] = df.apply(extend_and_cut_times, axis=1)

    # Round all time columns before saving
    time_columns = ['begin_sec', 'end_sec', 'duration',
                   'ext_begin_sec', 'ext_end_sec',
                   'ext_cut_begin_sec', 'ext_cut_end_sec']
    for col in time_columns:
        df[col] = df[col].round(6)

    # Save to CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Created CSV file: {args.output_csv}")
    print(f"Total segments processed: {len(df)}")
    print(f"Segments longer than {args.max_duration}s were removed")

if __name__ == "__main__":
    main()

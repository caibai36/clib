#!/usr/bin/env python3
"""
Script to create extended CSV with segment-level information for marmoset dataset.
Similar to nas5_extract_csv_info.py but adapted for the autistic marmoset dataset.

Creates:
- data/local/marmoset_segments/marmoset_segments.csv (combined)
- data/exp1/exp1.csv (Experiment_1 only)
- data/exp2/exp2.csv (Experiment_2 only)
"""

import os
import re
import argparse
import pandas as pd
from pathlib import Path
import warnings

def read_audacity_segments(segment_file):
    """Read Audacity format segment file."""
    segments = []
    with open(segment_file, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                try:
                    begin_sec = float(parts[0])
                    end_sec = float(parts[1])
                    label = parts[2]
                    segments.append({
                        'begin_sec': begin_sec,
                        'end_sec': end_sec,
                        'label': label
                    })
                except ValueError as e:
                    warnings.warn(f"Could not parse line in {segment_file}: {line}")
                    continue
    return segments

def extend_and_cut_times(row):
    """Calculate extended and cut times for each segment."""
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

def create_segment_csv(info_csv, output_csv, exp1_csv, exp2_csv, max_duration=3.0):
    """
    Create segment-level CSV with extended timing information.

    Args:
        info_csv: Path to marmoset_info.csv
        output_csv: Output CSV file path (combined)
        exp1_csv: Output CSV for Experiment_1
        exp2_csv: Output CSV for Experiment_2
        max_duration: Maximum segment duration to include
    """
    # Load info CSV
    if not os.path.exists(info_csv):
        raise FileNotFoundError(f"Info CSV not found: {info_csv}")

    info_df = pd.read_csv(info_csv)
    print(f"Loaded {len(info_df)} records from {info_csv}")

    all_segments = []
    missing_files = []

    for idx, row in info_df.iterrows():
        audioid = row['audioid']
        seg_file = Path(row['seg_path'])

        if not seg_file.exists():
            missing_files.append(str(seg_file))
            warnings.warn(f"Segment file not found: {seg_file}")
            continue

        # Read segments
        segments = read_audacity_segments(seg_file)

        # Add to table with all relevant info
        for seg in segments:
            segment_record = {
                'audioid': audioid,
                'experiment': row['experiment'],
                'date': row['date'],
                'file_id': row['file_id'],
                'subject_name': row['subject_name'],
                'phenotype': row['phenotype'],
                'age_days': row['age_days'],
                'age_weeks': row['age_weeks'],
                'age_type': row['age_type'],
                'age_value': row['age_value'],
                'begin_sec': seg['begin_sec'],
                'end_sec': seg['end_sec'],
                'label': seg['label']
            }
            all_segments.append(segment_record)

    if not all_segments:
        raise ValueError("No segments were extracted. Check if segment files exist.")

    # Create DataFrame
    df = pd.DataFrame(all_segments)
    print(f"\nExtracted {len(df)} segments total")

    # Calculate duration
    df['duration'] = df['end_sec'] - df['begin_sec']

    # Filter by max duration
    original_len = len(df)
    df = df[df['duration'] <= max_duration]
    removed_count = original_len - len(df)
    if removed_count > 0:
        print(f"Removed {removed_count} segments longer than {max_duration}s")

    # Filter out segments with negative or zero duration
    df = df[df['duration'] > 0]

    # Add extended timing columns
    print("Calculating extended timing information...")
    df[['ext_begin_sec', 'ext_end_sec', 'ext_cut_begin_sec', 'ext_cut_end_sec']] = \
        df.apply(extend_and_cut_times, axis=1)

    # Round time columns
    time_columns = ['begin_sec', 'end_sec', 'duration',
                   'ext_begin_sec', 'ext_end_sec',
                   'ext_cut_begin_sec', 'ext_cut_end_sec']
    for col in time_columns:
        df[col] = df[col].round(6)

    # Save combined CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Created combined CSV: {output_csv}")

    # Split by experiment and save
    df_exp1 = df[df['experiment'] == 'Experiment_1'].copy()
    df_exp2 = df[df['experiment'] == 'Experiment_2'].copy()

    if len(df_exp1) > 0:
        os.makedirs(os.path.dirname(exp1_csv), exist_ok=True)
        df_exp1.to_csv(exp1_csv, index=False)
        print(f"✓ Created Experiment_1 CSV: {exp1_csv} ({len(df_exp1)} segments)")

    if len(df_exp2) > 0:
        os.makedirs(os.path.dirname(exp2_csv), exist_ok=True)
        df_exp2.to_csv(exp2_csv, index=False)
        print(f"✓ Created Experiment_2 CSV: {exp2_csv} ({len(df_exp2)} segments)")

    # Print summary statistics
    print(f"\n{'='*60}")
    print("SEGMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Total segments: {len(df)}")

    print("\nSegments by experiment:")
    exp_counts = df.groupby('experiment').size()
    for exp, count in exp_counts.items():
        print(f"  {exp}: {count}")

    print("\nSegments by phenotype:")
    phenotype_counts = df.groupby('phenotype').size()
    for pheno, count in phenotype_counts.items():
        print(f"  {pheno}: {count}")

    print("\nSegments by subject:")
    subject_counts = df.groupby(['subject_name', 'phenotype']).size()
    for (subj, pheno), count in subject_counts.items():
        print(f"  {subj} ({pheno}): {count}")

    print("\nTop 10 labels:")
    label_counts = df.groupby('label').size().sort_values(ascending=False)
    for i, (label, count) in enumerate(label_counts.head(10).items()):
        print(f"  {i+1}. {label}: {count}")

    if len(label_counts) > 10:
        print(f"  ... and {len(label_counts) - 10} more label types")

    print("\nDuration statistics:")
    print(f"  Mean: {df['duration'].mean():.3f}s")
    print(f"  Median: {df['duration'].median():.3f}s")
    print(f"  Std: {df['duration'].std():.3f}s")
    print(f"  Min: {df['duration'].min():.3f}s")
    print(f"  Max: {df['duration'].max():.3f}s")

    # Experiment-specific statistics
    print(f"\n{'='*60}")
    print("EXPERIMENT-SPECIFIC STATISTICS")
    print(f"{'='*60}")

    for exp_name, exp_df in [('Experiment_1', df_exp1), ('Experiment_2', df_exp2)]:
        if len(exp_df) > 0:
            print(f"\n{exp_name}:")
            print(f"  Total segments: {len(exp_df)}")
            print(f"  WT segments: {len(exp_df[exp_df['phenotype'] == 'WT'])}")
            print(f"  Mut segments: {len(exp_df[exp_df['phenotype'] == 'Mut'])}")
            print(f"  Unique labels: {exp_df['label'].nunique()}")
            print(f"  Age range: {exp_df['age_days'].min()}-{exp_df['age_days'].max()} days")
            print(f"  Duration mean: {exp_df['duration'].mean():.3f}s")

    if missing_files:
        print(f"\n{'='*60}")
        print(f"WARNING: {len(missing_files)} segment files not found:")
        for f in missing_files[:10]:  # Show first 10
            print(f"  • {f}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")

    return df, df_exp1, df_exp2

def main():
    parser = argparse.ArgumentParser(
        description='Create segment-level CSV with extended timing information'
    )
    parser.add_argument('--info_csv',
                       default='data/local/marmoset_info/marmoset_info.csv',
                       help='Path to marmoset_info.csv')
    parser.add_argument('--output_csv',
                       default='data/local/marmoset_segments/marmoset_segments.csv',
                       help='Output CSV file path (combined)')
    parser.add_argument('--exp1_csv',
                       default='data/exp1/exp1.csv',
                       help='Output CSV for Experiment_1')
    parser.add_argument('--exp2_csv',
                       default='data/exp2/exp2.csv',
                       help='Output CSV for Experiment_2')
    parser.add_argument('--max_duration',
                       type=float,
                       default=3.0,
                       help='Maximum segment duration in seconds')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("MARMOSET SEGMENT EXTRACTION")
    print(f"{'='*60}")
    print(f"Input: {args.info_csv}")
    print(f"Output (combined): {args.output_csv}")
    print(f"Output (Exp1): {args.exp1_csv}")
    print(f"Output (Exp2): {args.exp2_csv}")
    print(f"Max duration: {args.max_duration}s")

    try:
        create_segment_csv(args.info_csv, args.output_csv,
                          args.exp1_csv, args.exp2_csv, args.max_duration)
        print(f"\n{'='*60}")
        print("EXTRACTION COMPLETE")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR: {e}")
        print(f"{'='*60}\n")
        raise

if __name__ == "__main__":
    main()

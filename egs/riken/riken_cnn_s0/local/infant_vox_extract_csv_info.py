#!/usr/bin/env python3
import pandas as pd
import argparse
import os
import numpy as np

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
        # For long segments: maintain original extent but adds cut points for middle 0.5s
        mid_point = (row['begin_sec'] + row['end_sec']) / 2
        return pd.Series({
            'ext_begin_sec': row['begin_sec'],
            'ext_end_sec': row['end_sec'],
            'ext_cut_begin_sec': mid_point - target_duration/2,
            'ext_cut_end_sec': mid_point + target_duration/2
        })

def merge_tw_sil_sequences(df, sil_threshold=0.5):
    """
    Merge successive sequences of "tw" and "sil" vocalizations,
    breaking sequences when a "sil" is longer than the threshold.

    Args:
        df: DataFrame containing call segments with 'audioid', 'label', 'begin_sec', 'end_sec', 'duration'
        sil_threshold: Maximum duration (in seconds) of "sil" segments to be merged (default: 0.5s)

    Returns:
        DataFrame with merged sequences and "sil" entries removed
    """
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()

    # Fill missing calltype_id with -1 (unknown)
    result_df['calltype_id'] = result_df['calltype_id'].fillna(-1)

    # Process each audio file separately
    merged_rows = []
    for audioid, group in result_df.groupby('audioid'):
        # Sort by begin_sec to ensure correct sequence processing
        group = group.sort_values('begin_sec')

        # Variables for tracking the current sequence
        current_sequence = []
        in_sequence = False

        # Process each row
        for idx, row in group.iterrows():
            if row['label'] == 'tw':
                # Always add a tw to the current sequence
                current_sequence.append(row)
                in_sequence = True
            elif row['label'] == 'sil' and in_sequence:
                # If sil is longer than threshold, finish the current sequence
                if row['duration'] > sil_threshold:
                    if current_sequence:
                        process_sequence(current_sequence, merged_rows)
                        current_sequence = []
                        in_sequence = False
                else:
                    # Otherwise, add the sil to the current sequence
                    current_sequence.append(row)
            else:
                # For any other label, finish the current sequence and add the row
                if in_sequence and current_sequence:
                    process_sequence(current_sequence, merged_rows)
                    current_sequence = []
                    in_sequence = False

                # Don't add sil, noise rows, or rows with invalid labels
                if (row['label'] != 'sil' and row['label'] != 'noise' and
                    not pd.isna(row['label'])):
                    merged_rows.append(row)

        # Process any remaining sequence
        if in_sequence and current_sequence:
            process_sequence(current_sequence, merged_rows)

    # Create a new dataframe from the merged rows
    return pd.DataFrame(merged_rows)

def process_sequence(sequence, merged_rows):
    """
    Process a sequence of "tw" and "sil" calls and add the result to merged_rows.

    Args:
        sequence: List of rows containing a sequence of "tw" and "sil" calls
        merged_rows: List to append the processed results to
    """
    # Find all "tw" entries in the sequence
    tw_entries = [row for row in sequence if row['label'] == 'tw']

    if len(tw_entries) == 0:
        # No "tw" entries (shouldn't happen), just return
        return
    elif len(tw_entries) == 1:
        # Just one "tw" entry, add it unchanged
        merged_rows.append(tw_entries[0])
    else:
        # Multiple "tw" entries, merge them
        merged_row = tw_entries[0].copy()
        merged_row['end_sec'] = tw_entries[-1]['end_sec']
        merged_row['duration'] = merged_row['end_sec'] - merged_row['begin_sec']
        merged_rows.append(merged_row)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert marmoset vocalization labels to infant_vox.csv format')
    parser.add_argument('--input', type=str, default='/data/share/bin-wu/data/marmoset/vocalization/marmoset_vox/original/InfantMarmosetsVox/labels.csv',
                        help='Input labels.csv file path')
    parser.add_argument('--output', type=str, default='data/infant_vox/infant_vox.csv',
                        help='Output infant_vox.csv file path')
    parser.add_argument('--sil-threshold', type=float, default=0.5,
                        help='Maximum duration (in seconds) of silence to be merged (default: 0.5s)')
    args = parser.parse_args()

    # Define the mapping from calltype to label
    calltype_to_label = {
        1: "ph",
        2: "tw",
        3: "tr",
        4: "trph",
        5: "tsek",
        6: "ek",
        7: "cr",
        8: "tw-tr",
        9: "tw-ph",
        10: "pp",
        11: "sil",
        12: "noise"
    }

    # Read the input CSV file
    print(f"Reading input file: {args.input}")
    df = pd.read_csv(args.input)

    # Rename columns to match the desired output format
    df = df.rename(columns={
        'filename': 'audioid',
        'start': 'begin_sec',
        'end': 'end_sec',
        'calltype': 'calltype_id'
    })

    # Extract twin and marmoset IDs from audioid
    df['twinid'] = df['audioid'].str.extract(r'Twin(\d+)')
    df['marmosetid'] = df['audioid'].str.extract(r'marmoset(\d+)')

    # Calculate duration for all rows
    df['duration'] = df['end_sec'] - df['begin_sec']

    # Map calltype to label, handling missing values
    df['label'] = df['calltype_id'].map(calltype_to_label)

    # Log information about missing labels
    missing_labels_count = df['label'].isna().sum()
    if missing_labels_count > 0:
        print(f"Warning: Found {missing_labels_count} entries with missing labels. These will be excluded from processing.")

    # Log information about noise labels
    noise_labels_count = (df['label'] == 'noise').sum()
    if noise_labels_count > 0:
        print(f"Found {noise_labels_count} 'noise' entries. These will be excluded from processing.")

    # Merge tw-sil sequences, breaking on long silences, and remove sil entries
    print(f"Merging tw-sil sequences (sil threshold: {args.sil_threshold}s)...")
    df = merge_tw_sil_sequences(df, args.sil_threshold)

    # Apply the extend_and_cut_times function
    df[['ext_begin_sec', 'ext_end_sec', 'ext_cut_begin_sec', 'ext_cut_end_sec']] = df.apply(extend_and_cut_times, axis=1)

    # Round all time columns to 6 decimal places
    time_columns = ['begin_sec', 'end_sec', 'duration', 'ext_begin_sec',
                   'ext_end_sec', 'ext_cut_begin_sec', 'ext_cut_end_sec']
    for col in time_columns:
        if col in df.columns:
            df[col] = df[col].round(6)

    # Select and reorder columns to match the desired output format
    output_df = df[['audioid', 'label', 'begin_sec', 'end_sec', 'ext_cut_begin_sec',
                    'ext_cut_end_sec', 'twinid', 'marmosetid', 'caller']]

    # Double check no "noise" labels remain
    if (output_df['label'] == 'noise').sum() > 0:
        print("Warning: Some 'noise' labels remain after processing!")

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the output CSV file
    print(f"Writing output file: {args.output}")
    output_df.to_csv(args.output, index=False)
    print(f"Conversion complete. {len(output_df)} records processed.")

if __name__ == "__main__":
    main()

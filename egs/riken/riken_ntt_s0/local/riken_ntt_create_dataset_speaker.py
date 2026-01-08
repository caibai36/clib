#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset Creator for Speaker Information from Speech Corpus.

This script creates a structured dataset from CSV with speaker information.
It organizes data by divisions (e.g., subjects) and creates folders for
speaker-related annotations (speaker, noise, direction, loudness).

The output structure will be:
/output_dir/division/folder_name/audio_id.txt
Where each annotation file contains multiple lines with:
begin_sec\tend_sec\tlabel

Each line represents a segment within the audio file.
"""

import os
import argparse
import pandas as pd


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Create speaker dataset from CSV info with speaker tags.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--info_csv",
        type=str,
        default="data/local/all_kana_comment_token_speaker.csv",
        help="Path to CSV file containing session and speaker information"
    )
    parser.add_argument(
        "--wav_field",
        type=str,
        default="session_id",
        help="Field in CSV that corresponds to audio ID"
    )
    parser.add_argument(
        "--division_field",
        type=str,
        default="subject",
        help="Field in CSV to use for divisions (leave empty for no division)"
    )
    parser.add_argument(
        "--field2folder",
        type=str,
        default="speaker:speaker noise:noise loudness:loudness direction:direction original_tag:original_tag",
        help="Mapping of CSV fields to folder names (format: field:folder_name)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/local/ntt_infant_data_speaker",
        help="Output directory for the speaker dataset"
    )
    parser.add_argument(
        "--include_empty",
        action="store_true",
        help="Include entries even if the label is empty"
    )

    return parser.parse_args()


def parse_field2folder(field2folder_str):
    """
    Parse the field2folder mapping string.

    Args:
        field2folder_str (str): String in format "field1:folder1 field2:folder2 ..."

    Returns:
        dict: Dictionary mapping fields to folder names
    """
    field2folder = {}
    for mapping in field2folder_str.split():
        if ':' in mapping:
            field, folder = mapping.split(':', 1)
            field2folder[field] = folder

    if field2folder:
        print("Field to folder mappings:")
        for field, folder in field2folder.items():
            print(f"  {field} -> {folder}")

    return field2folder


def create_speaker_dataset(args):
    """
    Create the speaker dataset structure based on the provided arguments.

    Args:
        args (argparse.Namespace): Command line arguments
    """
    # Get absolute path for output directory
    output_dir_abs = os.path.abspath(args.output_dir)

    print(f"Creating speaker dataset using:")
    print(f"  Info CSV: {args.info_csv}")
    print(f"  Output directory: {output_dir_abs}")

    # Parse field to folder mapping
    field2folder = parse_field2folder(args.field2folder)
    if not field2folder:
        print("Error: No valid field to folder mappings provided")
        return

    # Read the CSV file
    try:
        print(f"Reading CSV file: {args.info_csv}")
        df = pd.read_csv(args.info_csv)
        print(f"Loaded {len(df)} entries from CSV")

        # Verify required columns exist
        required_columns = [args.wav_field, "session_begin_sec", "session_end_sec"]
        for col in required_columns:
            if col not in df.columns:
                print(f"Error: Required column '{col}' not found in CSV")
                return

        # Verify speaker-related fields exist
        for field in field2folder.keys():
            if field not in df.columns:
                print(f"Warning: Field '{field}' not found in CSV")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Create output directory
    os.makedirs(output_dir_abs, exist_ok=True)

    # Group by division if specified
    if args.division_field and args.division_field in df.columns:
        print(f"Grouping data by {args.division_field}")
        grouped = df.groupby(args.division_field)
    else:
        print("No division specified or field not found, treating as a single group")
        grouped = [(None, df)]

    # Statistics
    total_stats = {field: {'total': 0, 'non_empty': 0} for field in field2folder.keys()}

    # Process each division
    for division, group_df in grouped:
        division_str = str(division) if division is not None else "default"
        print(f"\n{'='*60}")
        print(f"Processing division: {division_str} ({len(group_df)} entries)")
        print(f"{'='*60}")

        # Create division directory
        division_dir = os.path.join(output_dir_abs, division_str)
        os.makedirs(division_dir, exist_ok=True)

        # Statistics for this division
        division_stats = {field: {'total': 0, 'non_empty': 0} for field in field2folder.keys()}

        # Create folder for each field mapping
        for field, folder_name in field2folder.items():
            if field not in df.columns:
                print(f"Warning: Field '{field}' not found in CSV, skipping")
                continue

            # Create folder
            folder_path = os.path.join(division_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            # Group by audio_id (session_id)
            audio_groups = group_df.groupby(args.wav_field)

            # Process each audio file
            files_created = 0
            for audio_id, audio_df in audio_groups:
                # Sort by begin time to ensure correct order
                audio_df = audio_df.sort_values(by="session_begin_sec")

                # Create one file per audio ID containing all segments
                annotation_file = os.path.join(folder_path, f"{audio_id}.txt")

                with open(annotation_file, 'w', encoding='utf-8') as f:
                    segments_written = 0
                    # Process each segment/utterance in this audio file
                    for _, row in audio_df.iterrows():
                        # Only process if we have begin/end times
                        if pd.notna(row["session_begin_sec"]) and pd.notna(row["session_end_sec"]):
                            begin_sec = float(row["session_begin_sec"])
                            end_sec = float(row["session_end_sec"])

                            # Get the label value from the field
                            label = row[field] if pd.notna(row[field]) else ""

                            # Track statistics
                            division_stats[field]['total'] += 1
                            total_stats[field]['total'] += 1

                            if label and label.strip():
                                division_stats[field]['non_empty'] += 1
                                total_stats[field]['non_empty'] += 1

                            # Write segment (include empty labels if --include_empty)
                            if args.include_empty or (label and label.strip()):
                                f.write(f"{begin_sec:.6f}\t{end_sec:.6f}\t{label}\n")
                                segments_written += 1

                if segments_written > 0:
                    files_created += 1

            print(f"  {folder_name}: created {files_created} files")
            print(f"    Total segments: {division_stats[field]['total']}")
            print(f"    Non-empty segments: {division_stats[field]['non_empty']}")
            if division_stats[field]['total'] > 0:
                pct = 100.0 * division_stats[field]['non_empty'] / division_stats[field]['total']
                print(f"    Coverage: {pct:.1f}%")

    # Print overall statistics
    print(f"\n{'='*60}")
    print("Overall Statistics:")
    print(f"{'='*60}")
    for field, folder_name in field2folder.items():
        if field in total_stats:
            total = total_stats[field]['total']
            non_empty = total_stats[field]['non_empty']
            if total > 0:
                pct = 100.0 * non_empty / total
                print(f"{folder_name}:")
                print(f"  Total segments: {total}")
                print(f"  Non-empty segments: {non_empty}")
                print(f"  Coverage: {pct:.1f}%")


def main():
    """Main function to create the speaker dataset."""
    args = parse_arguments()
    create_speaker_dataset(args)

    # Get absolute path for output directory
    output_dir_abs = os.path.abspath(args.output_dir)
    print(f"\n{'='*60}")
    print(f"Speaker dataset creation complete!")
    print(f"Files saved to: {output_dir_abs}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


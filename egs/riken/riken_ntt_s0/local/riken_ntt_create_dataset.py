#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset Creator for Speech Corpus.

This script creates a structured dataset from CSV information and wav.scp file.
It organizes data by divisions (e.g., subjects) and creates folders for different
types of annotations (text, kana, etc.).

The output structure will be:
/output_dir/division/folder_name/audio_id.txt
Where each annotation file contains multiple lines with:
begin_sec\tend_sec\tlabel

Each line represents a segment within the audio file.
"""

import os
import csv
import argparse
import shutil
import pandas as pd
from pathlib import Path


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Create dataset from CSV info and wav.scp files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--wav_scp",
        type=str,
        default="data/local/wav.scp",
        help="Path to wav.scp file mapping audio_id to audio_path"
    )
    parser.add_argument(
        "--info_csv",
        type=str,
        default="data/local/all_kana_comment_token.csv",
        help="Path to CSV file containing session information"
    )
    parser.add_argument(
        "--wav_field",
        type=str,
        default="session_id",
        help="Field in CSV that corresponds to audio ID in wav.scp"
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
        default="text:text kana:kana kana_token:kana_token kana_comment_token:kana_comment_token comment:comment",
        help="Mapping of CSV fields to folder names (format: field:folder_name)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/local/ntt_infant_data",
        help="Output directory for the dataset"
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symbolic links to wav files instead of copying them"
    )

    return parser.parse_args()


def load_wav_scp(file_path):
    """
    Load wav.scp file mapping audio IDs to file paths.

    The wav.scp file should have the format:
    <audio_id> <audio_path>

    Args:
        file_path (str): Path to wav.scp file

    Returns:
        dict: Dictionary mapping audio IDs to file paths
    """
    wav_scp = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                # Split by whitespace, but only into two parts (ID and path)
                parts = line.split(maxsplit=1)
                if len(parts) == 2:
                    audio_id, audio_path = parts
                    wav_scp[audio_id] = audio_path

        print(f"Loaded {len(wav_scp)} audio file mappings from {file_path}")
        return wav_scp
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return {}


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


def create_dataset(args):
    """
    Create the dataset structure based on the provided arguments.

    Args:
        args (argparse.Namespace): Command line arguments
    """
    # Get absolute path for output directory
    output_dir_abs = os.path.abspath(args.output_dir)

    print(f"Creating dataset using:")
    print(f"  WAV SCP: {args.wav_scp}")
    print(f"  Info CSV: {args.info_csv}")
    print(f"  Output directory: {output_dir_abs}")

    # Load wav.scp
    wav_scp = load_wav_scp(args.wav_scp)
    if not wav_scp:
        print("Error: No valid entries found in wav.scp")
        return

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
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Create output directory
    os.makedirs(output_dir_abs, exist_ok=True)

    # Group by division if specified
    if args.division_field and args.division_field in df.columns:
        print(f"Grouping data by {args.division_field}")
        # Group by division field
        grouped = df.groupby(args.division_field)
    else:
        print("No division specified or field not found, treating as a single group")
        # No division, treat as a single group
        grouped = [(None, df)]

    # Process each division
    for division, group_df in grouped:
        division_str = str(division) if division is not None else "default"
        print(f"\nProcessing division: {division_str} ({len(group_df)} entries)")

        # Create division directory
        division_dir = os.path.join(output_dir_abs, division_str)
        os.makedirs(division_dir, exist_ok=True)

        # Create folder for each field mapping
        for field, folder_name in field2folder.items():
            if field not in df.columns:
                print(f"Warning: Field '{field}' not found in CSV")
                continue

            # Create folder
            folder_path = os.path.join(division_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            print(f"  Created folder: {folder_name}")

            # First group by audio_id (session_id)
            audio_groups = group_df.groupby(args.wav_field)

            # Process each audio file
            for audio_id, audio_df in audio_groups:
                # Sort by begin time to ensure correct order
                audio_df = audio_df.sort_values(by="session_begin_sec")

                # Create one file per audio ID containing all segments
                annotation_file = os.path.join(folder_path, f"{audio_id}.txt")

                with open(annotation_file, 'w', encoding='utf-8') as f:
                    # Process each segment/utterance in this audio file
                    for _, row in audio_df.iterrows():
                        # Only process if we have begin/end times
                        if pd.notna(row["session_begin_sec"]) and pd.notna(row["session_end_sec"]):
                            begin_sec = float(row["session_begin_sec"])
                            end_sec = float(row["session_end_sec"])

                            # Get the label value from the field
                            label = row[field] if pd.notna(row[field]) else ""

                            # Write this segment to the file
                            f.write(f"{begin_sec:.6f}\t{end_sec:.6f}\t{label}\n")

                print(f"  Created annotation file: {audio_id}.txt ({len(audio_df)} segments)")

        # Create wav directory for this division
        wav_folder = os.path.join(division_dir, "wav")
        os.makedirs(wav_folder, exist_ok=True)
        print(f"  Created folder: wav")

        # Get unique audio IDs for this division
        unique_wav_ids = group_df[args.wav_field].unique()
        print(f"  Processing {len(unique_wav_ids)} unique audio files")

        # Process WAV files for this division
        wav_files_processed = 0
        for wav_id in unique_wav_ids:
            if wav_id in wav_scp:
                src_path = wav_scp[wav_id]
                dst_path = os.path.join(wav_folder, f"{wav_id}.wav")

                # Check if destination already exists
                if os.path.exists(dst_path):
                    # Remove existing file or symlink
                    os.remove(dst_path)

                # Create symlink or copy based on the --symlink flag
                if args.symlink:
                    try:
                        os.symlink(src_path, dst_path)
                        print(f"  Created symlink: {wav_id} -> {src_path}")
                    except OSError as e:
                        print(f"  Error creating symlink for {wav_id}: {e}")
                        print(f"  Falling back to copy for {wav_id}")
                        shutil.copy2(src_path, dst_path)
                else:
                    # Default behavior: copy the file
                    shutil.copy2(src_path, dst_path)
                    print(f"  Copied file: {wav_id}")

                wav_files_processed += 1
            else:
                print(f"  Warning: Audio ID {wav_id} not found in wav.scp")

        print(f"  Processed {wav_files_processed} wav files for division {division_str}")


def main():
    """Main function to create the dataset."""
    args = parse_arguments()
    create_dataset(args)

    # Get absolute path for output directory
    output_dir_abs = os.path.abspath(args.output_dir)
    print(f"\nDataset creation complete. Files saved to {output_dir_abs}")


if __name__ == "__main__":
    main()

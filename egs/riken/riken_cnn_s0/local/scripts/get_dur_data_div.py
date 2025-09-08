#!/usr/bin/env python3
"""
Compute total duration of dataset splits from YAML division file and wav.scp.

This script reads a YAML file containing dataset split definitions and a wav.scp file
mapping audio IDs to file paths, then calculates the total duration for each dataset split.
Works with any split names (train, dev, test, val, etc.).
"""
import yaml
import librosa
import os
import argparse
from collections import defaultdict

def load_wav_scp(scp_file):
    """
    Load wav.scp file and return a dictionary mapping id to file path.

    Args:
        scp_file (str): Path to wav.scp file

    Returns:
        dict: Dictionary mapping wav_id to wav_path
    """
    wav_dict = {}
    with open(scp_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    wav_id, wav_path = parts
                    wav_dict[wav_id] = wav_path
    return wav_dict

def get_audio_duration(wav_path):
    """
    Get duration of audio file in seconds.

    Args:
        wav_path (str): Path to audio file

    Returns:
        float: Duration in seconds, 0 if error occurred
    """
    try:
        duration = librosa.get_duration(path=wav_path)
        return duration
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return 0

def compute_set_durations(yaml_file, scp_file, verbose=False):
    """
    Compute total duration for all dataset splits defined in YAML file.

    Args:
        yaml_file (str): Path to YAML file containing data division
        scp_file (str): Path to wav.scp file
        verbose (bool): Whether to show individual file processing

    Returns:
        tuple: (durations dict, file_counts dict, split_names list)
    """
    # Load the division yaml file
    with open(yaml_file, 'r') as f:
        data_division = yaml.safe_load(f)

    # Load wav.scp mapping
    wav_dict = load_wav_scp(scp_file)

    # Initialize duration counters
    durations = defaultdict(float)
    file_counts = defaultdict(int)

    # Find all dataset split keys (excluding non-list values like comments)
    split_names = []
    for key, value in data_division.items():
        if isinstance(value, list) and value:  # Only process non-empty lists
            split_names.append(key)

    if not split_names:
        print("Warning: No valid dataset splits found in YAML file")
        return durations, file_counts, split_names

    print(f"Found dataset splits: {', '.join(split_names)}")

    # Process each split
    for set_name in split_names:
        if verbose:
            print(f"\nProcessing {set_name} set...")

        missing_files = []
        missing_ids = []

        for wav_id in data_division[set_name]:
            if wav_id in wav_dict:
                wav_path = wav_dict[wav_id]
                if os.path.exists(wav_path):
                    duration = get_audio_duration(wav_path)
                    durations[set_name] += duration
                    file_counts[set_name] += 1
                    if verbose:
                        print(f"  {wav_id}: {duration:.2f}s")
                else:
                    missing_files.append((wav_id, wav_path))
            else:
                missing_ids.append(wav_id)

        # Report missing files/IDs for this split
        if missing_files:
            print(f"  Warning: {len(missing_files)} files not found in {set_name} set:")
            for wav_id, wav_path in missing_files[:5]:  # Show first 5
                print(f"    {wav_id}: {wav_path}")
            if len(missing_files) > 5:
                print(f"    ... and {len(missing_files) - 5} more")

        if missing_ids:
            print(f"  Warning: {len(missing_ids)} IDs not found in wav.scp for {set_name} set:")
            for wav_id in missing_ids[:5]:  # Show first 5
                print(f"    {wav_id}")
            if len(missing_ids) > 5:
                print(f"    ... and {len(missing_ids) - 5} more")

    return durations, file_counts, split_names

def main():
    """Main function to parse arguments and compute durations."""
    parser = argparse.ArgumentParser(
        description="Compute total duration of dataset splits from YAML division file and wav.scp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
File Format Requirements:

YAML Division File Format:
  The YAML file should contain dataset splits as key-value pairs where keys are
  split names and values are lists of audio IDs.

  Example:
    train:
      - sk055_4
      - ma027_3
      - kk001_1

    dev:
      - mk026_2
      - ma002_4

    test:
      - sk070_7
      - ma009_2

  Alternative formats (all supported):
    train: [sk055_4, ma027_3, kk001_1]  # Inline list format
    validation: [mk026_2]               # Any split name works
    test_clean: [sk070_7]              # Custom split names

  Notes:
    - Comments (lines starting with #) are ignored
    - Only keys with non-empty list values are processed as dataset splits
    - Metadata keys with non-list values are ignored

WAV.scp File Format:
  Kaldi-style format mapping audio IDs to file paths. Each line contains:
  <audio_id> <space> <path_to_audio_file>

  Example:
    kk001_1 /path/to/audio/kk001_1.wav
    kk001_2 /path/to/audio/kk001_2.wav
    sk055_4 /path/to/audio/sk055_4.wav
    ma027_3 /path/to/audio/ma027_3.wav

  Notes:
    - One audio file per line
    - Space-separated: ID followed by file path
    - Paths can be absolute or relative to script execution directory
    - Comments (lines starting with #) are ignored
    - Supports various audio formats: .wav, .flac, .mp3, .m4a, etc.
    - Audio IDs in YAML must match IDs in this file

Usage Examples:
  %(prog)s
  %(prog)s --data_div_yaml custom_division.yaml --wav_scp custom_wav.scp
  %(prog)s -d conf/data/splits.yaml -w data/wav.scp --verbose --sort-by-duration
  python local/scripts/get_dur_data_div.py --data_div_yaml ../riken_cnn_s0/conf/data/division_jay_half.yaml --wav_scp ../riken_cnn_s0/data/riken2024/wav.scp
  python local/scripts/get_dur_data_div.py --data_div_yaml conf/data/division_ntt_riken_model.yaml --wav_scp data/ntt_infant_phone/wav.scp
        """
    )

    parser.add_argument(
        '--data_div_yaml', '-d',
        type=str,
        default='conf/data/division_ntt_riken_model.yaml',
        help='''Path to YAML file containing dataset division.
Format: split_name: [list_of_audio_ids].
Example: train: [id1, id2], dev: [id3, id4]
(default: %(default)s)'''
    )

    parser.add_argument(
        '--wav_scp', '-w',
        type=str,
        default='data/ntt_infant/wav.scp',
        help='''Path to wav.scp file in Kaldi format.
Format: <audio_id> <path_to_audio_file> (space-separated, one per line).
Example: id1 /path/to/audio1.wav
(default: %(default)s)'''
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output showing individual file durations and processing details'
    )

    parser.add_argument(
        '--sort-by-duration',
        action='store_true',
        help='Sort dataset splits by total duration (longest first) in the output summary'
    )

    args = parser.parse_args()

    # Check if files exist
    if not os.path.exists(args.data_div_yaml):
        print(f"Error: YAML file not found: {args.data_div_yaml}")
        print("Please ensure the file exists and follows the required YAML format.")
        print("Run with --help to see format requirements.")
        return 1

    if not os.path.exists(args.wav_scp):
        print(f"Error: wav.scp file not found: {args.wav_scp}")
        print("Please ensure the file exists and follows the Kaldi wav.scp format.")
        print("Run with --help to see format requirements.")
        return 1

    print(f"Data division YAML: {args.data_div_yaml}")
    print(f"wav.scp file: {args.wav_scp}")

    # Compute durations
    durations, file_counts, split_names = compute_set_durations(
        args.data_div_yaml, args.wav_scp, args.verbose
    )

    if not split_names:
        print("\nNo valid dataset splits found in YAML file.")
        print("Please check that your YAML file contains keys with list values.")
        print("Run with --help to see required format.")
        return 1

    # Sort splits by duration if requested
    if args.sort_by_duration:
        split_names = sorted(split_names, key=lambda x: durations[x], reverse=True)

    # Print summary
    print("\n" + "="*60)
    print("DURATION SUMMARY")
    print("="*60)

    total_duration = 0
    total_files = 0

    for set_name in split_names:
        duration_sec = durations[set_name]
        duration_min = duration_sec / 60
        duration_hour = duration_sec / 3600
        files = file_counts[set_name]

        print(f"{set_name.upper():>10} SET:")
        print(f"  Files: {files:>6}")
        print(f"  Duration: {duration_sec:>8.2f}s ({duration_min:>6.2f}min / {duration_hour:>5.2f}h)")
        if files > 0:
            avg_duration = duration_sec / files
            print(f"  Average:  {avg_duration:>8.2f}s per file")
        print()

        total_duration += duration_sec
        total_files += files

    total_min = total_duration / 60
    total_hour = total_duration / 3600

    print(f"{'TOTAL':>10}:")
    print(f"  Files: {total_files:>6}")
    print(f"  Duration: {total_duration:>8.2f}s ({total_min:>6.2f}min / {total_hour:>5.2f}h)")
    if total_files > 0:
        avg_total = total_duration / total_files
        print(f"  Average:  {avg_total:>8.2f}s per file")

    return 0

if __name__ == "__main__":
    exit(main())

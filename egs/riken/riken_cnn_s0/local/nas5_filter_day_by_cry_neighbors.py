#!/usr/bin/env python3
"""
Filter data to keep cry sounds and their neighboring sounds.

This script filters data from input files, keeping only cry sounds (label='cr')
and any sounds that occur within a specified time threshold before or after each cry.
The time of a call is defined as the middle time point between ext_cut_begin_sec and ext_cut_end_sec.

Usage:
    python filter_by_cry_neighbors.py [--input_data INPUT_DATA] [--input_config INPUT_CONFIG]
                                    [--output_data OUTPUT_DATA] [--output_config OUTPUT_CONFIG]
                                    [--threshold THRESHOLD]
"""

import argparse
import numpy as np
import pandas as pd
import os


def filter_by_cry_neighbors(input_data_path, input_config_path, output_data_path, output_config_path, threshold=2.0):
    """
    Filter data to keep cry sounds and their neighboring sounds within a time threshold.

    Parameters:
    -----------
    input_data_path : str
        Path to input data file (.npy)
    input_config_path : str
        Path to input config file (.csv)
    output_data_path : str
        Path to output filtered data file (.npy)
    output_config_path : str
        Path to output filtered config file (.csv)
    threshold : float, optional
        Time threshold in seconds for neighboring sounds (before and after each cry), default=2.0

    Returns:
    --------
    tuple
        (filtered_data, filtered_config) - the filtered numpy array and pandas DataFrame
    """
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_config_path), exist_ok=True)

    # Load input data
    print(f"Loading data from {input_data_path}")
    input_data = np.load(input_data_path)

    # Load config file
    print(f"Loading config from {input_config_path}")
    config_df = pd.read_csv(input_config_path)

    # Calculate mid-point time for each sound
    config_df['mid_time'] = (config_df['ext_cut_begin_sec'] + config_df['ext_cut_end_sec']) / 2

    # Create a mask for cry sounds
    cry_mask = config_df['label'] == 'cr'
    cry_rows = config_df[cry_mask]

    # Initialize an empty mask for the final result
    keep_mask = np.zeros(len(config_df), dtype=bool)

    # Find all neighboring sounds within threshold for each cry
    for _, cry_row in cry_rows.iterrows():
        cry_time = cry_row['mid_time']
        audioid = cry_row['audioid']

        # Find neighbors in the same audio file
        same_audio_mask = config_df['audioid'] == audioid
        time_diff = abs(config_df['mid_time'] - cry_time)

        # Keep sounds that are within threshold
        neighbors_mask = (time_diff <= threshold) & same_audio_mask
        keep_mask = keep_mask | neighbors_mask

    # Apply the mask to filter the config dataframe
    filtered_config = config_df[keep_mask]

    # Get indices from the filtered config
    indices = filtered_config.index

    # Filter the data array using the indices (adjust for header row)
    filtered_data = input_data[indices - 1]

    # Save filtered data and config
    print(f"Saving filtered data to {output_data_path}")
    np.save(output_data_path, filtered_data)

    print(f"Saving filtered config to {output_config_path}")
    # Remove the temporary mid_time column before saving
    filtered_config = filtered_config.drop(columns=['mid_time'])
    filtered_config.to_csv(output_config_path, index=False)

    print(f"Original data shape: {input_data.shape}")
    print(f"Filtered data shape: {filtered_data.shape}")
    print(f"Original config rows: {len(config_df)}")
    print(f"Filtered config rows: {len(filtered_config)}")
    print(f"Number of cry sounds: {len(cry_rows)}")

    return filtered_data, filtered_config


def parse_arguments():
    parser = argparse.ArgumentParser(description="Filter data to keep cry sounds and their neighboring sounds")

    parser.add_argument("--input_data", type=str,
                        default="exp/data/division_nas5_b1_mae_pretrained_all_days/train_input.npy",
                        help="Path to input data file (.npy)")
    parser.add_argument("--input_config", type=str,
                        default="exp/data/division_nas5_b1_mae_pretrained_all_days/train_config.csv",
                        help="Path to input config file (.csv)")
    parser.add_argument("--output_data", type=str,
                        default="exp/caller_identification/cry_neighbors_b1_0day_157days_within1s.npy",
                        help="Path to output filtered data file (.npy)")
    parser.add_argument("--output_config", type=str,
                        default="exp/caller_identification/cry_neighbors_b1_0day_157days_within1s_config.csv",
                        help="Path to output filtered config file (.csv)")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Time threshold in seconds for neighboring sounds (before and after each cry), default=1.0")

    return parser.parse_args()


def main():
    args = parse_arguments()

    filter_by_cry_neighbors(
        args.input_data,
        args.input_config,
        args.output_data,
        args.output_config,
        args.threshold
    )


if __name__ == "__main__":
    main()

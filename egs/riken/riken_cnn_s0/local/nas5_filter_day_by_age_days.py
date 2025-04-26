#!/usr/bin/env python3
"""
Filter data based on age in days.

This script filters data from input files based on the age_days column in the config file,
keeping only records where m <= age_days < n (m included, n not included).
Each line in the config file corresponds to one day of data.

Usage:
    python filter_by_age.py [--input_data INPUT_DATA] [--input_config INPUT_CONFIG] 
                           [--output_data OUTPUT_DATA] [--output_config OUTPUT_CONFIG]
                           [--begin_day BEGIN_DAY] [--end_day END_DAY]
"""

import argparse
import numpy as np
import pandas as pd
import os


def filter_by_age_days(input_data_path, input_config_path, output_data_path, output_config_path, begin_day=0, end_day=36):
    """
    Filter data based on age_days column from the config file.
    
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
    begin_day : int, optional
        Beginning day to include (inclusive), default=0
    end_day : int, optional
        Ending day to include (exclusive), default=36
        
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
    
    # Filter by age_days
    print(f"Filtering data for age_days between {begin_day} (inclusive) and {end_day} (exclusive)")
    mask = (config_df['age_days'] >= begin_day) & (config_df['age_days'] < end_day)
    filtered_config = config_df[mask]
    
    # Get indices from the filtered config (skip header row)
    indices = filtered_config.index
    
    # Filter the data array using the indices
    filtered_data = input_data[indices - 1]  # Adjust for header row in CSV
    
    # Save filtered data and config
    print(f"Saving filtered data to {output_data_path}")
    np.save(output_data_path, filtered_data)
    
    print(f"Saving filtered config to {output_config_path}")
    filtered_config.to_csv(output_config_path, index=False)
    
    print(f"Original data shape: {input_data.shape}")
    print(f"Filtered data shape: {filtered_data.shape}")
    print(f"Original config rows: {len(config_df)}")
    print(f"Filtered config rows: {len(filtered_config)}")
    
    return filtered_data, filtered_config


def parse_arguments():
    parser = argparse.ArgumentParser(description="Filter data based on age in days")
    
    parser.add_argument("--input_data", type=str, 
                        default="exp/data/division_nas5_b3_mae_pretrained_all_days/train_input.npy",
                        help="Path to input data file (.npy)")
    parser.add_argument("--input_config", type=str, 
                        default="exp/data/division_nas5_b3_mae_pretrained_all_days/train_config.csv",
                        help="Path to input config file (.csv)")
    parser.add_argument("--output_data", type=str, 
                        default="exp/caller_identification/parents_b3_0day_35days.npy",
                        help="Path to output filtered data file (.npy)")
    parser.add_argument("--output_config", type=str, 
                        default="exp/caller_identification/parents_b3_0day_35days_config.csv",
                        help="Path to output filtered config file (.csv)")
    parser.add_argument("--begin_day", type=int, default=0,
                        help="Beginning day to include (inclusive), default=0")
    parser.add_argument("--end_day", type=int, default=36,
                        help="Ending day to include (exclusive), default=36")
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    filter_by_age_days(
        args.input_data,
        args.input_config,
        args.output_data, 
        args.output_config,
        args.begin_day,
        args.end_day
    )


if __name__ == "__main__":
    main()

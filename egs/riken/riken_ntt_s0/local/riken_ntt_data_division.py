#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset Division Script for NTT Infant Speech Data

This script divides the NTT infant dataset into training, development, and test sets
based on unique session IDs. It reads the session IDs from a CSV file, randomly
assigns them to sets according to specified ratios, and outputs the division
to a YAML configuration file.

Usage:
   python script_name.py [--csv_info_file PATH] [--division_field FIELD]
                        [--train_ratio FLOAT] [--dev_ratio FLOAT] [--test_ratio FLOAT]
                        [--output_division PATH] [--seed INT]
"""

import argparse
import csv
import yaml
import random
import os

def parse_arguments():
   """Parse command line arguments for dataset division."""
   parser = argparse.ArgumentParser(description="Divide NTT infant dataset into train/dev/test sets")

   parser.add_argument('--csv_info_file', type=str, default='data/local/all.csv',
                       help='Path to the CSV file containing session information')
   parser.add_argument('--division_field', type=str, default='session_id',
                       help='Field in CSV to use for dividing the data (e.g., session_id)')
   parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Proportion of data for training set (default: 0.8)')
   parser.add_argument('--dev_ratio', type=float, default=0.1,
                       help='Proportion of data for development set (default: 0.1)')
   parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Proportion of data for test set (default: 0.1)')
   parser.add_argument('--output_division', type=str, default='conf/data/division_ntt.yaml',
                       help='Output path for the YAML division file')
   parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible data splits (default: None)')

   return parser.parse_args()

def main():
   """
   Main function to divide dataset and create YAML config file.

   Steps:
   1. Parse command line arguments
   2. Read unique IDs from the CSV file
   3. Randomly split IDs according to specified ratios
   4. Write the division to a YAML file
   """
   # Parse arguments
   args = parse_arguments()

   # Set random seed if provided
   if args.seed is not None:
       random.seed(args.seed)

   # Ensure the output directory exists
   os.makedirs(os.path.dirname(args.output_division), exist_ok=True)

   # Extract unique values of the division field from CSV
   unique_ids = set()
   with open(args.csv_info_file, 'r', encoding='utf-8') as f:
       reader = csv.DictReader(f)
       for row in reader:
           unique_ids.add(row[args.division_field])

   # Convert to list and sort for deterministic initial state before shuffling
   unique_ids = sorted(list(unique_ids))
   random.shuffle(unique_ids)

   # Calculate split sizes
   total_ids = len(unique_ids)

   # Normalize ratios to ensure they sum to 1
   total_ratio = args.train_ratio + args.dev_ratio + args.test_ratio
   if total_ratio != 1.0:
       train_ratio = args.train_ratio / total_ratio
       dev_ratio = args.dev_ratio / total_ratio
       test_ratio = args.test_ratio / total_ratio
   else:
       train_ratio = args.train_ratio
       dev_ratio = args.dev_ratio
       test_ratio = args.test_ratio

   train_size = int(total_ids * train_ratio)
   dev_size = int(total_ids * dev_ratio)
   test_size = total_ids - train_size - dev_size  # Ensure all IDs are assigned

   # Split the data
   train_ids = unique_ids[:train_size]
   dev_ids = unique_ids[train_size:train_size + dev_size]
   test_ids = unique_ids[train_size + dev_size:]

   # Create division dictionary with comment for YAML
   division = {
       '# Lists of wave ids for training, development, and test sets': None,
       'train': train_ids,
       'dev': dev_ids,
       'test': test_ids
   }

   # Write to YAML file with sort_keys for consistent output
   with open(args.output_division, 'w', encoding='utf-8') as f:
       yaml.dump(division, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

   print(f"Randomly split the train/dev/test into {args.output_division}")
   print(f"Train: {len(train_ids)} IDs ({train_ratio*100:.1f}%)")
   print(f"Dev: {len(dev_ids)} IDs ({dev_ratio*100:.1f}%)")
   print(f"Test: {len(test_ids)} IDs ({test_ratio*100:.1f}%)")
   if args.seed is not None:
       print(f"Using random seed: {args.seed}")

if __name__ == "__main__":
   main()

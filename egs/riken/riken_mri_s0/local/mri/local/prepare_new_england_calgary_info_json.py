#!/usr/bin/env python3
"""
Script to create info.json files for New England and Calgary datasets.
Each info.json contains subject information indexed by ID.
"""

import argparse
import os
import yaml
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create info.json files for MRI datasets'
    )
    parser.add_argument(
        '--data_dir_new_england',
        type=str,
        default='./data_mri/new_england',
        help='Path to New England data directory'
    )
    parser.add_argument(
        '--data_dir_calgary',
        type=str,
        default='./data_mri/calgary',
        help='Path to Calgary data directory'
    )
    parser.add_argument(
        '--modalities',
        type=str,
        nargs='+',
        default=['t1w', 'dwi', 't1w_dwi'],
        help='Which modalities to process (default: all)'
    )
    return parser.parse_args()


def load_yaml_file(filepath):
    """Load a YAML file and return its contents."""
    if not os.path.exists(filepath):
        return {}
    with open(filepath, 'r') as f:
        return yaml.safe_load(f) or {}


def load_modality_data(modality_dir, dataset_name):
    """
    Load all YAML files from a modality directory and organize by ID.
    Returns a dictionary with ID as key and all properties as nested dict.
    """
    if not os.path.exists(modality_dir):
        return {}

    # Find all YAML files in the directory
    yaml_files = {}
    for file in os.listdir(modality_dir):
        if file.endswith('.yaml'):
            # Extract field name from filename (e.g., id2age.yaml -> age)
            field_name = file.replace('id2', '').replace('.yaml', '')
            yaml_files[field_name] = file

    # Load all YAML files
    data_by_field = {}
    for field_name, filename in yaml_files.items():
        filepath = os.path.join(modality_dir, filename)
        data_by_field[field_name] = load_yaml_file(filepath)

    # Get all unique IDs
    all_ids = set()
    for field_data in data_by_field.values():
        all_ids.update(field_data.keys())

    # Reorganize data by ID
    dataset_info = {}
    for id_ in sorted(all_ids):
        dataset_info[id_] = {
            'id': id_,
            'dataset': dataset_name
        }
        for field_name, field_data in data_by_field.items():
            if id_ in field_data:
                dataset_info[id_][field_name] = field_data[id_]

    return dataset_info


def save_info_json(data, output_path):
    """Save data to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  Saved {len(data)} subjects to {output_path}")


def print_statistics(data, dataset_name, modality):
    """Print statistics about the dataset."""
    total = len(data)

    # Count genders
    genders = {'m': 0, 'f': 0, 'unknown': 0}
    for id_, info in data.items():
        gender = info.get('gender', 'unknown')
        if gender in ['m', 'f']:
            genders[gender] += 1
        else:
            genders['unknown'] += 1

    # Age statistics
    ages = [info['age'] for id_, info in data.items() if 'age' in info]

    print(f"  Total subjects: {total}")
    print(f"  Gender: M={genders['m']}, F={genders['f']}")
    if ages:
        print(f"  Age range: {min(ages):.1f} - {max(ages):.1f} months (mean: {sum(ages)/len(ages):.1f})")

    # Timepoints (if available)
    timepoints = {}
    for id_, info in data.items():
        if 'timepoint' in info:
            tp = info['timepoint']
            timepoints[tp] = timepoints.get(tp, 0) + 1

    if timepoints:
        print(f"  Timepoints: {timepoints}")


def process_modality(modality, new_england_dir, calgary_dir):
    """Process a single modality for both datasets."""
    print(f"\nProcessing modality: {modality}")
    print("=" * 60)

    # Load and save New England data
    ne_modality_dir = os.path.join(new_england_dir, modality)
    if os.path.exists(ne_modality_dir):
        print(f"New England {modality}:")
        ne_data = load_modality_data(ne_modality_dir, 'new_england')
        if ne_data:
            ne_json_path = os.path.join(ne_modality_dir, 'info.json')
            save_info_json(ne_data, ne_json_path)
            print_statistics(ne_data, 'new_england', modality)
        else:
            print(f"  No data found")

    # Load and save Calgary data
    cal_modality_dir = os.path.join(calgary_dir, modality)
    if os.path.exists(cal_modality_dir):
        print(f"\nCalgary {modality}:")
        cal_data = load_modality_data(cal_modality_dir, 'calgary')
        if cal_data:
            cal_json_path = os.path.join(cal_modality_dir, 'info.json')
            save_info_json(cal_data, cal_json_path)
            print_statistics(cal_data, 'calgary', modality)
        else:
            print(f"  No data found")


def main():
    args = parse_args()

    new_england_dir = args.data_dir_new_england
    calgary_dir = args.data_dir_calgary
    modalities = args.modalities

    print("=" * 60)
    print("Creating info.json files for MRI datasets")
    print("=" * 60)
    print(f"New England directory: {new_england_dir}")
    print(f"Calgary directory: {calgary_dir}")
    print(f"Modalities: {', '.join(modalities)}")

    # Process each modality
    for modality in modalities:
        process_modality(modality, new_england_dir, calgary_dir)

    # Summary
    print("\n" + "=" * 60)
    print("Summary - Created files:")
    print("=" * 60)
    for modality in modalities:
        ne_json = os.path.join(new_england_dir, modality, 'info.json')
        cal_json = os.path.join(calgary_dir, modality, 'info.json')

        if os.path.exists(ne_json):
            print(f"  - {ne_json}")
        if os.path.exists(cal_json):
            print(f"  - {cal_json}")

    print("\n✓ Done!")


if __name__ == '__main__':
    main()

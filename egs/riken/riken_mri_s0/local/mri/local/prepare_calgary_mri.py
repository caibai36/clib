#!/usr/bin/env python3
"""
Script to prepare Calgary Preschool dataset for processing.
Creates ID mappings and organizes data for T1w, DWI, and combined T1w+DWI.

Dataset: Calgary Preschool MRI Dataset
University of Calgary Conjoint Health Research Ethics Board (CHREB) REB13-0020

DTI b750 Dataset: 396 unprocessed b750 diffusion weighted MRI scans from 120 participants aged 2-8 years
Correspondence: clebel@ucalgary.ca
"""

import argparse
import os
import pandas as pd
import yaml
from pathlib import Path
import glob


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare Calgary Preschool dataset for MRI processing'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='/data02/share/bin-wu/data/human/brain/harvard_mri/raw/calgary/5kz2p/osfstorage/',
        help='Path to raw dataset directory'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='./data_mri/calgary',
        help='Path to output directory'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='exp/mri/data_prep',
        help='Directory to store logs of unmatched files'
    )
    return parser.parse_args()


def load_participants_data(dataset_path):
    """Load calgary_preschool_20200213.csv file."""
    csv_path = os.path.join(dataset_path, 'calgary_preschool_20200213.csv')
    df = pd.read_csv(csv_path)

    # Convert age from years to months
    df['age'] = df['age_years'] * 12.0

    # Round age to nearest integer
    df['age_rounded'] = df['age'].round(0).astype(int)

    return df


def sex_to_gender(sex_value):
    """Convert biological_sex_f0_m1 to gender string.
    Sex: 0=female, 1=male -> f, m
    """
    if pd.isna(sex_value):
        return None
    return 'm' if int(sex_value) == 1 else 'f'


def check_file_exists(file_path):
    """Check if a file exists."""
    return os.path.isfile(file_path)


def normalize_scan_id(scan_id):
    """Normalize scan_id by removing hyphens and converting to uppercase."""
    # Handle cases like PS0935-10-1 -> PS0935101
    return scan_id.replace('-', '').upper()

def find_t1w_file(dataset_path, preschool_id, scan_id):
    """
    Find T1w file for a given preschool_id and scan_id.
    """
    t1_base = os.path.join(dataset_path, 'T1_Dataset', str(preschool_id))

    if not os.path.exists(t1_base):
        return None

    # Special hard-coded case for PS0322-2 (file is in PS0322-10-2 directory)
    if str(preschool_id) == '10159' and scan_id == 'PS0322-2':
        special_path = os.path.join(t1_base, 'PS0322-10-2', 'PS0322102.nii.gz')
        if os.path.exists(special_path):
            return special_path

    # Normalize the CSV scan_id for comparison
    scan_id_normalized = scan_id.replace('-', '').replace('_', '').upper()

    # Look for scan_id directory
    for subdir in os.listdir(t1_base):
        subdir_path = os.path.join(t1_base, subdir)
        if not os.path.isdir(subdir_path):
            continue

        nii_files = glob.glob(os.path.join(subdir_path, '*.nii.gz'))
        if not nii_files:
            continue

        subdir_normalized = subdir.replace('-', '').replace('_', '').upper()

        if subdir_normalized == scan_id_normalized or scan_id_normalized in subdir_normalized:
            return nii_files[0]

        for nii_file in nii_files:
            basename = os.path.basename(nii_file).replace('.nii.gz', '')
            basename_normalized = basename.replace('-', '').replace('_', '').upper()
            if scan_id_normalized in basename_normalized:
                return nii_file

    return None


def find_dwi_file(dataset_path, preschool_id, scan_id):
    """
    Find DWI file for a given preschool_id and scan_id.
    """
    dwi_base = os.path.join(dataset_path, 'DTI_Dataset_b750', str(preschool_id))

    if not os.path.exists(dwi_base):
        return None

    # Special hard-coded case for PS0322-2 (file is named PS0322102_750.nii)
    if str(preschool_id) == '10159' and scan_id == 'PS0322-2':
        special_path = os.path.join(dwi_base, 'PS0322102_750.nii')
        if os.path.exists(special_path):
            return special_path

    # Normalize the CSV scan_id for comparison
    scan_id_normalized = scan_id.replace('-', '').replace('_', '').upper()

    nii_files = glob.glob(os.path.join(dwi_base, '*.nii'))

    for nii_file in nii_files:
        basename = os.path.basename(nii_file).replace('.nii', '').replace('_750', '')
        basename_normalized = basename.replace('-', '').replace('_', '').upper()
        if scan_id_normalized in basename_normalized:
            return nii_file

    return None


def get_dwi_shared_files(dataset_path):
    """
    Get shared DWI files (bval, bvec) that apply to all scans.
    """
    info_dir = os.path.join(dataset_path, 'DTI_Dataset_b750',
                            'Calgary_Preschool_DTI_b750_Dataset_Information')

    return {
        'bval': os.path.join(info_dir, 'bval_750.bval'),
        'bvec': os.path.join(info_dir, 'bvec_750.bvec'),
    }


def find_all_actual_files(dataset_path):
    """
    Find all actual T1w and DWI files in the dataset.
    Returns sets of file paths.
    """
    print("Scanning for all actual files in dataset...")

    # Find all T1w files
    t1w_files = set()
    t1_dataset = os.path.join(dataset_path, 'T1_Dataset')
    if os.path.exists(t1_dataset):
        for preschool_dir in os.listdir(t1_dataset):
            preschool_path = os.path.join(t1_dataset, preschool_dir)
            if os.path.isdir(preschool_path):
                # Find all .nii.gz files recursively
                for root, dirs, files in os.walk(preschool_path):
                    for file in files:
                        if file.endswith('.nii.gz'):
                            t1w_files.add(os.path.join(root, file))

    # Find all DWI files
    dwi_files = set()
    dwi_dataset = os.path.join(dataset_path, 'DTI_Dataset_b750')
    if os.path.exists(dwi_dataset):
        for preschool_dir in os.listdir(dwi_dataset):
            preschool_path = os.path.join(dwi_dataset, preschool_dir)
            if os.path.isdir(preschool_path) and preschool_dir != 'Calgary_Preschool_DTI_b750_Dataset_Information':
                # Find all .nii files
                for file in os.listdir(preschool_path):
                    if file.endswith('.nii'):
                        dwi_files.add(os.path.join(preschool_path, file))

    print(f"  Found {len(t1w_files)} T1w files")
    print(f"  Found {len(dwi_files)} DWI files")

    return t1w_files, dwi_files


def scan_available_data(dataset_path, df):
    """
    Scan the dataset to find which IDs have T1w and/or DWI data.
    Returns dictionaries with available data info and sets of matched files.
    """
    t1w_data = {}
    dwi_data = {}
    matched_t1w_files = set()
    matched_dwi_files = set()

    # Get shared DWI files
    dwi_shared_files = get_dwi_shared_files(dataset_path)

    for _, row in df.iterrows():
        preschool_id = row['preschool_id']
        scan_id = row['scan_id']

        # Create ID in format preschool_id_scan_id
        id_str = f"{preschool_id}_{scan_id}"

        # Check T1w
        t1w_path = find_t1w_file(dataset_path, preschool_id, scan_id)

        if t1w_path:
            t1w_data[id_str] = {
                'nii_path': t1w_path,
                'age': float(row['age']),
                'age_rounded': int(row['age_rounded']),
                'gender': sex_to_gender(row['biological_sex_f0_m1']),
                'preschool_id': preschool_id,
                'scan_id': scan_id
            }
            matched_t1w_files.add(t1w_path)

        # Check DWI
        dwi_path = find_dwi_file(dataset_path, preschool_id, scan_id)

        if dwi_path:
            dwi_data[id_str] = {
                'nii_path': dwi_path,
                'nii_ap': dwi_path,  # Only one phase encoding in this dataset
                'bval': dwi_shared_files['bval'],
                'bvec': dwi_shared_files['bvec'],
                'age': float(row['age']),
                'age_rounded': int(row['age_rounded']),
                'gender': sex_to_gender(row['biological_sex_f0_m1']),
                'preschool_id': preschool_id,
                'scan_id': scan_id
            }
            matched_dwi_files.add(dwi_path)

    return t1w_data, dwi_data, matched_t1w_files, matched_dwi_files


def write_unmatched_files(log_dir, all_t1w_files, all_dwi_files, matched_t1w_files, matched_dwi_files):
    """
    Write lists of unmatched files to log directory.
    """
    os.makedirs(log_dir, exist_ok=True)

    # Find unmatched files
    unmatched_t1w = all_t1w_files - matched_t1w_files
    unmatched_dwi = all_dwi_files - matched_dwi_files

    # Write unmatched T1w files
    t1w_log = os.path.join(log_dir, 'calgary_unmatched_t1w_files.txt')
    with open(t1w_log, 'w') as f:
        f.write(f"# Calgary Dataset - Unmatched T1w files\n")
        f.write(f"# (found in dataset but no CSV entry)\n")
        f.write(f"# Total: {len(unmatched_t1w)} files\n")
        f.write(f"# Generated: {pd.Timestamp.now()}\n\n")
        for file_path in sorted(unmatched_t1w):
            f.write(f"{file_path}\n")

    # Write unmatched DWI files
    dwi_log = os.path.join(log_dir, 'calgary_unmatched_dwi_files.txt')
    with open(dwi_log, 'w') as f:
        f.write(f"# Calgary Dataset - Unmatched DWI files\n")
        f.write(f"# (found in dataset but no CSV entry)\n")
        f.write(f"# Total: {len(unmatched_dwi)} files\n")
        f.write(f"# Generated: {pd.Timestamp.now()}\n\n")
        for file_path in sorted(unmatched_dwi):
            f.write(f"{file_path}\n")

    # Write summary
    summary_log = os.path.join(log_dir, 'calgary_unmatched_summary.txt')
    with open(summary_log, 'w') as f:
        f.write("Calgary Dataset - Unmatched Files Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")

        f.write(f"T1w Files:\n")
        f.write(f"  Total found in dataset: {len(all_t1w_files)}\n")
        f.write(f"  Matched with CSV: {len(matched_t1w_files)}\n")
        f.write(f"  Unmatched: {len(unmatched_t1w)}\n")
        if len(all_t1w_files) > 0:
            f.write(f"  Match rate: {len(matched_t1w_files)/len(all_t1w_files)*100:.1f}%\n\n")

        f.write(f"DWI Files:\n")
        f.write(f"  Total found in dataset: {len(all_dwi_files)}\n")
        f.write(f"  Matched with CSV: {len(matched_dwi_files)}\n")
        f.write(f"  Unmatched: {len(unmatched_dwi)}\n")
        if len(all_dwi_files) > 0:
            f.write(f"  Match rate: {len(matched_dwi_files)/len(all_dwi_files)*100:.1f}%\n\n")

        if unmatched_t1w:
            f.write("\nUnmatched T1w files:\n")
            for file_path in sorted(unmatched_t1w):
                f.write(f"  {file_path}\n")

        if unmatched_dwi:
            f.write("\nUnmatched DWI files:\n")
            for file_path in sorted(unmatched_dwi):
                f.write(f"  {file_path}\n")

    return len(unmatched_t1w), len(unmatched_dwi), t1w_log, dwi_log, summary_log


def create_yaml_mappings(data_dict, output_dir, modality):
    """Create YAML mapping files for a given modality."""
    os.makedirs(output_dir, exist_ok=True)

    # id2nii.yaml (primary image file)
    id2nii = {id_: info['nii_path'] for id_, info in data_dict.items()}
    with open(os.path.join(output_dir, 'id2nii.yaml'), 'w') as f:
        yaml.dump(id2nii, f, default_flow_style=False, sort_keys=True)

    # For DWI, create additional mappings for all files
    if modality == 'DWI':
        # id2nii_ap.yaml (same as nii_path for Calgary)
        id2nii_ap = {id_: info['nii_ap'] for id_, info in data_dict.items()}
        with open(os.path.join(output_dir, 'id2nii_ap.yaml'), 'w') as f:
            yaml.dump(id2nii_ap, f, default_flow_style=False, sort_keys=True)

        # id2bval.yaml (shared file for all)
        id2bval = {id_: info['bval'] for id_, info in data_dict.items()}
        with open(os.path.join(output_dir, 'id2bval.yaml'), 'w') as f:
            yaml.dump(id2bval, f, default_flow_style=False, sort_keys=True)

        # id2bvec.yaml (shared file for all)
        id2bvec = {id_: info['bvec'] for id_, info in data_dict.items()}
        with open(os.path.join(output_dir, 'id2bvec.yaml'), 'w') as f:
            yaml.dump(id2bvec, f, default_flow_style=False, sort_keys=True)

    # id2age.yaml
    id2age = {id_: info['age'] for id_, info in data_dict.items()}
    with open(os.path.join(output_dir, 'id2age.yaml'), 'w') as f:
        yaml.dump(id2age, f, default_flow_style=False, sort_keys=True)

    # id2age_rounded.yaml
    id2age_rounded = {id_: info['age_rounded'] for id_, info in data_dict.items()}
    with open(os.path.join(output_dir, 'id2age_rounded.yaml'), 'w') as f:
        yaml.dump(id2age_rounded, f, default_flow_style=False, sort_keys=True)

    # id2gender.yaml (m/f instead of 0/1)
    id2gender = {id_: info['gender'] for id_, info in data_dict.items()}
    with open(os.path.join(output_dir, 'id2gender.yaml'), 'w') as f:
        yaml.dump(id2gender, f, default_flow_style=False, sort_keys=True)

    print(f"Created YAML mappings for {modality} in {output_dir}")
    print(f"  Total IDs: {len(data_dict)}")


def main():
    args = parse_args()

    dataset_path = args.dataset_path
    output_path = args.output_path
    log_dir = args.log_dir

    print("=" * 60)
    print("Calgary Preschool Dataset Preparation")
    print("=" * 60)
    print(f"Dataset path: {dataset_path}")
    print(f"Output path: {output_path}")
    print(f"Log directory: {log_dir}")
    print()

    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Load participants data
    print("Loading participants data...")
    df = load_participants_data(dataset_path)

    # Save info.csv
    info_csv_path = os.path.join(output_path, 'info.csv')
    # Add id column in format preschool_id_scan_id
    df['id'] = df.apply(lambda row: f"{row['preschool_id']}_{row['scan_id']}", axis=1)
    # Add gender column (m/f)
    df['gender'] = df['biological_sex_f0_m1'].apply(sex_to_gender)
    df.to_csv(info_csv_path, index=False)
    print(f"Saved info.csv to {info_csv_path}")
    print(f"Total entries in CSV: {len(df)}")
    print(f"Unique participants: {df['preschool_id'].nunique()}")
    print()

    # Find all actual files in dataset
    all_t1w_files, all_dwi_files = find_all_actual_files(dataset_path)
    print()

    # Scan available data
    print("Matching CSV entries with data files...")
    print("(This may take a few minutes due to the directory structure...)")
    t1w_data, dwi_data, matched_t1w_files, matched_dwi_files = scan_available_data(dataset_path, df)

    print(f"Found {len(t1w_data)} IDs with T1w data")
    print(f"Found {len(dwi_data)} IDs with DWI data")
    print()

    # Write unmatched files log
    print("Logging unmatched files...")
    num_unmatched_t1w, num_unmatched_dwi, t1w_log, dwi_log, summary_log = write_unmatched_files(
        log_dir, all_t1w_files, all_dwi_files, matched_t1w_files, matched_dwi_files
    )

    print(f"Unmatched files logged to {log_dir}/")
    print(f"  - calgary_unmatched_t1w_files.txt: {num_unmatched_t1w} files")
    print(f"  - calgary_unmatched_dwi_files.txt: {num_unmatched_dwi} files")
    print(f"  - calgary_unmatched_summary.txt")
    print()

    # Find IDs with both T1w and DWI
    t1w_dwi_ids = set(t1w_data.keys()) & set(dwi_data.keys())
    print(f"Found {len(t1w_dwi_ids)} IDs with both T1w and DWI data")
    print()

    # Create T1w mappings
    print("Creating T1w mappings...")
    t1w_dir = os.path.join(output_path, 't1w')
    create_yaml_mappings(t1w_data, t1w_dir, 'T1w')
    print()

    # Create DWI mappings
    print("Creating DWI mappings...")
    dwi_dir = os.path.join(output_path, 'dwi')
    create_yaml_mappings(dwi_data, dwi_dir, 'DWI')
    print()

    # Create T1w+DWI mappings (only IDs with both)
    print("Creating T1w+DWI mappings...")
    t1w_dwi_data_t1w = {id_: t1w_data[id_] for id_ in t1w_dwi_ids}
    t1w_dwi_data_dwi = {id_: dwi_data[id_] for id_ in t1w_dwi_ids}

    t1w_dwi_dir = os.path.join(output_path, 't1w_dwi')
    os.makedirs(t1w_dwi_dir, exist_ok=True)

    # For t1w_dwi, create combined mappings
    # T1w related
    id2nii_t1w = {id_: info['nii_path'] for id_, info in t1w_dwi_data_t1w.items()}
    with open(os.path.join(t1w_dwi_dir, 'id2nii_t1w.yaml'), 'w') as f:
        yaml.dump(id2nii_t1w, f, default_flow_style=False, sort_keys=True)

    # DWI related
    id2nii_dwi_ap = {id_: info['nii_ap'] for id_, info in t1w_dwi_data_dwi.items()}
    with open(os.path.join(t1w_dwi_dir, 'id2nii_dwi_ap.yaml'), 'w') as f:
        yaml.dump(id2nii_dwi_ap, f, default_flow_style=False, sort_keys=True)

    id2bval = {id_: info['bval'] for id_, info in t1w_dwi_data_dwi.items()}
    with open(os.path.join(t1w_dwi_dir, 'id2bval.yaml'), 'w') as f:
        yaml.dump(id2bval, f, default_flow_style=False, sort_keys=True)

    id2bvec = {id_: info['bvec'] for id_, info in t1w_dwi_data_dwi.items()}
    with open(os.path.join(t1w_dwi_dir, 'id2bvec.yaml'), 'w') as f:
        yaml.dump(id2bvec, f, default_flow_style=False, sort_keys=True)

    # Age and gender (use T1w data as reference)
    id2age = {id_: info['age'] for id_, info in t1w_dwi_data_t1w.items()}
    with open(os.path.join(t1w_dwi_dir, 'id2age.yaml'), 'w') as f:
        yaml.dump(id2age, f, default_flow_style=False, sort_keys=True)

    id2age_rounded = {id_: info['age_rounded'] for id_, info in t1w_dwi_data_t1w.items()}
    with open(os.path.join(t1w_dwi_dir, 'id2age_rounded.yaml'), 'w') as f:
        yaml.dump(id2age_rounded, f, default_flow_style=False, sort_keys=True)

    id2gender = {id_: info['gender'] for id_, info in t1w_dwi_data_t1w.items()}
    with open(os.path.join(t1w_dwi_dir, 'id2gender.yaml'), 'w') as f:
        yaml.dump(id2gender, f, default_flow_style=False, sort_keys=True)

    print(f"Created YAML mappings for T1w+DWI in {t1w_dwi_dir}")
    print(f"  Total IDs: {len(t1w_dwi_ids)}")
    print()

    # Print gender distribution
    genders_t1w = [info['gender'] for info in t1w_data.values() if info['gender'] is not None]
    male_count = genders_t1w.count('m')
    female_count = genders_t1w.count('f')

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Output directory: {output_path}")
    print(f"  - info.csv: {len(df)} entries")
    print(f"  - t1w/: {len(t1w_data)} IDs")
    print(f"  - dwi/: {len(dwi_data)} IDs")
    print(f"  - t1w_dwi/: {len(t1w_dwi_ids)} IDs")
    print()
    print(f"Log directory: {log_dir}")
    print(f"  - Unmatched T1w files: {num_unmatched_t1w}")
    print(f"  - Unmatched DWI files: {num_unmatched_dwi}")
    print()
    print("Gender distribution (T1w data):")
    print(f"  Male (m): {male_count}")
    print(f"  Female (f): {female_count}")
    print()
    print("Age statistics (T1w data):")
    ages = [info['age'] for info in t1w_data.values()]
    if ages:
        print(f"  Range: {min(ages):.2f} - {max(ages):.2f} months")
        print(f"  Range (years): {min(ages)/12:.2f} - {max(ages)/12:.2f} years")
        print(f"  Rounded range: {min([info['age_rounded'] for info in t1w_data.values()])} - "
              f"{max([info['age_rounded'] for info in t1w_data.values()])} months")
    print()
    print("Files created in t1w/ directory:")
    print("  - id2nii.yaml")
    print("  - id2age.yaml")
    print("  - id2age_rounded.yaml")
    print("  - id2gender.yaml (m/f format)")
    print()
    print("Files created in dwi/ directory:")
    print("  - id2nii.yaml")
    print("  - id2nii_ap.yaml")
    print("  - id2bval.yaml (shared across all scans)")
    print("  - id2bvec.yaml (shared across all scans)")
    print("  - id2age.yaml")
    print("  - id2age_rounded.yaml")
    print("  - id2gender.yaml (m/f format)")
    print()
    print("Files created in t1w_dwi/ directory:")
    print("  - id2nii_t1w.yaml")
    print("  - id2nii_dwi_ap.yaml")
    print("  - id2bval.yaml")
    print("  - id2bvec.yaml")
    print("  - id2age.yaml")
    print("  - id2age_rounded.yaml")
    print("  - id2gender.yaml (m/f format)")
    print()

    # Show some example IDs
    if t1w_data:
        print("Example IDs (first 5):")
        for i, id_ in enumerate(sorted(t1w_data.keys())[:5]):
            print(f"  {id_}")
    print()

    if num_unmatched_t1w > 0 or num_unmatched_dwi > 0:
        print("Warning: Some files in the dataset could not be matched to CSV entries.")
        print(f"   Check {log_dir}/ for details.")
        print()

    print("Done!")
    print()


if __name__ == '__main__':
    main()

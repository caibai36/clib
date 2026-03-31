#!/usr/bin/env python3
"""
Script to prepare New England dataset (ds006169-1.0.3) for processing.
Creates ID mappings and organizes data for T1w, DWI, and combined T1w+DWI.

Dataset: Longitudinal trajectories of brain development from infancy to school age
Reference: Turesky TK, Escalante ES, Loh M, Gaab N. (2024)
           https://doi.org/10.1101/2024.06.29.601366

Session mapping:
  ses-01 = infant (INF)
  ses-02 = toddler (TOD)
  ses-03 = pre-reading (PRE)
  ses-04 = beginning reading (BEG)
  ses-05 = emergent reading (REA)
"""

import argparse
import os
import pandas as pd
import yaml
from pathlib import Path
import io
import glob


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare New England dataset for MRI processing'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default='/data02/share/bin-wu/data/human/brain/harvard_mri/raw/new_england/ds006169-1.0.3/',
        help='Path to raw dataset directory'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='./data_mri/new_england',
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
    """Load and clean participants.tsv file."""
    tsv_path = os.path.join(dataset_path, 'participants.tsv')

    # Read the TSV file and clean it (remove quotes)
    with open(tsv_path, 'r') as f:
        lines = [line.strip().strip('"') for line in f]

    # Parse as CSV
    df = pd.read_csv(io.StringIO('\n'.join(lines)), sep=',')

    # Add rounded age column
    df['age_rounded'] = df['age'].round(0).astype(int)

    return df


def sex_to_gender(sex_value):
    """Convert Sex value to gender string.
    Sex: 0=male, 1=female -> m, f
    """
    if pd.isna(sex_value):
        return None
    return 'f' if int(sex_value) == 1 else 'm'


def check_file_exists(file_path):
    """Check if a file exists."""
    return os.path.isfile(file_path)


def get_dwi_files(dataset_path, participant_id, session):
    """
    Get all DWI-related files for a participant/session.
    Returns dict with all file paths if complete, None otherwise.
    """
    dwi_dir = os.path.join(
        dataset_path,
        participant_id,
        f"ses-{session:02d}",
        "dwi"
    )

    base_name = f"{participant_id}_ses-{session:02d}_dwi_AP"

    files = {
        'nii_ap': os.path.join(dwi_dir, f"{base_name}.nii.gz"),
        'bval': os.path.join(dwi_dir, f"{base_name}.bval"),
        'bvec': os.path.join(dwi_dir, f"{base_name}.bvec"),
        'json': os.path.join(dwi_dir, f"{base_name}.json"),
        'nii_pa': os.path.join(dwi_dir, f"{participant_id}_ses-{session:02d}_dwi_PA.nii.gz")
    }

    # Check if all essential files exist
    # PA file is optional for some processing pipelines
    required_files = ['nii_ap', 'bval', 'bvec']
    if all(check_file_exists(files[key]) for key in required_files):
        return files
    return None


def find_all_actual_files(dataset_path):
    """
    Find all actual T1w and DWI files in the dataset.
    Returns sets of file paths.
    """
    print("Scanning for all actual files in dataset...")

    # Find all T1w files
    t1w_files = set()
    for root, dirs, files in os.walk(dataset_path):
        if 'anat' in root:
            for file in files:
                if file.endswith('_T1w.nii.gz'):
                    t1w_files.add(os.path.join(root, file))

    # Find all DWI files (AP direction primary files)
    dwi_files = set()
    for root, dirs, files in os.walk(dataset_path):
        if 'dwi' in root:
            for file in files:
                if file.endswith('_dwi_AP.nii.gz'):
                    dwi_files.add(os.path.join(root, file))

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

    for _, row in df.iterrows():
        participant_id = row['participant_id']
        session = row['session']

        # Create ID in format sub-XX_ses-YY
        id_str = f"{participant_id}_ses-{session:02d}"

        # Check T1w
        t1w_path = os.path.join(
            dataset_path,
            participant_id,
            f"ses-{session:02d}",
            "anat",
            f"{participant_id}_ses-{session:02d}_T1w.nii.gz"
        )

        if check_file_exists(t1w_path):
            t1w_data[id_str] = {
                'nii_path': t1w_path,
                'age': float(row['age']),
                'age_rounded': int(row['age_rounded']),
                'timepoint': row['timepoint'],
                'gender': sex_to_gender(row['Sex'])
            }
            matched_t1w_files.add(t1w_path)

        # Check DWI - get all related files
        dwi_files = get_dwi_files(dataset_path, participant_id, session)
        if dwi_files:
            dwi_data[id_str] = {
                'nii_path': dwi_files['nii_ap'],  # Primary DWI file (AP)
                'nii_ap': dwi_files['nii_ap'],
                'nii_pa': dwi_files['nii_pa'],
                'bval': dwi_files['bval'],
                'bvec': dwi_files['bvec'],
                'json': dwi_files['json'],
                'age': float(row['age']),
                'age_rounded': int(row['age_rounded']),
                'timepoint': row['timepoint'],
                'gender': sex_to_gender(row['Sex'])
            }
            matched_dwi_files.add(dwi_files['nii_ap'])

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
    t1w_log = os.path.join(log_dir, 'new_england_unmatched_t1w_files.txt')
    with open(t1w_log, 'w') as f:
        f.write(f"# New England Dataset - Unmatched T1w files\n")
        f.write(f"# (found in dataset but no CSV entry)\n")
        f.write(f"# Total: {len(unmatched_t1w)} files\n")
        f.write(f"# Generated: {pd.Timestamp.now()}\n\n")
        for file_path in sorted(unmatched_t1w):
            f.write(f"{file_path}\n")

    # Write unmatched DWI files
    dwi_log = os.path.join(log_dir, 'new_england_unmatched_dwi_files.txt')
    with open(dwi_log, 'w') as f:
        f.write(f"# New England Dataset - Unmatched DWI files\n")
        f.write(f"# (found in dataset but no CSV entry)\n")
        f.write(f"# Total: {len(unmatched_dwi)} files\n")
        f.write(f"# Generated: {pd.Timestamp.now()}\n\n")
        for file_path in sorted(unmatched_dwi):
            f.write(f"{file_path}\n")

    # Write summary
    summary_log = os.path.join(log_dir, 'new_england_unmatched_summary.txt')
    with open(summary_log, 'w') as f:
        f.write("New England Dataset - Unmatched Files Summary\n")
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
            f.write("\nSample unmatched T1w files (first 10):\n")
            for file_path in sorted(unmatched_t1w)[:10]:
                f.write(f"  {file_path}\n")

        if unmatched_dwi:
            f.write("\nSample unmatched DWI files (first 10):\n")
            for file_path in sorted(unmatched_dwi)[:10]:
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
        # id2nii_ap.yaml
        id2nii_ap = {id_: info['nii_ap'] for id_, info in data_dict.items()}
        with open(os.path.join(output_dir, 'id2nii_ap.yaml'), 'w') as f:
            yaml.dump(id2nii_ap, f, default_flow_style=False, sort_keys=True)

        # id2nii_pa.yaml
        id2nii_pa = {id_: info['nii_pa'] for id_, info in data_dict.items()}
        with open(os.path.join(output_dir, 'id2nii_pa.yaml'), 'w') as f:
            yaml.dump(id2nii_pa, f, default_flow_style=False, sort_keys=True)

        # id2bval.yaml
        id2bval = {id_: info['bval'] for id_, info in data_dict.items()}
        with open(os.path.join(output_dir, 'id2bval.yaml'), 'w') as f:
            yaml.dump(id2bval, f, default_flow_style=False, sort_keys=True)

        # id2bvec.yaml
        id2bvec = {id_: info['bvec'] for id_, info in data_dict.items()}
        with open(os.path.join(output_dir, 'id2bvec.yaml'), 'w') as f:
            yaml.dump(id2bvec, f, default_flow_style=False, sort_keys=True)

        # id2json.yaml
        id2json = {id_: info['json'] for id_, info in data_dict.items()}
        with open(os.path.join(output_dir, 'id2json.yaml'), 'w') as f:
            yaml.dump(id2json, f, default_flow_style=False, sort_keys=True)

    # id2age.yaml
    id2age = {id_: info['age'] for id_, info in data_dict.items()}
    with open(os.path.join(output_dir, 'id2age.yaml'), 'w') as f:
        yaml.dump(id2age, f, default_flow_style=False, sort_keys=True)

    # id2age_rounded.yaml
    id2age_rounded = {id_: info['age_rounded'] for id_, info in data_dict.items()}
    with open(os.path.join(output_dir, 'id2age_rounded.yaml'), 'w') as f:
        yaml.dump(id2age_rounded, f, default_flow_style=False, sort_keys=True)

    # id2timepoint.yaml
    id2timepoint = {id_: info['timepoint'] for id_, info in data_dict.items()}
    with open(os.path.join(output_dir, 'id2timepoint.yaml'), 'w') as f:
        yaml.dump(id2timepoint, f, default_flow_style=False, sort_keys=True)

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
    print("New England Dataset Preparation")
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
    # Add id column in format sub-XX_ses-YY
    df['id'] = df.apply(lambda row: f"{row['participant_id']}_ses-{row['session']:02d}", axis=1)
    # Add gender column (m/f)
    df['gender'] = df['Sex'].apply(sex_to_gender)
    df.to_csv(info_csv_path, index=False)
    print(f"Saved info.csv to {info_csv_path}")
    print(f"Total entries in CSV: {len(df)}")
    print()

    # Find all actual files in dataset
    all_t1w_files, all_dwi_files = find_all_actual_files(dataset_path)
    print()

    # Scan available data
    print("Matching CSV entries with data files...")
    t1w_data, dwi_data, matched_t1w_files, matched_dwi_files = scan_available_data(dataset_path, df)

    print(f"Found {len(t1w_data)} IDs with T1w data")
    print(f"Found {len(dwi_data)} IDs with DWI data (complete sets)")
    print()

    # Write unmatched files log
    print("Logging unmatched files...")
    num_unmatched_t1w, num_unmatched_dwi, t1w_log, dwi_log, summary_log = write_unmatched_files(
        log_dir, all_t1w_files, all_dwi_files, matched_t1w_files, matched_dwi_files
    )

    print(f"Unmatched files logged to {log_dir}/")
    print(f"  - new_england_unmatched_t1w_files.txt: {num_unmatched_t1w} files")
    print(f"  - new_england_unmatched_dwi_files.txt: {num_unmatched_dwi} files")
    print(f"  - new_england_unmatched_summary.txt")
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

    id2nii_dwi_pa = {id_: info['nii_pa'] for id_, info in t1w_dwi_data_dwi.items()}
    with open(os.path.join(t1w_dwi_dir, 'id2nii_dwi_pa.yaml'), 'w') as f:
        yaml.dump(id2nii_dwi_pa, f, default_flow_style=False, sort_keys=True)

    id2bval = {id_: info['bval'] for id_, info in t1w_dwi_data_dwi.items()}
    with open(os.path.join(t1w_dwi_dir, 'id2bval.yaml'), 'w') as f:
        yaml.dump(id2bval, f, default_flow_style=False, sort_keys=True)

    id2bvec = {id_: info['bvec'] for id_, info in t1w_dwi_data_dwi.items()}
    with open(os.path.join(t1w_dwi_dir, 'id2bvec.yaml'), 'w') as f:
        yaml.dump(id2bvec, f, default_flow_style=False, sort_keys=True)

    id2json = {id_: info['json'] for id_, info in t1w_dwi_data_dwi.items()}
    with open(os.path.join(t1w_dwi_dir, 'id2json.yaml'), 'w') as f:
        yaml.dump(id2json, f, default_flow_style=False, sort_keys=True)

    # Age, timepoint, gender (use T1w data as reference)
    id2age = {id_: info['age'] for id_, info in t1w_dwi_data_t1w.items()}
    with open(os.path.join(t1w_dwi_dir, 'id2age.yaml'), 'w') as f:
        yaml.dump(id2age, f, default_flow_style=False, sort_keys=True)

    id2age_rounded = {id_: info['age_rounded'] for id_, info in t1w_dwi_data_t1w.items()}
    with open(os.path.join(t1w_dwi_dir, 'id2age_rounded.yaml'), 'w') as f:
        yaml.dump(id2age_rounded, f, default_flow_style=False, sort_keys=True)

    id2timepoint = {id_: info['timepoint'] for id_, info in t1w_dwi_data_t1w.items()}
    with open(os.path.join(t1w_dwi_dir, 'id2timepoint.yaml'), 'w') as f:
        yaml.dump(id2timepoint, f, default_flow_style=False, sort_keys=True)

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

    # Print timepoint distribution
    timepoints = [info['timepoint'] for info in t1w_data.values()]
    timepoint_counts = pd.Series(timepoints).value_counts().sort_index()

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
    print("Timepoint distribution (T1w data):")
    timepoint_map = {
        'INF': 'ses-01 (infant)',
        'TOD': 'ses-02 (toddler)',
        'PRE': 'ses-03 (pre-reading)',
        'BEG': 'ses-04 (beginning reading)',
        'REA': 'ses-05 (emergent reading)'
    }
    for tp, count in timepoint_counts.items():
        tp_label = timepoint_map.get(tp, tp)
        print(f"  {tp_label}: {count}")
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
    print("  - id2timepoint.yaml")
    print("  - id2gender.yaml (m/f format)")
    print()
    print("Files created in dwi/ directory:")
    print("  - id2nii.yaml (AP)")
    print("  - id2nii_ap.yaml")
    print("  - id2nii_pa.yaml")
    print("  - id2bval.yaml")
    print("  - id2bvec.yaml")
    print("  - id2json.yaml")
    print("  - id2age.yaml")
    print("  - id2age_rounded.yaml")
    print("  - id2timepoint.yaml")
    print("  - id2gender.yaml (m/f format)")
    print()
    print("Files created in t1w_dwi/ directory:")
    print("  - id2nii_t1w.yaml")
    print("  - id2nii_dwi_ap.yaml")
    print("  - id2nii_dwi_pa.yaml")
    print("  - id2bval.yaml")
    print("  - id2bvec.yaml")
    print("  - id2json.yaml")
    print("  - id2age.yaml")
    print("  - id2age_rounded.yaml")
    print("  - id2timepoint.yaml")
    print("  - id2gender.yaml (m/f format)")
    print()

    if num_unmatched_t1w > 0 or num_unmatched_dwi > 0:
        print("Warning: Some files in the dataset could not be matched to CSV entries.")
        print(f"   Check {log_dir}/ for details.")
        print()

    print("Done!")
    print()


if __name__ == '__main__':
    main()

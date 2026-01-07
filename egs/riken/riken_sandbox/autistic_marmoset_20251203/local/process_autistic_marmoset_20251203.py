#!/usr/bin/env python3
"""
Script to process autistic marmoset dataset from Nakanishi lab.
"""

import os
import re
import argparse
import pandas as pd
from pathlib import Path
import warnings
import shutil

# Phenotype mapping
PHENOTYPE_MAP = {
    'Momo': 'WT', 'momo': 'WT',
    'Tsubaki': 'WT', 'tsubaki': 'WT',
    'Lave': 'WT', 'Lav': 'WT',
    'Matsuba': 'Mut', 'matsuba': 'Mut',
    'Tokiwa': 'Mut', 'tokiwa': 'Mut',
    'Warabi': 'Mut'
}

def normalize_name(name):
    """Normalize marmoset name."""
    name_map = {
        'momo': 'Momo', 'tsubaki': 'Tsubaki', 'tokiwa': 'Tokiwa',
        'matsuba': 'Matsuba', 'lav': 'Lave', 'lave': 'Lave'
    }
    return name_map.get(name.lower(), name.capitalize())

def parse_audio_filename(filename):
    """
    Parse audio filename.

    Format: YYMMDD_ID_L_Name_Age.wav
    Age formats: P## (e.g., P7, P31) or ##W (e.g., 5W, 9W)
    ID formats: #### (e.g., 0705), ####X (e.g., 0685w), ####-# (e.g., 0650-1)
    """
    basename = os.path.splitext(filename)[0].strip()

    # Try patterns with different age formats
    patterns = [
        # With L_, age format P## (letter first)
        (r'^(\d{6})_([\w-]+)_L_([a-zA-Z]+)_(P\d+)$', 'P'),
        # With L_, age format ##W or ##w (digits first)
        (r'^(\d{6})_([\w-]+)_L_([a-zA-Z]+)_(\d+[Ww])$', 'W'),
        # Without L_, age format P##
        (r'^(\d{6})_([\w-]+)_([a-zA-Z]+)_(P\d+)$', 'P'),
        # Without L_, age format ##W or ##w
        (r'^(\d{6})_([\w-]+)_([a-zA-Z]+)_(\d+[Ww])$', 'W'),
    ]

    match = None
    age_format = None
    for pattern, fmt in patterns:
        match = re.match(pattern, basename, re.IGNORECASE)
        if match:
            age_format = fmt
            break

    if not match:
        return None

    date_str, file_id, name, age_str = match.groups()

    name = normalize_name(name)

    # Parse age based on format
    if age_format == 'P':
        age_type = 'P'
        age_value = int(age_str[1:])  # Remove 'P' prefix
        age_days = age_value
        age_weeks = age_value // 7
    else:  # age_format == 'W'
        age_type = 'W'
        age_value = int(age_str[:-1])  # Remove 'W' suffix
        age_weeks = age_value
        age_days = age_value * 7

    return {
        'date': date_str, 'id': file_id, 'name': name,
        'age_type': age_type, 'age_value': age_value,
        'age_days': age_days, 'age_weeks': age_weeks,
        'phenotype': PHENOTYPE_MAP.get(name, 'Unknown')
    }

def parse_annotation_filename(filename):
    """Parse annotation filename."""
    basename = os.path.splitext(filename)[0]

    # Experiment_2 format: Phenotype_Name_Age_ID_YYMMDD_Stats_C
    match = re.match(r'^(WT|Mut)_(\w+)_(\d+)w_(\w+)_(\d{6})_Stats_C', basename, re.IGNORECASE)
    if match:
        phenotype, name, age_weeks, file_id, date_str = match.groups()
        return {
            'phenotype': phenotype,
            'name': normalize_name(name),
            'age_type': 'W',
            'age_value': int(age_weeks),
            'age_weeks': int(age_weeks),
            'age_days': int(age_weeks) * 7,
            'id': file_id,
            'date': date_str
        }

    # Experiment_1 format - remove prefixes first
    clean_name = re.sub(r'^(c_processed_|F_C_LC_)', '', basename)
    clean_name = re.sub(r'^(Dcp|Dpc|D)(?=\d{6})', '', clean_name)

    # Try pattern WITH age (most common)
    match = re.match(r'^(\d{6})_([\w-]+)_(?:L_)?(\w+)_(P\d+)_Stats', clean_name, re.IGNORECASE)
    if match:
        date_str, file_id, name, age_str = match.groups()
        age_type = 'P'
        age_value = int(age_str[1:])
        age_days = age_value
        age_weeks = age_value // 7

        name_normalized = normalize_name(name)
        return {
            'phenotype': PHENOTYPE_MAP.get(name_normalized, 'Unknown'),
            'name': name_normalized,
            'age_type': age_type,
            'age_value': age_value,
            'age_weeks': age_weeks,
            'age_days': age_days,
            'id': file_id,
            'date': date_str
        }

    # Try pattern WITHOUT age (rare case)
    match = re.match(r'^(\d{6})_([\w-]+)_(?:L_)?(\w+)_Stats', clean_name, re.IGNORECASE)
    if match:
        date_str, file_id, name = match.groups()
        name_normalized = normalize_name(name)
        return {
            'phenotype': PHENOTYPE_MAP.get(name_normalized, 'Unknown'),
            'name': name_normalized,
            'age_type': None,
            'age_value': None,
            'age_weeks': None,
            'age_days': None,
            'id': file_id,
            'date': date_str
        }

    return None

def find_matching_annotation(audio_info, annotation_files):
    """Find matching annotation file."""
    audio_name = audio_info['name']
    audio_age_value = audio_info['age_value']
    audio_age_type = audio_info['age_type']
    audio_id = audio_info['id']
    audio_date = audio_info['date']

    best_match = None
    best_score = 0

    for ann_file in annotation_files:
        ann_info = parse_annotation_filename(ann_file.name)
        if not ann_info:
            continue

        score = 0

        # Name must match (REQUIRED)
        if ann_info['name'] != audio_name:
            continue
        score += 100

        # If annotation has age info, it must match
        if ann_info['age_value'] is not None:
            if ann_info['age_value'] != audio_age_value:
                continue
            score += 100

            if ann_info['age_type'] != audio_age_type:
                continue
            score += 100
        else:
            # No age in annotation - still valid but lower score
            score += 50

        # ID match (BONUS) - strip letter suffixes for comparison
        ann_id_base = ann_info['id'].rstrip('wn')
        audio_id_base = audio_id.rstrip('wn')
        if ann_info['id'] == audio_id or ann_id_base == audio_id_base:
            score += 50

        # Date match (BONUS)
        if ann_info['date'] == audio_date:
            score += 50

        if score > best_score:
            best_score = score
            best_match = (ann_file, ann_info)

    return best_match

def convert_csv_to_audacity(csv_file, output_file, label_map=None):
    """Convert CSV to Audacity format."""
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        warnings.warn(f"Error reading {csv_file}: {e}")
        return False

    required_cols = ['Begin Time (s)', 'End Time (s)', 'Label']
    if not all(col in df.columns for col in required_cols):
        warnings.warn(f"Missing required columns in {csv_file}")
        return False

    if 'Accepted' in df.columns:
        df = df[df['Accepted'] == 1]

    if label_map:
        df['Label'] = df['Label'].map(lambda x: label_map.get(x, x))

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf8') as f:
        for _, row in df.iterrows():
            f.write(f"{row['Begin Time (s)']:.6f}\t{row['End Time (s)']:.6f}\t{row['Label']}\n")

    return True

def convert_xlsx_to_audacity(xlsx_file, output_file, label_map=None):
    """Convert Excel to Audacity format."""
    try:
        df = pd.read_excel(xlsx_file)
    except Exception as e:
        warnings.warn(f"Error reading {xlsx_file}: {e}")
        return False

    required_cols = ['Begin Time (s)', 'End Time (s)', 'Label']
    if not all(col in df.columns for col in required_cols):
        warnings.warn(f"Missing required columns in {xlsx_file}")
        return False

    if 'Accepted' in df.columns:
        df = df[df['Accepted'] == 1]

    if label_map:
        df['Label'] = df['Label'].map(lambda x: label_map.get(x, x))

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf8') as f:
        for _, row in df.iterrows():
            f.write(f"{row['Begin Time (s)']:.6f}\t{row['End Time (s)']:.6f}\t{row['Label']}\n")

    return True

def copy_raw_csv(annotation_file, raw_csv_dir, audioid):
    """
    Copy raw CSV file to raw_csv directory.
    If annotation is .xlsx, convert it to CSV first.
    """
    output_csv = raw_csv_dir / f"{audioid}.csv"

    try:
        if annotation_file.suffix == '.csv':
            # Direct copy for CSV files
            shutil.copy2(annotation_file, output_csv)
        else:
            # Convert XLSX to CSV
            df = pd.read_excel(annotation_file)
            df.to_csv(output_csv, index=False)
        return True
    except Exception as e:
        warnings.warn(f"Error copying raw CSV for {audioid}: {e}")
        return False

def process_dataset(root_dir, output_dir, label_map=None):
    """Process the entire dataset."""
    root_path = Path(root_dir)
    output_path = Path(output_dir)

    all_records = []
    missing_files = []
    processing_errors = []

    for exp_dir in ['Experiment_1', 'Experiment_2']:
        exp_path = root_path / exp_dir
        if not exp_path.exists():
            warnings.warn(f"Experiment directory not found: {exp_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {exp_dir}...")
        print(f"{'='*60}")

        exp_output = output_path / f"local/{exp_dir}"
        wav_dir = exp_output / "wav"
        seg_dir = exp_output / "seg"
        raw_csv_dir = exp_output / "raw_csv"  # Add raw_csv directory

        os.makedirs(wav_dir, exist_ok=True)
        os.makedirs(seg_dir, exist_ok=True)
        os.makedirs(raw_csv_dir, exist_ok=True)  # Create raw_csv directory

        audio_dir = exp_path / "Audio"
        annotation_dir = exp_path / "Annotation"

        if not audio_dir.exists() or not annotation_dir.exists():
            warnings.warn(f"Missing Audio or Annotation directory in {exp_dir}")
            continue

        annotation_files = list(annotation_dir.glob("*.csv")) + list(annotation_dir.glob("*.xlsx"))
        annotation_files = [f for f in annotation_files if not f.name.startswith('.')]

        print(f"\nFound {len(annotation_files)} annotation files")

        wav_files = list(audio_dir.glob("*.wav"))
        wav_files = [f for f in wav_files if not f.name.startswith('.')]
        print(f"Found {len(wav_files)} audio files\n")

        for wav_file in sorted(wav_files):
            filename = wav_file.stem

            audio_info = parse_audio_filename(wav_file.name)
            if not audio_info:
                processing_errors.append(f"Could not parse audio filename: {wav_file.name}")
                print(f"  ✗ {filename}: Could not parse filename")
                continue

            match_result = find_matching_annotation(audio_info, annotation_files)

            if not match_result:
                missing_files.append(f"No annotation found for: {filename}")
                print(f"  ✗ {filename}: No matching annotation")
                continue

            annotation_file, ann_info = match_result

            # Create symbolic link for wav file
            wav_link = wav_dir / f"{filename}.wav"
            if not wav_link.exists():
                try:
                    wav_link.symlink_to(wav_file)
                except Exception as e:
                    warnings.warn(f"Could not create symlink for {wav_file}: {e}")
                    continue

            # Convert annotation to Audacity format
            seg_file = seg_dir / f"{filename}.txt"
            if annotation_file.suffix == '.csv':
                success = convert_csv_to_audacity(annotation_file, seg_file, label_map)
            else:
                success = convert_xlsx_to_audacity(annotation_file, seg_file, label_map)

            if not success:
                processing_errors.append(f"Failed to convert: {annotation_file.name}")
                print(f"  ✗ {filename}: Failed to convert annotation")
                continue

            # Copy raw CSV file
            copy_raw_csv(annotation_file, raw_csv_dir, filename)

            # Use audio info for age (more reliable), ann_info for phenotype
            all_records.append({
                'audioid': filename,
                'experiment': exp_dir,
                'date': audio_info['date'],
                'file_id': audio_info['id'],
                'subject_name': audio_info['name'],
                'phenotype': ann_info['phenotype'],
                'age_days': audio_info['age_days'],
                'age_weeks': audio_info['age_weeks'],
                'age_type': audio_info['age_type'],
                'age_value': audio_info['age_value'],
                'wav_path': str(wav_file),
                'seg_path': str(seg_file),
                'raw_csv_path': str(raw_csv_dir / f"{filename}.csv")
            })

            print(f"  ✓ {filename} → {annotation_file.name}")

        exp_records = [r for r in all_records if r['experiment'] == exp_dir]
        if exp_records:
            wav_scp_file = exp_output / "wav.scp"
            with open(wav_scp_file, 'w') as f:
                for record in sorted(exp_records, key=lambda x: x['audioid']):
                    f.write(f"{record['audioid']} {record['wav_path']}\n")

            print(f"\n✓ Created wav.scp with {len(exp_records)} entries")

    if all_records:
        df = pd.DataFrame(all_records)
        df = df.sort_values(['experiment', 'subject_name', 'age_days'])

        info_csv = output_path / "local/marmoset_info/marmoset_info.csv"
        os.makedirs(info_csv.parent, exist_ok=True)
        df.to_csv(info_csv, index=False)

        print(f"\n{'='*60}")
        print(f"Created info CSV: {info_csv}")
        print(f"Total records: {len(df)}")

        print(f"\n{'='*60}")
        print("DATASET SUMMARY")
        print(f"{'='*60}")
        print(f"Total files processed: {len(df)}")
        print(f"\nBy Phenotype:")
        for pheno, count in df.groupby('phenotype').size().items():
            print(f"  {pheno}: {count}")

        print(f"\nBy Subject:")
        for (subj, pheno), count in df.groupby(['subject_name', 'phenotype']).size().items():
            print(f"  {subj} ({pheno}): {count}")

        print(f"\nBy Experiment:")
        for exp, count in df.groupby('experiment').size().items():
            print(f"  {exp}: {count}")

        print(f"\nAge range:")
        print(f"  Days: {df['age_days'].min()} - {df['age_days'].max()}")
        print(f"  Weeks: {df['age_weeks'].min()} - {df['age_weeks'].max()}")

    if missing_files or processing_errors:
        print(f"\n{'='*60}")
        print("ISSUES")
        print(f"{'='*60}")

    if missing_files:
        print(f"\nMissing Annotations ({len(missing_files)}):")
        for msg in missing_files:
            print(f"  • {msg}")

    if processing_errors:
        print(f"\nProcessing Errors ({len(processing_errors)}):")
        for msg in processing_errors:
            print(f"  • {msg}")

    return len(all_records), len(missing_files), len(processing_errors)

def main():
    parser = argparse.ArgumentParser(description='Process autistic marmoset dataset')
    parser.add_argument('--root_dir',
                       default='/work02/home/bin-wu/workspace/projects/sandbox/test_nakanishi_marmoset/autistic_marmoset_2025_12_03')
    parser.add_argument('--output_dir',
                       default='./data')
    parser.add_argument('--label_map', default=None)

    args = parser.parse_args()

    label_map = None
    if args.label_map:
        import json
        with open(args.label_map) as f:
            label_map = json.load(f)

    print(f"\n{'='*60}")
    print("AUTISTIC MARMOSET DATASET PROCESSING")
    print(f"{'='*60}")
    print(f"Root: {args.root_dir}")
    print(f"Output: {args.output_dir}")

    processed, missing, errors = process_dataset(args.root_dir, args.output_dir, label_map)

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"✓ Processed: {processed}")
    print(f"⚠ Missing: {missing}")
    print(f"✗ Errors: {errors}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

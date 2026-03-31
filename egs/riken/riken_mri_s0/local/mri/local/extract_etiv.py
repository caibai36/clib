#!/usr/bin/env python3
"""
Extract eTIV and MaskVol from infant FreeSurfer outputs.

This script handles two different FreeSurfer processing pipelines used for infants
of different ages:

1. CUSTOM PIPELINE (younger infants):
   - Uses specialized infant processing with custom atlas registration
   - eTIV stored in separate file: stats/eTIV.txt
   - Calculated via NiftyReg registration to infant template
   - No eTIV field in aseg.stats

2. DEFAULT PIPELINE (older infants):
   - Uses standard FreeSurfer processing with --etiv flag
   - eTIV stored directly in aseg.stats
   - Calculated via standard FreeSurfer talairach registration
   - No eTIV.txt file

Both pipelines store MaskVol (brain mask volume) in aseg.stats.

The script:
- Automatically detects which pipeline was used for each subject
- Extracts eTIV from the appropriate source (eTIV.txt takes precedence if both exist)
- Extracts MaskVol from aseg.stats
- Merges with age information from info.csv files
- Generates detailed logs of processing patterns and anomalies
- Reports age ranges for subjects processed with each pipeline
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import sys


class DataExtractionLogger:
    """Logger for tracking data extraction patterns and anomalies."""

    def __init__(self):
        self.stats = {
            'etiv_txt_only': [],           # Has eTIV.txt, no eTIV in aseg.stats (expected for young)
            'aseg_only': [],               # Has eTIV in aseg.stats, no eTIV.txt (expected for old)
            'both_sources': [],            # Has both eTIV.txt AND eTIV in aseg.stats (unexpected)
            'neither_source': [],          # No eTIV from either source (error)
            'maskvol_missing': [],         # MaskVol not found in aseg.stats (error)
            'aseg_missing': [],            # aseg.stats file doesn't exist (error)
            'values_mismatch': [],         # eTIV from both sources but values differ significantly
        }

    def log_subject(self, subject_id: str, category: str, details: str = "", age: Optional[float] = None):
        """Log a subject under a specific category."""
        entry = {'id': subject_id}
        if details:
            entry['details'] = details
        if age is not None:
            entry['age'] = age
        self.stats[category].append(entry)

    def print_summary(self):
        """Print detailed summary of all findings."""
        print("\n" + "=" * 70)
        print("DETAILED EXTRACTION SUMMARY")
        print("=" * 70)

        # Expected cases
        print("\n✓ EXPECTED CASES:")
        print(f"\n  1. Custom pipeline (eTIV.txt only, no eTIV in aseg.stats):")
        print(f"     Count: {len(self.stats['etiv_txt_only'])}")
        if self.stats['etiv_txt_only']:
            ages = [s['age'] for s in self.stats['etiv_txt_only'] if 'age' in s and s['age'] is not None]
            if ages:
                print(f"     Age range: {min(ages):.1f} - {max(ages):.1f} months")
                print(f"     Mean age: {sum(ages)/len(ages):.1f} months")
            else:
                print(f"     Age range: No age data available")
            print(f"     Examples: {', '.join([s['id'] for s in self.stats['etiv_txt_only'][:3]])}")

        print(f"\n  2. Default pipeline (eTIV in aseg.stats, no eTIV.txt):")
        print(f"     Count: {len(self.stats['aseg_only'])}")
        if self.stats['aseg_only']:
            ages = [s['age'] for s in self.stats['aseg_only'] if 'age' in s and s['age'] is not None]
            if ages:
                print(f"     Age range: {min(ages):.1f} - {max(ages):.1f} months")
                print(f"     Mean age: {sum(ages)/len(ages):.1f} months")
            else:
                print(f"     Age range: No age data available")
            print(f"     Examples: {', '.join([s['id'] for s in self.stats['aseg_only'][:3]])}")

        # Unexpected/anomalous cases
        has_anomalies = False

        print("\n" + "-" * 70)
        print("⚠ UNEXPECTED/ANOMALOUS CASES:")

        if self.stats['both_sources']:
            has_anomalies = True
            print(f"\n  ⚠ Both eTIV.txt AND eTIV in aseg.stats present:")
            print(f"     Count: {len(self.stats['both_sources'])}")
            print(f"     Note: Script uses eTIV.txt value (custom pipeline takes precedence)")
            for entry in self.stats['both_sources']:
                age_str = f", age={entry['age']:.1f}mo" if 'age' in entry and entry['age'] is not None else ""
                print(f"     - {entry['id']}: {entry.get('details', '')}{age_str}")

        if self.stats['values_mismatch']:
            has_anomalies = True
            print(f"\n  ⚠ eTIV values from both sources differ significantly (>1%):")
            print(f"     Count: {len(self.stats['values_mismatch'])}")
            for entry in self.stats['values_mismatch']:
                age_str = f", age={entry['age']:.1f}mo" if 'age' in entry and entry['age'] is not None else ""
                print(f"     - {entry['id']}: {entry.get('details', '')}{age_str}")

        if self.stats['neither_source']:
            has_anomalies = True
            print(f"\n  ❌ No eTIV found in either source:")
            print(f"     Count: {len(self.stats['neither_source'])}")
            for entry in self.stats['neither_source']:
                age_str = f", age={entry['age']:.1f}mo" if 'age' in entry and entry['age'] is not None else ""
                print(f"     - {entry['id']}: {entry.get('details', '')}{age_str}")

        if self.stats['maskvol_missing']:
            has_anomalies = True
            print(f"\n  ❌ MaskVol not found in aseg.stats:")
            print(f"     Count: {len(self.stats['maskvol_missing'])}")
            for entry in self.stats['maskvol_missing']:
                age_str = f", age={entry['age']:.1f}mo" if 'age' in entry and entry['age'] is not None else ""
                print(f"     - {entry['id']}{age_str}")

        if self.stats['aseg_missing']:
            has_anomalies = True
            print(f"\n  ❌ aseg.stats file missing:")
            print(f"     Count: {len(self.stats['aseg_missing'])}")
            for entry in self.stats['aseg_missing']:
                age_str = f", age={entry['age']:.1f}mo" if 'age' in entry and entry['age'] is not None else ""
                print(f"     - {entry['id']}{age_str}")

        if not has_anomalies:
            print(f"\n  ✓ No anomalies detected!")

        print("\n" + "=" * 70)


def load_age_info(info_csv: Path, dataset_name: str) -> Dict[str, float]:
    """
    Load age information from info.csv file.

    Args:
        info_csv: Path to info.csv file
        dataset_name: 'calgary' or 'new_england'

    Returns:
        Dictionary mapping subject_id to age_rounded
    """
    age_map = {}

    if not info_csv.exists():
        print(f"Warning: Info file not found: {info_csv}")
        return age_map

    with open(info_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if dataset_name == 'calgary':
                # Calgary format: id column contains subject_id
                subject_id = row.get('id', '')
                age_rounded = row.get('age_rounded', '')
            else:  # new_england
                # New England format: id column contains subject_id with session
                subject_id = row.get('id', '')
                age_rounded = row.get('age_rounded', '')

            if subject_id and age_rounded:
                try:
                    age_map[subject_id] = float(age_rounded)
                except ValueError:
                    pass

    return age_map


def parse_aseg_stats(aseg_path: Path) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse aseg.stats file to extract eTIV and MaskVol.

    Returns:
        Tuple of (etiv, mask_vol), either can be None if not found
    """
    etiv = None
    mask_vol = None

    if not aseg_path.exists():
        return etiv, mask_vol

    with open(aseg_path, 'r') as f:
        for line in f:
            # Look for eTIV
            if 'EstimatedTotalIntraCranialVol' in line and 'eTIV' in line:
                match = re.search(r'(\d+\.\d+)\s*,\s*mm\^3', line)
                if match:
                    etiv = float(match.group(1))

            # Look for MaskVol
            if line.startswith('# Measure Mask, MaskVol'):
                match = re.search(r'(\d+\.\d+)\s*,\s*mm\^3', line)
                if match:
                    mask_vol = float(match.group(1))

    return etiv, mask_vol


def read_etiv_txt(etiv_path: Path) -> Optional[float]:
    """Read eTIV value from eTIV.txt file."""
    if not etiv_path.exists():
        return None

    with open(etiv_path, 'r') as f:
        content = f.read().strip()
        try:
            return float(content)
        except ValueError:
            return None


def extract_subject_data(
    subject_dir: Path,
    subject_id: str,
    age: Optional[float],
    logger: DataExtractionLogger
) -> Dict[str, Optional[float]]:
    """
    Extract eTIV and MaskVol for a single subject.

    Args:
        subject_dir: Path to subject's FreeSurfer directory
        subject_id: Subject identifier
        age: Age in months (or None if not available)
        logger: Logger for tracking extraction patterns

    Returns:
        Dictionary with id, etiv, mask_vol, and age_rounded
    """
    result = {
        'id': subject_id,
        'etiv': None,
        'mask_vol': None,
        'age_rounded': age
    }

    stats_dir = subject_dir / 'stats'

    # Check if aseg.stats exists
    aseg_path = stats_dir / 'aseg.stats'
    if not aseg_path.exists():
        logger.log_subject(subject_id, 'aseg_missing', age=age)
        return result

    # Get values from aseg.stats
    etiv_from_aseg, mask_vol = parse_aseg_stats(aseg_path)

    result['mask_vol'] = mask_vol

    # Check for MaskVol
    if mask_vol is None:
        logger.log_subject(subject_id, 'maskvol_missing', age=age)

    # Check for eTIV.txt
    etiv_txt_path = stats_dir / 'eTIV.txt'
    etiv_from_txt = read_etiv_txt(etiv_txt_path)

    # Determine eTIV source and log accordingly
    has_txt = etiv_from_txt is not None
    has_aseg = etiv_from_aseg is not None

    if has_txt and has_aseg:
        # Both sources present - unexpected but not necessarily wrong
        details = f"eTIV.txt={etiv_from_txt:.2f}, aseg.stats={etiv_from_aseg:.2f}"
        logger.log_subject(subject_id, 'both_sources', details, age=age)

        # Check if values match closely
        diff_percent = abs(etiv_from_txt - etiv_from_aseg) / etiv_from_aseg * 100
        if diff_percent > 1.0:  # More than 1% difference
            mismatch_details = f"eTIV.txt={etiv_from_txt:.2f}, aseg.stats={etiv_from_aseg:.2f}, diff={diff_percent:.2f}%"
            logger.log_subject(subject_id, 'values_mismatch', mismatch_details, age=age)

        # Use eTIV.txt (custom pipeline takes precedence)
        result['etiv'] = etiv_from_txt

    elif has_txt and not has_aseg:
        # Custom pipeline (young infant) - expected case
        logger.log_subject(subject_id, 'etiv_txt_only', age=age)
        result['etiv'] = etiv_from_txt

    elif not has_txt and has_aseg:
        # Default pipeline (older infant) - expected case
        logger.log_subject(subject_id, 'aseg_only', age=age)
        result['etiv'] = etiv_from_aseg

    else:
        # Neither source has eTIV - error
        details = "No eTIV.txt and no eTIV in aseg.stats"
        logger.log_subject(subject_id, 'neither_source', details, age=age)

    return result


def process_dataset(
    freesurfer_base: Path,
    dataset_name: str,
    subjects_file: Path,
    info_csv: Path,
    output_csv: Path,
    logger: DataExtractionLogger
) -> None:
    """
    Process all subjects for a given dataset.

    Args:
        freesurfer_base: Base directory containing FreeSurfer outputs
        dataset_name: Name of dataset (calgary or new_england)
        subjects_file: Path to file containing subject IDs
        info_csv: Path to info.csv with age information
        output_csv: Path to output CSV file
        logger: Logger for tracking extraction patterns
    """
    print(f"\nProcessing {dataset_name.upper()} dataset")
    print(f"=" * 70)

    # Load age information
    print(f"Loading age information from: {info_csv}")
    age_map = load_age_info(info_csv, dataset_name)
    print(f"Loaded age information for {len(age_map)} subjects")

    # Read subject IDs
    with open(subjects_file, 'r') as f:
        subject_ids = [line.strip() for line in f if line.strip()]

    print(f"Processing {len(subject_ids)} subjects from {dataset_name}...")

    # Process each subject
    results = []
    dataset_dir = freesurfer_base / dataset_name

    for subject_id in subject_ids:
        subject_dir = dataset_dir / subject_id

        # Get age for this subject
        age = age_map.get(subject_id)

        if not subject_dir.exists():
            print(f"Warning: Directory not found for {subject_id}")
            results.append({
                'id': subject_id,
                'etiv': None,
                'mask_vol': None,
                'age_rounded': age
            })
            continue

        data = extract_subject_data(subject_dir, subject_id, age, logger)
        results.append(data)

        # Print status
        status = []
        if data['etiv'] is not None:
            status.append(f"eTIV={data['etiv']:.2f}")
        else:
            status.append("eTIV=MISSING")

        if data['mask_vol'] is not None:
            status.append(f"MaskVol={data['mask_vol']:.2f}")
        else:
            status.append("MaskVol=MISSING")

        if data['age_rounded'] is not None:
            status.append(f"age={data['age_rounded']:.1f}mo")
        else:
            status.append("age=UNKNOWN")

        print(f"  {subject_id}: {', '.join(status)}")

    # Write output CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'etiv', 'mask_vol', 'age_rounded'])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWrote output to: {output_csv}")

    # Print summary
    total = len(results)
    with_etiv = sum(1 for r in results if r['etiv'] is not None)
    with_mask = sum(1 for r in results if r['mask_vol'] is not None)
    with_age = sum(1 for r in results if r['age_rounded'] is not None)

    print(f"Summary: {total} subjects")
    print(f"  - eTIV found: {with_etiv}/{total}")
    print(f"  - MaskVol found: {with_mask}/{total}")
    print(f"  - Age info found: {with_age}/{total}")

    # Print age statistics
    ages = [r['age_rounded'] for r in results if r['age_rounded'] is not None]
    if ages:
        print(f"  - Age range: {min(ages):.1f} - {max(ages):.1f} months")
        print(f"  - Mean age: {sum(ages)/len(ages):.1f} months")


def main():
    parser = argparse.ArgumentParser(
        description='Extract eTIV and MaskVol from infant FreeSurfer outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script processes infant FreeSurfer data that may have been processed with
two different pipelines:

  CUSTOM PIPELINE:
    - Uses infant-specific atlas registration (NiftyReg)
    - eTIV stored in: stats/eTIV.txt
    - Used for younger infants with rapidly developing brains

  DEFAULT PIPELINE:
    - Uses standard FreeSurfer with --etiv flag
    - eTIV stored in: stats/aseg.stats
    - Used for older infants with more mature brain anatomy

The script automatically detects which pipeline was used and extracts data
from the appropriate source. It also merges age information from info.csv
files and reports the actual age ranges for subjects processed with each
pipeline (without assuming a specific cutoff age).

Examples:
  # Process both datasets with default paths
  %(prog)s

  # Process only Calgary dataset
  %(prog)s --dataset calgary

  # Use custom FreeSurfer directory
  %(prog)s --freesurfer-dir /custom/path
        """
    )

    parser.add_argument(
        '--freesurfer-dir',
        type=Path,
        default='/data02/share/bin-wu/data/human/brain/harvard_mri/processed/infant_freesurfer/age_le_50_months',
        help='FreeSurfer processed data directory (default: /data02/share/bin-wu/data/human/brain/harvard_mri/processed/infant_freesurfer/age_le_50_months)'
    )

    parser.add_argument(
        '--dataset',
        choices=['calgary', 'new_england', 'both'],
        default='both',
        help='Which dataset to process (default: both)'
    )

    # Calgary paths
    parser.add_argument(
        '--calgary-subjects',
        type=Path,
        default='conf/mri/subjects/subjects_calgary_t1_age_le_50th_month.txt',
        help='Calgary subjects file (default: conf/mri/subjects/subjects_calgary_t1_age_le_50th_month.txt)'
    )

    parser.add_argument(
        '--calgary-info',
        type=Path,
        default='data_mri/calgary/info.csv',
        help='Calgary info CSV with age data (default: data_mri/calgary/info.csv)'
    )

    parser.add_argument(
        '--calgary-output',
        type=Path,
        default='exp/mri/affective_vocalization_analysis/batch_outputs/calgary/infant_freesurfer_etiv_calgary.csv',
        help='Calgary output CSV (default: exp/mri/affective_vocalization_analysis/batch_outputs/calgary/infant_freesurfer_etiv_calgary.csv)'
    )

    # New England paths
    parser.add_argument(
        '--new-england-subjects',
        type=Path,
        default='conf/mri/subjects/subjects_new_england_t1_age_le_50th_month.txt',
        help='New England subjects file (default: conf/mri/subjects/subjects_new_england_t1_age_le_50th_month.txt)'
    )

    parser.add_argument(
        '--new-england-info',
        type=Path,
        default='data_mri/new_england/info.csv',
        help='New England info CSV with age data (default: data_mri/new_england/info.csv)'
    )

    parser.add_argument(
        '--new-england-output',
        type=Path,
        default='exp/mri/affective_vocalization_analysis/batch_outputs/new_england/infant_free_surfer_etiv_new_england.csv',
        help='New England output CSV (default: exp/mri/affective_vocalization_analysis/batch_outputs/new_england/infant_free_surfer_etiv_new_england.csv)'
    )

    args = parser.parse_args()

    # Print script purpose
    print("=" * 70)
    print("INFANT FREESURFER eTIV AND MaskVol EXTRACTION")
    print("=" * 70)
    print("\nThis script extracts brain volume metrics from infant FreeSurfer outputs.")
    print("\nWHY TWO DIFFERENT PIPELINES?")
    print("  Infant brains undergo rapid development. Different processing approaches")
    print("  are used based on brain maturity:")
    print()
    print("  • CUSTOM PIPELINE:")
    print("    - Infant-specific atlas registration (NiftyReg)")
    print("    - eTIV stored in: stats/eTIV.txt")
    print("    - Used for younger infants")
    print()
    print("  • DEFAULT PIPELINE:")
    print("    - Standard FreeSurfer with --etiv flag")
    print("    - eTIV stored in: stats/aseg.stats")
    print("    - Used for older infants")
    print()
    print("The script will report the actual age ranges for subjects processed")
    print("with each pipeline based on the data.")
    print("=" * 70)

    # Create logger
    logger = DataExtractionLogger()

    # Process datasets
    if args.dataset in ['calgary', 'both']:
        if not args.calgary_subjects.exists():
            print(f"Error: Calgary subjects file not found: {args.calgary_subjects}")
        else:
            process_dataset(
                Path(args.freesurfer_dir),
                'calgary',
                args.calgary_subjects,
                args.calgary_info,
                args.calgary_output,
                logger
            )

    if args.dataset in ['new_england', 'both']:
        if not args.new_england_subjects.exists():
            print(f"Error: New England subjects file not found: {args.new_england_subjects}")
        else:
            process_dataset(
                Path(args.freesurfer_dir),
                'new_england',
                args.new_england_subjects,
                args.new_england_info,
                args.new_england_output,
                logger
            )

    # Print detailed summary of all findings
    logger.print_summary()


if __name__ == '__main__':
    main()

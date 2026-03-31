#!/usr/bin/env env python3
"""
Prepare CSV tables for R Linear Mixed Model analysis
6 systems: vocal_motor, cerebellum_vocal_control, limbic_vocalization_cortical,
          limbic_vocalization_subcortical, auditory_vocal_integration, basal_ganglia_vocal_control
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare brain volume data for R LMM analysis (6 systems)'
    )

    # Input files
    parser.add_argument('--calgary-volumes',
                        default='exp/mri/affective_vocalization_analysis/batch_outputs/calgary/all_subjects_affective_vocalization_calgary.csv',
                        help='Calgary brain volume CSV')
    parser.add_argument('--new-england-volumes',
                        default='exp/mri/affective_vocalization_analysis/batch_outputs/new_england/all_subjects_affective_vocalization_new_england.csv',
                        help='New England brain volume CSV')
    parser.add_argument('--calgary-info',
                        default='data_mri/calgary/info.csv',
                        help='Calgary demographic info CSV')
    parser.add_argument('--new-england-info',
                        default='data_mri/new_england/info.csv',
                        help='New England demographic info CSV')
    parser.add_argument('--calgary-etiv',
                        default='exp/mri/affective_vocalization_analysis/batch_outputs/calgary/infant_freesurfer_etiv_calgary.csv',
                        help='Calgary eTIV/MaskVol CSV')
    parser.add_argument('--new-england-etiv',
                        default='exp/mri/affective_vocalization_analysis/batch_outputs/new_england/infant_free_surfer_etiv_new_england.csv',
                        help='New England eTIV/MaskVol CSV')

    # Output
    parser.add_argument('--output-dir',
                        default='exp/mri/r_tables',
                        help='Output directory for R-ready tables')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed information')

    return parser.parse_args()

def extract_base_id(subject_id, dataset):
    """
    Extract base subject ID correctly for both datasets

    Calgary:     10006_PS14_001 → 10006
    New England: sub-01_ses-01  → sub-01
    """
    if dataset == 'calgary':
        # Take first part before underscore (preschool_id)
        return subject_id.split('_')[0]
    else:  # new_england
        # Remove session part
        return subject_id.split('_ses-')[0] if '_ses-' in subject_id else subject_id

def remap_system_to_6_categories(row):
    """
    Remap 5 systems to 6 systems by splitting limbic_vocalization

    Original 5 systems:
    - vocal_motor
    - cerebellum_vocal_control
    - limbic_vocalization (SPLIT THIS)
    - auditory_vocal_integration
    - basal_ganglia_vocal_control

    New 6 systems:
    - vocal_motor
    - cerebellum_vocal_control
    - limbic_vocalization_cortical (from aparc.stats)
    - limbic_vocalization_subcortical (from aseg.stats)
    - auditory_vocal_integration
    - basal_ganglia_vocal_control
    """
    system = row['system']
    stats_file = row['stats_file']

    if system == 'limbic_vocalization':
        # Split based on stats file
        if 'aparc.stats' in stats_file:
            return 'limbic_vocalization_cortical'
        else:  # aseg.stats
            return 'limbic_vocalization_subcortical'
    else:
        # Keep original system name
        return system

def classify_cortical_subcortical(system):
    """
    Classify 6 systems into cortical vs subcortical for 2-way analysis
    """
    if system in ['vocal_motor', 'auditory_vocal_integration', 'limbic_vocalization_cortical']:
        return 'Cortical'
    elif system in ['cerebellum_vocal_control', 'basal_ganglia_vocal_control', 'limbic_vocalization_subcortical']:
        return 'Subcortical'
    else:
        return None

def extract_volume(row):
    """Extract volume from appropriate field"""
    stats_file = row['stats_file']

    # aseg.stats uses Volume_mm3
    if 'aseg.stats' in stats_file and pd.notna(row.get('Volume_mm3')):
        return row['Volume_mm3']
    # aparc.stats uses GrayVol
    elif pd.notna(row.get('GrayVol')):
        return row['GrayVol']
    else:
        return np.nan

def prepare_calgary_info(info_df):
    """Prepare Calgary demographic info"""
    info_prep = info_df.copy()
    info_prep['id'] = info_prep['preschool_id'].astype(str) + '_' + info_prep['scan_id']
    info_prep['gender'] = info_prep['biological_sex_f0_m1'].map({0: 'F', 1: 'M'})
    info_prep['age_rounded'] = info_prep['age_rounded'].astype(float)

    return info_prep[['id', 'age_rounded', 'age', 'gender']]

def prepare_new_england_info(info_df):
    """Prepare New England demographic info"""
    info_prep = info_df.copy()
    info_prep['session_str'] = info_prep['session'].astype(int).astype(str).str.zfill(2)
    info_prep['id'] = info_prep['participant_id'] + '_ses-' + info_prep['session_str']
    info_prep['gender'] = info_prep['Sex'].map({0: 'F', 1: 'M'})
    info_prep['age_rounded'] = info_prep['age_rounded'].astype(float)

    return info_prep[['id', 'age_rounded', 'age', 'gender']]

def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("PREPARING 6-SYSTEM TABLES FOR R LMM ANALYSIS")
    print("="*80)
    print("\n6 Systems:")
    print("  1. vocal_motor")
    print("  2. cerebellum_vocal_control")
    print("  3. limbic_vocalization_cortical")
    print("  4. limbic_vocalization_subcortical")
    print("  5. auditory_vocal_integration")
    print("  6. basal_ganglia_vocal_control")

    # ===========================
    # Load Data
    # ===========================
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)

    df_calgary_vol = pd.read_csv(args.calgary_volumes)
    df_new_england_vol = pd.read_csv(args.new_england_volumes)

    info_calgary = pd.read_csv(args.calgary_info)
    info_new_england = pd.read_csv(args.new_england_info)

    etiv_calgary = pd.read_csv(args.calgary_etiv)
    etiv_new_england = pd.read_csv(args.new_england_etiv)

    print(f"\n✓ Calgary volumes: {len(df_calgary_vol)} rows")
    print(f"✓ New England volumes: {len(df_new_england_vol)} rows")

    # ===========================
    # Prepare Info
    # ===========================
    print("\n" + "="*80)
    print("PREPARING DEMOGRAPHIC INFO")
    print("="*80)

    info_calgary_prep = prepare_calgary_info(info_calgary)
    info_new_england_prep = prepare_new_england_info(info_new_england)

    print(f"\n✓ Calgary info: {len(info_calgary_prep)} subjects")
    print(f"✓ New England info: {len(info_new_england_prep)} subjects")

    # ===========================
    # Process Calgary
    # ===========================
    print("\n" + "="*80)
    print("PROCESSING CALGARY")
    print("="*80)

    df_calgary = df_calgary_vol.copy()

    # Remap to 6 systems
    df_calgary['system_6way'] = df_calgary.apply(remap_system_to_6_categories, axis=1)
    df_calgary['volume'] = df_calgary.apply(extract_volume, axis=1)
    df_calgary = df_calgary.dropna(subset=['volume'])

    # Merge with info and etiv
    df_calgary = df_calgary.merge(
        info_calgary_prep,
        left_on='subject_id',
        right_on='id',
        how='left'
    )
    df_calgary = df_calgary.merge(
        etiv_calgary,
        left_on='subject_id',
        right_on='id',
        how='left',
        suffixes=('', '_etiv')
    )
    df_calgary['dataset'] = 'calgary'

    print(f"\n✓ Processed: {len(df_calgary)} rows")
    print(f"✓ Unique subjects: {df_calgary['subject_id'].nunique()}")

    # Print system breakdown
    if args.verbose:
        print("\nSystem distribution (Calgary):")
        print(df_calgary['system_6way'].value_counts().sort_index())

    # ===========================
    # Process New England
    # ===========================
    print("\n" + "="*80)
    print("PROCESSING NEW ENGLAND")
    print("="*80)

    df_new_england = df_new_england_vol.copy()

    # Remap to 6 systems
    df_new_england['system_6way'] = df_new_england.apply(remap_system_to_6_categories, axis=1)
    df_new_england['volume'] = df_new_england.apply(extract_volume, axis=1)
    df_new_england = df_new_england.dropna(subset=['volume'])

    # Merge with info and etiv
    df_new_england = df_new_england.merge(
        info_new_england_prep,
        left_on='subject_id',
        right_on='id',
        how='left'
    )
    df_new_england = df_new_england.merge(
        etiv_new_england,
        left_on='subject_id',
        right_on='id',
        how='left',
        suffixes=('', '_etiv')
    )
    df_new_england['dataset'] = 'new_england'

    print(f"\n✓ Processed: {len(df_new_england)} rows")
    print(f"✓ Unique subjects: {df_new_england['subject_id'].nunique()}")

    # Print system breakdown
    if args.verbose:
        print("\nSystem distribution (New England):")
        print(df_new_england['system_6way'].value_counts().sort_index())

    # ===========================
    # Combine Datasets
    # ===========================
    print("\n" + "="*80)
    print("COMBINING DATASETS")
    print("="*80)

    df_all = pd.concat([df_calgary, df_new_england], ignore_index=True)

    # Remove rows with missing critical data
    df_all = df_all.dropna(subset=['volume', 'age', 'gender', 'mask_vol'])

    # Add base ID CORRECTLY for both datasets
    df_all['ID_base'] = df_all.apply(
        lambda row: extract_base_id(row['subject_id'], row['dataset']),
        axis=1
    )

    print(f"\n✓ Combined: {len(df_all)} rows")
    print(f"✓ Total unique scan IDs: {df_all['subject_id'].nunique()}")
    print(f"✓ Total unique base IDs: {df_all['ID_base'].nunique()}")

    # Verify ID extraction
    print("\n" + "-"*80)
    print("ID EXTRACTION VERIFICATION")
    print("-"*80)

    print("\nCalgary examples (first 10 unique):")
    calgary_example = df_all[df_all['dataset'] == 'calgary'][['subject_id', 'ID_base']].drop_duplicates().head(10)
    print(calgary_example.to_string(index=False))

    print("\nNew England examples (first 10 unique):")
    ne_example = df_all[df_all['dataset'] == 'new_england'][['subject_id', 'ID_base']].drop_duplicates().head(10)
    print(ne_example.to_string(index=False))

    # Check for longitudinal subjects
    print("\n" + "-"*80)
    print("LONGITUDINAL STRUCTURE")
    print("-"*80)

    scans_per_base = df_all.groupby('ID_base')['subject_id'].nunique()
    print(f"\nBase IDs with 1 scan:  {(scans_per_base == 1).sum()}")
    print(f"Base IDs with 2 scans: {(scans_per_base == 2).sum()}")
    print(f"Base IDs with 3 scans: {(scans_per_base == 3).sum()}")
    print(f"Base IDs with 4 scans: {(scans_per_base == 4).sum()}")
    print(f"Base IDs with 5 scans: {(scans_per_base == 5).sum()}")
    print(f"Base IDs with 6 scans: {(scans_per_base == 6).sum()}")
    print(f"Base IDs with 7 scans: {(scans_per_base == 7).sum()}")
    print(f"Base IDs with 8 scans: {(scans_per_base == 8).sum()}")
    print(f"Base IDs with 9 scans: {(scans_per_base == 9).sum()}")
    print(f"Base IDs with 10+ scans: {(scans_per_base >= 10).sum()}")

    # Show example of subject with multiple scans
    multi_scan_subjects = scans_per_base[scans_per_base > 1].index[:3]
    if len(multi_scan_subjects) > 0:
        print(f"\nExample: Subject {multi_scan_subjects[0]} has multiple scans:")
        example_scans = df_all[df_all['ID_base'] == multi_scan_subjects[0]][['subject_id', 'ID_base', 'age', 'dataset']].drop_duplicates().sort_values('age')
        print(example_scans.to_string(index=False))

    print("\nOverall system distribution:")
    print(df_all['system_6way'].value_counts().sort_index())

    # ===========================
    # Aggregate by System (6-way)
    # ===========================
    print("\n" + "="*80)
    print("AGGREGATING VOLUMES BY SYSTEM")
    print("="*80)

    # Sum volumes per subject per system (sums left + right hemispheres)
    volume_by_system = df_all.groupby([
        'dataset', 'subject_id', 'ID_base', 'age', 'age_rounded', 'gender', 'system_6way', 'mask_vol'
    ])['volume'].sum().reset_index()

    # Add transformed variables for R
    volume_by_system['log_age'] = np.log(volume_by_system['age'])

    print(f"\n✓ Aggregated: {len(volume_by_system)} rows")
    print(f"✓ Unique base subjects: {volume_by_system['ID_base'].nunique()}")
    print(f"  Calgary: {volume_by_system[volume_by_system['dataset']=='calgary']['ID_base'].nunique()}")
    print(f"  New England: {volume_by_system[volume_by_system['dataset']=='new_england']['ID_base'].nunique()}")

    if args.verbose:
        print("\nVolumes per system:")
        print(volume_by_system.groupby('system_6way')['volume'].agg(['count', 'mean', 'std']).round(1))

    # ===========================
    # Create Cortical vs Subcortical (2-way)
    # ===========================
    print("\n" + "="*80)
    print("CREATING CORTICAL VS SUBCORTICAL AGGREGATION")
    print("="*80)

    # Add cortical/subcortical classification
    volume_by_system['major_category'] = volume_by_system['system_6way'].apply(classify_cortical_subcortical)

    # Aggregate by major category
    cortical_subcortical = volume_by_system.groupby([
        'dataset', 'subject_id', 'ID_base', 'age', 'age_rounded',
        'log_age', 'gender', 'mask_vol', 'major_category'
    ])['volume'].sum().reset_index()

    cortical_subcortical = cortical_subcortical[cortical_subcortical['major_category'].notna()]

    print(f"\n✓ Cortical/Subcortical: {len(cortical_subcortical)} rows")

    if args.verbose:
        print("\nVolumes by category:")
        print(cortical_subcortical.groupby('major_category')['volume'].agg(['count', 'mean', 'std']).round(1))

    # ===========================
    # Print Summary Statistics
    # ===========================
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print("\n1. Age range by dataset:")
    age_stats = volume_by_system.groupby('dataset')['age'].agg(['min', 'max', 'mean', 'std'])
    print(age_stats.round(2))

    print("\n2. Subjects by dataset:")
    subj_counts = volume_by_system.groupby('dataset')['ID_base'].nunique()
    print(subj_counts)

    print("\n3. Gender distribution:")
    gender_counts = volume_by_system.groupby(['dataset', 'gender'])['ID_base'].nunique().unstack(fill_value=0)
    print(gender_counts)

    print("\n4. MaskVol statistics by dataset:")
    mask_stats = volume_by_system.groupby('dataset')['mask_vol'].agg(['mean', 'std', 'min', 'max'])
    print(mask_stats.round(1))

    # ===========================
    # Save Tables
    # ===========================
    print("\n" + "="*80)
    print("SAVING R-READY TABLES")
    print("="*80)

    # Table 1: Volume by 6 systems (for individual system analysis)
    table1 = volume_by_system[[
        'dataset', 'subject_id', 'ID_base', 'age', 'age_rounded', 'log_age',
        'gender', 'system_6way', 'mask_vol', 'volume'
    ]].copy()

    output_file1 = output_dir / 'volume_by_system_6way.csv'
    table1.to_csv(output_file1, index=False)
    print(f"\n✓ Table 1: {output_file1}")
    print(f"  Rows: {len(table1):,}")
    print(f"  Columns: {list(table1.columns)}")

    # Table 2: Cortical vs Subcortical (for 2-way analysis)
    table2 = cortical_subcortical[[
        'dataset', 'subject_id', 'ID_base', 'age', 'age_rounded', 'log_age',
        'gender', 'major_category', 'mask_vol', 'volume'
    ]].copy()

    output_file2 = output_dir / 'volume_cortical_subcortical.csv'
    table2.to_csv(output_file2, index=False)
    print(f"\n✓ Table 2: {output_file2}")
    print(f"  Rows: {len(table2):,}")
    print(f"  Columns: {list(table2.columns)}")

    # Table 3: Wide format (all 6 systems as columns)
    table3 = volume_by_system.pivot_table(
        index=['dataset', 'subject_id', 'ID_base', 'age', 'age_rounded', 'log_age', 'gender', 'mask_vol'],
        columns='system_6way',
        values='volume',
        aggfunc='sum'
    ).reset_index()

    output_file3 = output_dir / 'volume_wide_6systems.csv'
    table3.to_csv(output_file3, index=False)
    print(f"\n✓ Table 3: {output_file3}")
    print(f"  Rows: {len(table3):,}")

    # ===========================
    # Create Data Dictionary
    # ===========================
    data_dict = {
        'Column': [
            'dataset', 'subject_id', 'ID_base', 'age', 'age_rounded', 'log_age',
            'gender', 'system_6way', 'major_category', 'mask_vol', 'volume'
        ],
        'Description': [
            'Dataset name: calgary or new_england',
            'Full subject ID with scan/session (e.g., 10006_PS14_001 or sub-01_ses-01)',
            'Base subject ID for random effects (e.g., 10006 or sub-01)',
            'Age in months (continuous)',
            'Age rounded to nearest month',
            'Natural log of age (for LMM)',
            'Gender: F or M',
            '6-way system classification',
            '2-way classification: Cortical or Subcortical',
            'Mask volume (proxy for ICV) in mm³',
            'Brain volume sum for system in mm³'
        ],
        'Type': [
            'categorical', 'categorical', 'categorical', 'numeric', 'numeric', 'numeric',
            'categorical', 'categorical', 'categorical', 'numeric', 'numeric'
        ],
        'Example_Calgary': [
            'calgary', '10006_PS14_001', '10006', '49.67', '50', '3.91',
            'M', 'vocal_motor', 'Cortical', '1650020', '157977'
        ],
        'Example_NewEngland': [
            'new_england', 'sub-01_ses-01', 'sub-01', '6.70', '7', '1.90',
            'F', 'vocal_motor', 'Cortical', '778664', '87567'
        ]
    }

    data_dict_df = pd.DataFrame(data_dict)
    output_file4 = output_dir / 'data_dictionary.csv'
    data_dict_df.to_csv(output_file4, index=False)
    print(f"\n✓ Data Dictionary: {output_file4}")

    # ===========================
    # Final Summary
    # ===========================
    print("\n" + "="*80)
    print("✓ TABLE PREPARATION COMPLETE")
    print("="*80)

    print(f"\nOutput directory: {output_dir}")
    print("\nFiles created:")
    print("  1. volume_by_system_6way.csv       - 6 systems (long format)")
    print("  2. volume_cortical_subcortical.csv - 2 categories (long format)")
    print("  3. volume_wide_6systems.csv         - 6 systems (wide format)")
    print("  4. data_dictionary.csv              - Column descriptions")

    print("\n6 Systems included:")
    for i, sys in enumerate(sorted(table1['system_6way'].unique()), 1):
        count = (table1['system_6way'] == sys).sum()
        print(f"  {i}. {sys:<40} ({count:>4} observations)")

    print("\n" + "="*80)
    print("✓ ID_base correctly extracted for both datasets:")
    print("  Calgary:     10006_PS14_001 → 10006")
    print("  New England: sub-01_ses-01  → sub-01")
    print("="*80)
    print("\nReady for R LMM analysis!")
    print("Next: Run run_lmm_analysis_6systems.R")
    print("="*80)

if __name__ == '__main__':
    main()

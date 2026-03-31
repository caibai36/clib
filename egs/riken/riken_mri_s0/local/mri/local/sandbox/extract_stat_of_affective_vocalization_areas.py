#!/usr/bin/env python3
"""
Extract affective vocalization ROI measurements from FreeSurfer stats files
Based on the mapping configuration CSV file
"""

import pandas as pd
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

def parse_aseg_stats(stats_file, roi_name, label_index, verbose=False):
    """
    Parse aseg.stats file to extract volume
    
    Format:
    # ColHeaders  Index SegId NVoxels Volume_mm3 StructName normMean normStdDev normMin normMax normRange
      1   4      7238     7008.4  Left-Lateral-Ventricle  65.3670  19.1107  30.0000  154.0000  124.0000
     13  18       994      982.3  Left-Amygdala          127.8330   5.8613 105.0000  145.0000   40.0000
    
    Columns: [0]=Index, [1]=SegId, [2]=NVoxels, [3]=Volume_mm3, [4]=StructName, 
             [5]=normMean, [6]=normStdDev, [7]=normMin, [8]=normMax, [9]=normRange
    """
    try:
        with open(stats_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                if not line.strip():
                    continue
                    
                parts = line.split()
                if len(parts) >= 10:
                    try:
                        index = int(parts[0])
                        seg_id = int(parts[1])
                        nvoxels = int(parts[2])
                        volume_mm3 = float(parts[3])
                        struct_name = parts[4]
                        norm_mean = float(parts[5])
                        norm_std_dev = float(parts[6])
                        norm_min = float(parts[7])
                        norm_max = float(parts[8])
                        norm_range = float(parts[9])
                        
                        # Match by SegId (label_index) or StructName
                        if seg_id == int(label_index) or struct_name == roi_name:
                            if verbose:
                                print(f"    Found in aseg.stats: {struct_name} (SegId={seg_id}, Volume={volume_mm3})")
                            return {
                                'Index': index,
                                'SegId': seg_id,
                                'NVoxels': nvoxels,
                                'Volume_mm3': volume_mm3,
                                'StructName': struct_name,
                                'normMean': norm_mean,
                                'normStdDev': norm_std_dev,
                                'normMin': norm_min,
                                'normMax': norm_max,
                                'normRange': norm_range
                            }
                    except ValueError:
                        continue
                        
    except FileNotFoundError:
        print(f"Warning: Stats file not found: {stats_file}")
        return None
    except Exception as e:
        print(f"Error parsing {stats_file}: {e}")
        return None
    
    if verbose:
        print(f"    NOT found in aseg.stats: {roi_name} (label={label_index})")
    return None

def parse_aparc_stats(stats_file, roi_name, verbose=False):
    """
    Parse aparc.stats file to extract cortical measurements
    
    Format:
    # ColHeaders StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd
    bankssts                                   1843   1186   4412  3.606 0.588     0.101     0.022       14     1.6
    parsopercularis                             714    444   2959  4.050 0.688     0.106     0.033       11     0.8
    
    Columns: [0]=StructName, [1]=NumVert, [2]=SurfArea, [3]=GrayVol, [4]=ThickAvg, [5]=ThickStd,
             [6]=MeanCurv, [7]=GausCurv, [8]=FoldInd, [9]=CurvInd
    """
    try:
        with open(stats_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                if not line.strip():
                    continue
                    
                parts = line.split()
                if len(parts) >= 10:
                    # Column 0 is StructName
                    struct_name = parts[0]
                    
                    # Check if this is the ROI we're looking for
                    if struct_name == roi_name:
                        try:
                            num_vert = int(parts[1])
                            surf_area = float(parts[2])
                            gray_vol = float(parts[3])
                            thick_avg = float(parts[4])
                            thick_std = float(parts[5])
                            mean_curv = float(parts[6])
                            gauss_curv = float(parts[7])
                            fold_ind = float(parts[8])
                            curv_ind = float(parts[9])
                            
                            if verbose:
                                print(f"    Found in aparc.stats: {struct_name} (GrayVol={gray_vol}, ThickAvg={thick_avg}, ThickStd={thick_std})")
                            
                            return {
                                'StructName': struct_name,
                                'NumVert': num_vert,
                                'SurfArea': surf_area,
                                'GrayVol': gray_vol,
                                'ThickAvg': thick_avg,
                                'ThickStd': thick_std,
                                'MeanCurv': mean_curv,
                                'GausCurv': gauss_curv,
                                'FoldInd': fold_ind,
                                'CurvInd': curv_ind
                            }
                        except ValueError as e:
                            if verbose:
                                print(f"    Error parsing values for {struct_name}: {e}")
                            continue
                            
    except FileNotFoundError:
        print(f"Warning: Stats file not found: {stats_file}")
        return None
    except Exception as e:
        print(f"Error parsing {stats_file}: {e}")
        return None
    
    if verbose:
        print(f"    NOT found in aparc.stats: {roi_name}")
    return None

def infer_subject_id_from_stats_dir(stats_dir):
    """
    Infer subject ID from stats directory path
    
    Args:
        stats_dir: Path to stats directory
    
    Returns:
        Subject ID string
    """
    # Extract subject ID from path like: .../ifs_outputs/id1/stats/
    path_parts = Path(stats_dir).parts
    
    # Find 'stats' in path and get parent
    try:
        stats_idx = path_parts.index('stats')
        if stats_idx > 0:
            return path_parts[stats_idx - 1]
    except ValueError:
        pass
    
    # Fallback: use parent directory name
    return Path(stats_dir).parent.name

def extract_subject_data(subject_id, stats_dir, config_csv, dataset_name, verbose=False):
    """
    Extract all ROI data for a subject
    
    Args:
        subject_id: Subject identifier
        stats_dir: Path to FreeSurfer stats directory
        config_csv: Path to configuration CSV file
        dataset_name: Name of the dataset
        verbose: Print verbose output
    
    Returns:
        DataFrame with all extracted measurements
    """
    
    if verbose:
        print(f"  Reading config: {config_csv}")
    
    # Read configuration CSV WITH HEADER
    config_df = pd.read_csv(config_csv)
    
    # Verify expected columns
    expected_cols = ['system', 'label_index', 'freesurfer_stats_roi_name', 
                     'freesurfer_stats_file_name', 'hemisphere', 
                     'component_description', 'status']
    
    if not all(col in config_df.columns for col in expected_cols):
        print(f"Error: Config CSV missing required columns")
        print(f"Expected: {expected_cols}")
        print(f"Found: {list(config_df.columns)}")
        sys.exit(1)
    
    if verbose:
        print(f"  Config loaded: {len(config_df)} ROIs")
    
    results = []
    success_count = 0
    fail_count = 0
    
    for idx, row in config_df.iterrows():
        stats_path = os.path.join(stats_dir, row['freesurfer_stats_file_name'])
        
        if verbose:
            print(f"\n  [{idx+1}/{len(config_df)}] Processing: {row['freesurfer_stats_roi_name']} ({row['system']})")
            print(f"    Stats file: {row['freesurfer_stats_file_name']}")
        
        # Base result dictionary
        result = {
            'dataset_name': dataset_name,
            'subject_id': subject_id,
            'system': row['system'],
            'label_index': row['label_index'],
            'roi_name': row['freesurfer_stats_roi_name'],
            'stats_file': row['freesurfer_stats_file_name'],
            'hemisphere': row['hemisphere'],
            'component_description': row['component_description'],
            'status': row['status']
        }
        
        # Parse based on file type
        data = None
        if row['freesurfer_stats_file_name'] == 'aseg.stats':
            data = parse_aseg_stats(stats_path, row['freesurfer_stats_roi_name'], 
                                   row['label_index'], verbose)
            if data:
                # Add aseg-specific columns
                result.update({
                    'Index': data['Index'],
                    'SegId': data['SegId'],
                    'NVoxels': data['NVoxels'],
                    'Volume_mm3': data['Volume_mm3'],
                    'StructName': data['StructName'],
                    'normMean': data['normMean'],
                    'normStdDev': data['normStdDev'],
                    'normMin': data['normMin'],
                    'normMax': data['normMax'],
                    'normRange': data['normRange'],
                    # Set aparc columns to None
                    'NumVert': None,
                    'SurfArea': None,
                    'GrayVol': None,
                    'ThickAvg': None,
                    'ThickStd': None,
                    'MeanCurv': None,
                    'GausCurv': None,
                    'FoldInd': None,
                    'CurvInd': None
                })
                success_count += 1
            else:
                # Set all measurement columns to None
                result.update({
                    'Index': None, 'SegId': None, 'NVoxels': None, 'Volume_mm3': None,
                    'StructName': None, 'normMean': None, 'normStdDev': None,
                    'normMin': None, 'normMax': None, 'normRange': None,
                    'NumVert': None, 'SurfArea': None, 'GrayVol': None,
                    'ThickAvg': None, 'ThickStd': None, 'MeanCurv': None,
                    'GausCurv': None, 'FoldInd': None, 'CurvInd': None
                })
                fail_count += 1
                
        elif row['freesurfer_stats_file_name'] in ['lh.aparc.stats', 'rh.aparc.stats']:
            data = parse_aparc_stats(stats_path, row['freesurfer_stats_roi_name'], verbose)
            if data:
                # Add aparc-specific columns
                result.update({
                    'StructName': data['StructName'],
                    'NumVert': data['NumVert'],
                    'SurfArea': data['SurfArea'],
                    'GrayVol': data['GrayVol'],
                    'ThickAvg': data['ThickAvg'],
                    'ThickStd': data['ThickStd'],
                    'MeanCurv': data['MeanCurv'],
                    'GausCurv': data['GausCurv'],
                    'FoldInd': data['FoldInd'],
                    'CurvInd': data['CurvInd'],
                    # Set aseg columns to None
                    'Index': None,
                    'SegId': None,
                    'NVoxels': None,
                    'Volume_mm3': None,
                    'normMean': None,
                    'normStdDev': None,
                    'normMin': None,
                    'normMax': None,
                    'normRange': None
                })
                success_count += 1
            else:
                # Set all measurement columns to None
                result.update({
                    'Index': None, 'SegId': None, 'NVoxels': None, 'Volume_mm3': None,
                    'StructName': None, 'normMean': None, 'normStdDev': None,
                    'normMin': None, 'normMax': None, 'normRange': None,
                    'NumVert': None, 'SurfArea': None, 'GrayVol': None,
                    'ThickAvg': None, 'ThickStd': None, 'MeanCurv': None,
                    'GausCurv': None, 'FoldInd': None, 'CurvInd': None
                })
                fail_count += 1
        else:
            print(f"Warning: Unknown stats file type: {row['freesurfer_stats_file_name']}")
            result.update({
                'Index': None, 'SegId': None, 'NVoxels': None, 'Volume_mm3': None,
                'StructName': None, 'normMean': None, 'normStdDev': None,
                'normMin': None, 'normMax': None, 'normRange': None,
                'NumVert': None, 'SurfArea': None, 'GrayVol': None,
                'ThickAvg': None, 'ThickStd': None, 'MeanCurv': None,
                'GausCurv': None, 'FoldInd': None, 'CurvInd': None
            })
            fail_count += 1
        
        results.append(result)
    
    if not verbose:
        print(f"  Extracted: {success_count} ROIs, Failed: {fail_count} ROIs")
    
    return pd.DataFrame(results)

def process_multiple_subjects(subjects_file, subjects_dir, config_csv, output_dir, dataset_name, verbose=False):
    """
    Process multiple subjects from a text file
    """
    
    # Read subject list
    with open(subjects_file, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]
    
    print(f"Processing {len(subjects)} subjects from dataset: {dataset_name}")
    
    all_results = []
    
    for i, subject_id in enumerate(subjects, 1):
        print(f"\n[{i}/{len(subjects)}] Processing subject: {subject_id}")
        print("-" * 60)
        
        # Construct stats directory path
        stats_dir = os.path.join(subjects_dir, subject_id, 'stats')
        
        if not os.path.exists(stats_dir):
            print(f"  Warning: Stats directory not found: {stats_dir}")
            continue
        
        # Extract data
        df = extract_subject_data(subject_id, stats_dir, config_csv, dataset_name, verbose)
        all_results.append(df)
        
        # Create subject-specific output directory
        subject_output_dir = os.path.join(output_dir, subject_id)
        os.makedirs(subject_output_dir, exist_ok=True)
        
        # Save individual subject file
        output_file = os.path.join(subject_output_dir, f"{subject_id}_affective_vocalization.csv")
        df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")
    
    # Combine all subjects
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_file = os.path.join(output_dir, f"all_subjects_affective_vocalization_{dataset_name}.csv")
        combined_df.to_csv(combined_file, index=False)
        print(f"\n{'='*60}")
        print(f"Combined results saved to: {combined_file}")
        print(f"{'='*60}")
        
        # Print summary
        print_summary(combined_df)
        
        return combined_df
    else:
        print("No subjects were successfully processed.")
        return None

def print_summary(df):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nDataset: {df['dataset_name'].iloc[0]}")
    print(f"Total subjects: {df['subject_id'].nunique()}")
    print(f"Total measurements: {len(df)}")
    
    # Count successes based on whether we have any measurement data
    has_data = df['GrayVol'].notna() | df['Volume_mm3'].notna()
    print(f"Successful extractions: {has_data.sum()} ({has_data.sum()/len(df)*100:.1f}%)")
    print(f"Failed extractions: {(~has_data).sum()} ({(~has_data).sum()/len(df)*100:.1f}%)")
    
    print("\n" + "-"*60)
    print("BY SYSTEM")
    print("-"*60)
    system_summary = df.groupby('system').apply(
        lambda x: pd.Series({
            'Total': len(x),
            'Success': (x['GrayVol'].notna() | x['Volume_mm3'].notna()).sum(),
            'Failed': (x['GrayVol'].isna() & x['Volume_mm3'].isna()).sum()
        })
    )
    print(system_summary)
    
    print("\n" + "-"*60)
    print("BY HEMISPHERE")
    print("-"*60)
    hemi_summary = df.groupby('hemisphere').apply(
        lambda x: pd.Series({
            'Total': len(x),
            'Success': (x['GrayVol'].notna() | x['Volume_mm3'].notna()).sum()
        })
    )
    print(hemi_summary)
    
    # Print failed ROIs if any
    failed_rois = df[(df['GrayVol'].isna()) & (df['Volume_mm3'].isna())]
    if len(failed_rois) > 0:
        print("\n" + "-"*60)
        print("FAILED EXTRACTIONS")
        print("-"*60)
        for idx, row in failed_rois.iterrows():
            print(f"  {row['roi_name']} ({row['stats_file']}) - {row['system']}")
    
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(
        description='Extract affective vocalization ROI measurements from FreeSurfer stats files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python %(prog)s --stats-dir /path/to/ifs_outputs/id1/stats
  python %(prog)s --stats-dir /path/to/id1/stats --verbose
  python %(prog)s --subjects-dir /path/to/ifs_outputs --subjects-file subjects.txt
        '''
    )
    
    parser.add_argument('--stats-dir', type=str,
                       help='Direct path to stats directory')
    parser.add_argument('--subjects-dir', type=str,
                       help='Base directory containing subject directories')
    parser.add_argument('--subjects-file', type=str,
                       help='Text file with one subject ID per line')
    parser.add_argument('--output', type=str, 
                       default='exp/mri/affective_vocalization_analysis/outputs',
                       help='Output directory')
    parser.add_argument('--config', type=str, 
                       default='conf/mri/affective_vocalization_freesurfer_dk_atlas.csv',
                       help='Configuration CSV file')
    parser.add_argument('--dataset', type=str, default='test',
                       help='Dataset name')
    parser.add_argument('--subject-id', type=str,
                       help='Override subject ID')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output')
    
    args = parser.parse_args()
    
    if not args.stats_dir and not (args.subjects_dir and args.subjects_file):
        parser.error("Must specify either --stats-dir or both --subjects-dir and --subjects-file")
    
    os.makedirs(args.output, exist_ok=True)
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    print("="*60)
    print("AFFECTIVE VOCALIZATION ROI EXTRACTION")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print("="*60 + "\n")
    
    if args.stats_dir:
        if not os.path.exists(args.stats_dir):
            print(f"Error: Stats directory not found: {args.stats_dir}")
            sys.exit(1)
        
        subject_id = args.subject_id or infer_subject_id_from_stats_dir(args.stats_dir)
        print(f"Subject ID: {subject_id}\n")
        
        df = extract_subject_data(subject_id, args.stats_dir, args.config, args.dataset, args.verbose)
        
        subject_output_dir = os.path.join(args.output, subject_id)
        os.makedirs(subject_output_dir, exist_ok=True)
        
        output_file = os.path.join(subject_output_dir, f"{subject_id}_affective_vocalization.csv")
        df.to_csv(output_file, index=False)
        
        print(f"\nOutput: {output_file}")
        print_summary(df)
    else:
        process_multiple_subjects(args.subjects_file, args.subjects_dir, 
                                 args.config, args.output, args.dataset, args.verbose)

if __name__ == "__main__":
    main()

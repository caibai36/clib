#!/usr/bin/env python3
"""
Create summary table from actual CSV data analysis with correct birth dates.
"""

import pandas as pd
import argparse
from datetime import datetime

def duration_to_seconds(dur_str):
    """Convert duration string (H:MM:SS) to seconds."""
    if pd.isna(dur_str) or dur_str in ['unknown', 'not_calculated']:
        return 0
    try:
        parts = str(dur_str).split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except:
        pass
    return 0

def seconds_to_duration_string(total_seconds):
    """Convert seconds to duration string format."""
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    return f"{hours}h{minutes:02d}m{seconds:02d}s"

def get_birth_date(dataid):
    """Get the correct birth date for each family based on the screenshot table."""
    birth_dates = {
        'booth0_f1': '20230722',
        'booth1_f1': '20240302', 
        'booth2_f1': '20240309',
        'booth3_f1': '20240202',
        'booth4_f1': '20240130',
        'booth1_f2': '20240805',
        'booth2_f2': '20240810', 
        'booth3_f2': '20240706',
        'booth4_f2': '20240704'
    }
    return birth_dates.get(dataid, 'unknown')

def get_upload_status(dataid):
    """Get the upload status for each family based on the screenshot table."""
    status_map = {
        'booth0_f1': 'finished',
        'booth1_f1': 'uploading',
        'booth2_f1': 'finished', 
        'booth3_f1': 'uploading',
        'booth4_f1': 'uploading',
        'booth1_f2': 'uploading',
        'booth2_f2': 'uploading',
        'booth3_f2': 'uploading', 
        'booth4_f2': 'uploading'
    }
    return status_map.get(dataid, 'uploading')

def create_summary_from_csv(csv_file):
    """Create summary table by analyzing actual CSV data."""
    df = pd.read_csv(csv_file)
    
    print(f"Analyzing {len(df)} records from CSV...")
    print(f"Unique dataid values found: {sorted(df['dataid'].unique())}")
    
    # Convert duration to seconds for calculation
    df['duration_seconds'] = df['audio_duration'].apply(duration_to_seconds)
    
    # Parse dates
    df['date_parsed'] = pd.to_datetime(df['date'], format='%Y_%m_%d', errors='coerce')
    
    summary_rows = []
    
    # Group by dataid and analyze each family/generation
    for dataid, group in df.groupby('dataid'):
        print(f"Processing {dataid}: {len(group)} files")
        
        # Get family info (should be consistent within group)
        mother = group['mother'].iloc[0] if len(group) > 0 else 'unknown'
        father = group['father'].iloc[0] if len(group) > 0 else 'unknown'
        infant = group['infant'].iloc[0] if len(group) > 0 else 'unknown'
        
        # Calculate recording date range
        valid_dates = group['date_parsed'].dropna()
        if not valid_dates.empty:
            start_date = valid_dates.min().strftime('%Y%m%d')
            end_date = valid_dates.max().strftime('%Y%m%d')
            
            if start_date == end_date:
                rec_span = start_date
            else:
                rec_span = f"{start_date}-{end_date}"
            
            # Count unique recording days
            rec_days = valid_dates.dt.date.nunique()
        else:
            rec_span = 'unknown'
            rec_days = 0
        
        # Calculate file and duration statistics
        rec_files = len(group)
        total_seconds = group['duration_seconds'].sum()
        rec_hours = seconds_to_duration_string(total_seconds)
        
        # Convert dataid format (e.g., b4_f2 -> booth4_f2)
        display_dataid = dataid.replace('b', 'booth')
        
        # Get birth date and upload status from predefined mapping
        birth_date = get_birth_date(display_dataid)
        upload_status = get_upload_status(display_dataid)
        
        summary_rows.append({
            'dataid': display_dataid,
            'uploaded_state': upload_status,
            'members': 'family',
            'mother': mother,
            'father': father,
            'infant': infant,
            'birth_date': birth_date,
            'rec_span': rec_span,
            '#rec_days': rec_days,
            '#rec_files': rec_files,
            '#rec_hours': rec_hours
        })
    
    # Create DataFrame and sort
    summary_df = pd.DataFrame(summary_rows)
    
    # Sort by booth number and generation
    def sort_key(dataid):
        parts = dataid.replace('booth', '').split('_')
        booth_num = int(parts[0]) if parts[0].isdigit() else 0
        generation = parts[1] if len(parts) > 1 else 'f1'
        return (booth_num, generation)
    
    summary_df['sort_key'] = summary_df['dataid'].apply(sort_key)
    summary_df = summary_df.sort_values('sort_key').drop('sort_key', axis=1)
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(description='Create summary table from marmoset CSV data')
    parser.add_argument('-i', '--input', default='data/nas5_f1_f2_dur/nas5_f1_f2_dur.csv',
                       help='Input CSV file')
    parser.add_argument('-o', '--output', default='data/nas5_f1_f2_dur/family_summary.csv',
                       help='Output CSV file')
    parser.add_argument('--show-sample', action='store_true',
                       help='Show sample of input data')
    
    args = parser.parse_args()
    
    try:
        # Show sample data if requested
        if args.show_sample:
            df = pd.read_csv(args.input)
            print("Sample of input data:")
            print(df.head())
            print(f"\nColumns: {list(df.columns)}")
            print(f"Total rows: {len(df)}")
            print(f"Unique dataid values: {sorted(df['dataid'].unique())}")
            return
        
        # Create summary table
        summary_df = create_summary_from_csv(args.input)
        
        # Save to CSV
        summary_df.to_csv(args.output, index=False)
        
        # Display table
        print("\nFamily Summary Table:")
        print("=" * 120)
        print(summary_df.to_string(index=False))
        print(f"\nTable saved to: {args.output}")
        
        # Show statistics
        print(f"\nStatistics:")
        print(f"Total families/generations: {len(summary_df)}")
        print(f"Total files analyzed: {summary_df['#rec_files'].sum()}")
        print(f"Total recording days: {summary_df['#rec_days'].sum()}")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file {args.input}")
        print("Make sure the CSV file exists and the path is correct.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Extract metadata from marmoset vocalization audio files and create CSV table.
Processes file paths to extract booth, generation, family info, and audio duration.
"""

import argparse
import os
import re
import csv
import subprocess
from pathlib import Path

def get_audio_duration(filepath):
    """Get audio duration using sox."""
    try:
        result = subprocess.run(['sox', '--info', '-D', filepath], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            seconds = float(result.stdout.strip())
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}:{minutes:02d}:{secs:02d}"
    except Exception as e:
        print(f"Error getting duration for {filepath}: {e}")
    return "unknown"

def parse_filepath(filepath):
    """Extract metadata from file path."""
    path = Path(filepath)
    
    # Determine generation from path
    if 'nas5_f2' in str(path):
        generation = 'f2'
    elif 'nas5' in str(path):
        generation = 'f1'
    else:
        generation = 'unknown'
    
    # Extract booth and family info from directory name
    # Pattern: b4_1372F_1169M_3196M
    family_pattern = r'(b\d+)_(\d+F)_(\d+M)_(\d+[MF])'
    family_match = re.search(family_pattern, str(path))
    
    if family_match:
        booth = family_match.group(1)
        mother = family_match.group(2)
        father = family_match.group(3)
        infant = family_match.group(4)
        dataid = f"{booth}_{generation}"
    else:
        booth = mother = father = infant = dataid = "unknown"
    
    # Extract date from path (format: 20241011)
    date_pattern = r'(\d{8})'
    date_match = re.search(date_pattern, str(path))
    if date_match:
        date_str = date_match.group(1)
        formatted_date = f"{date_str[:4]}_{date_str[4:6]}_{date_str[6:8]}"
    else:
        formatted_date = "unknown"
    
    # Extract mic info
    if 'mic_L' in str(path):
        mic = 'mic_L'
    elif 'mic_R' in str(path):
        mic = 'mic_R'
    else:
        mic = 'unknown'
    
    # Extract audio filename (without extension)
    audioid = path.stem
    
    return {
        'dataid': dataid,
        'audioid': audioid,
        'mother': mother,
        'father': father,
        'infant': infant,
        'date': formatted_date,
        'mic': mic,
        'filepath': str(path)
    }

def main():
    # Default paths
    default_paths = [
        '/data03/share/bin-wu/data/marmoset/vocalization/riken_long/nas5_f2/',
        '/data01/share/bin-wu/data/marmoset/vocalization/riken_long/nas5/b*'
    ]
    
    parser = argparse.ArgumentParser(description='Extract marmoset audio metadata to CSV')
    parser.add_argument('paths', nargs='*', default=default_paths,
                       help=f'Search paths for audio files (default: {" ".join(default_paths)})')
    parser.add_argument('-o', '--output', default='data/nas5_f1_f2_dur/nas5_f1_f2_dur.csv', 
                       help='Output CSV filename (default: data/nas5_f1_f2_dur/nas5_f1_f2_dur.csv)')
    parser.add_argument('--no-duration', action='store_true',
                       help='Skip duration calculation (faster)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Find all wav files
    wav_files = []
    for search_path in args.paths:
        # Handle glob patterns for paths like /path/b*
        if '*' in search_path:
            import glob
            expanded_paths = glob.glob(search_path)
            for expanded_path in expanded_paths:
                if os.path.exists(expanded_path):
                    for root, dirs, files in os.walk(expanded_path):
                        for file in files:
                            if file.endswith('.wav'):
                                wav_files.append(os.path.join(root, file))
        else:
            if os.path.exists(search_path):
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        if file.endswith('.wav'):
                            wav_files.append(os.path.join(root, file))
    
    print(f"Found {len(wav_files)} WAV files")
    
    # Process files and create CSV
    with open(args.output, 'w', newline='') as csvfile:
        fieldnames = ['dataid', 'audioid', 'mother', 'father', 'infant', 
                     'date', 'mic', 'audio_duration']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for filepath in wav_files:
            metadata = parse_filepath(filepath)
            
            # Get duration if requested
            if not args.no_duration:
                duration = get_audio_duration(filepath)
            else:
                duration = "not_calculated"
            
            metadata['audio_duration'] = duration
            
            # Remove filepath before writing to CSV
            del metadata['filepath']
            writer.writerow(metadata)
            
            print(f"Processed: {metadata['dataid']} - {metadata['audioid']}")
    
    print(f"CSV saved to: {args.output}")

if __name__ == "__main__":
    main()

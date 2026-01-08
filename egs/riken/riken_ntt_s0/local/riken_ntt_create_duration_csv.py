#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
import subprocess
from collections import defaultdict
import statistics


def get_audio_duration(wav_path):
    """Get duration of audio file using soxi or ffprobe."""
    try:
        # Try soxi first
        result = subprocess.run(
            ['soxi', '-D', str(wav_path)],
            capture_output=True,
            text=True,
            check=True
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # Fall back to ffprobe
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 
                 'format=duration', '-of', 
                 'default=noprint_wrappers=1:nokey=1', str(wav_path)],
                capture_output=True,
                text=True,
                check=True
            )
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            return None


def seconds_to_hms(seconds):
    """Convert seconds to HH:MM:SS.mmm format."""
    if seconds is None:
        return ''
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def save_statistics(sessions, dur_info_dir):
    """Save statistics to multiple CSV files."""
    
    # Create output directory
    dur_info_path = Path(dur_info_dir)
    dur_info_path.mkdir(parents=True, exist_ok=True)
    
    # Collect durations by different fields
    durations_by_subject = defaultdict(list)
    durations_by_month = defaultdict(list)
    durations_by_subject_month = defaultdict(list)
    all_durations = []
    session_counts = defaultdict(int)
    
    for session in sessions:
        if session['dur_sec']:
            dur = float(session['dur_sec'])
            all_durations.append(dur)
            durations_by_subject[session['subject']].append(dur)
            durations_by_month[session['month']].append(dur)
            key = f"{session['subject']}_{session['month']}"
            durations_by_subject_month[key].append(dur)
            session_counts[session['subject']] += 1
    
    # 1. Overall statistics
    overall_stats = []
    if all_durations:
        overall_stats.append({
            'metric': 'total_sessions',
            'value': len(all_durations),
            'unit': 'count'
        })
        overall_stats.append({
            'metric': 'total_duration_sec',
            'value': f"{sum(all_durations):.3f}",
            'unit': 'seconds'
        })
        overall_stats.append({
            'metric': 'total_duration_hms',
            'value': seconds_to_hms(sum(all_durations)),
            'unit': 'HH:MM:SS.mmm'
        })
        overall_stats.append({
            'metric': 'mean_duration_sec',
            'value': f"{statistics.mean(all_durations):.3f}",
            'unit': 'seconds'
        })
        overall_stats.append({
            'metric': 'median_duration_sec',
            'value': f"{statistics.median(all_durations):.3f}",
            'unit': 'seconds'
        })
        if len(all_durations) > 1:
            overall_stats.append({
                'metric': 'std_dev_sec',
                'value': f"{statistics.stdev(all_durations):.3f}",
                'unit': 'seconds'
            })
        overall_stats.append({
            'metric': 'min_duration_sec',
            'value': f"{min(all_durations):.3f}",
            'unit': 'seconds'
        })
        overall_stats.append({
            'metric': 'max_duration_sec',
            'value': f"{max(all_durations):.3f}",
            'unit': 'seconds'
        })
    
    with open(dur_info_path / 'overall_stats.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['metric', 'value', 'unit'])
        writer.writeheader()
        writer.writerows(overall_stats)
    
    # 2. Statistics by subject
    subject_stats = []
    for subject in sorted(durations_by_subject.keys()):
        durs = durations_by_subject[subject]
        subject_stats.append({
            'subject': subject,
            'num_sessions': len(durs),
            'total_dur_sec': f"{sum(durs):.3f}",
            'total_dur_hms': seconds_to_hms(sum(durs)),
            'mean_dur_sec': f"{statistics.mean(durs):.3f}",
            'median_dur_sec': f"{statistics.median(durs):.3f}",
            'std_dev_sec': f"{statistics.stdev(durs):.3f}" if len(durs) > 1 else '',
            'min_dur_sec': f"{min(durs):.3f}",
            'max_dur_sec': f"{max(durs):.3f}"
        })
    
    with open(dur_info_path / 'stats_by_subject.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'subject', 'num_sessions', 'total_dur_sec', 'total_dur_hms',
            'mean_dur_sec', 'median_dur_sec', 'std_dev_sec', 'min_dur_sec', 'max_dur_sec'
        ])
        writer.writeheader()
        writer.writerows(subject_stats)
    
    # 3. Statistics by month
    month_stats = []
    for month in sorted(durations_by_month.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        durs = durations_by_month[month]
        month_stats.append({
            'month': month,
            'num_sessions': len(durs),
            'total_dur_sec': f"{sum(durs):.3f}",
            'total_dur_hms': seconds_to_hms(sum(durs)),
            'mean_dur_sec': f"{statistics.mean(durs):.3f}",
            'median_dur_sec': f"{statistics.median(durs):.3f}",
            'std_dev_sec': f"{statistics.stdev(durs):.3f}" if len(durs) > 1 else '',
            'min_dur_sec': f"{min(durs):.3f}",
            'max_dur_sec': f"{max(durs):.3f}"
        })
    
    with open(dur_info_path / 'stats_by_month.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'month', 'num_sessions', 'total_dur_sec', 'total_dur_hms',
            'mean_dur_sec', 'median_dur_sec', 'std_dev_sec', 'min_dur_sec', 'max_dur_sec'
        ])
        writer.writeheader()
        writer.writerows(month_stats)
    
    # 4. Statistics by subject and month
    subject_month_stats = []
    for key in sorted(durations_by_subject_month.keys()):
        subject, month = key.split('_')
        durs = durations_by_subject_month[key]
        subject_month_stats.append({
            'subject': subject,
            'month': month,
            'num_sessions': len(durs),
            'total_dur_sec': f"{sum(durs):.3f}",
            'total_dur_hms': seconds_to_hms(sum(durs)),
            'mean_dur_sec': f"{statistics.mean(durs):.3f}",
            'median_dur_sec': f"{statistics.median(durs):.3f}",
            'std_dev_sec': f"{statistics.stdev(durs):.3f}" if len(durs) > 1 else '',
            'min_dur_sec': f"{min(durs):.3f}",
            'max_dur_sec': f"{max(durs):.3f}"
        })
    
    with open(dur_info_path / 'stats_by_subject_month.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'subject', 'month', 'num_sessions', 'total_dur_sec', 'total_dur_hms',
            'mean_dur_sec', 'median_dur_sec', 'std_dev_sec', 'min_dur_sec', 'max_dur_sec'
        ])
        writer.writeheader()
        writer.writerows(subject_month_stats)
    
    # 5. Session count by subject
    subject_count = []
    for subject in sorted(session_counts.keys()):
        subject_count.append({
            'subject': subject,
            'num_sessions': session_counts[subject]
        })
    
    with open(dur_info_path / 'session_count_by_subject.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['subject', 'num_sessions'])
        writer.writeheader()
        writer.writerows(subject_count)
    
    # 6. Create a summary README
    readme_content = f"""Duration Statistics Summary
===========================

Generated files:
1. overall_stats.csv - Overall dataset statistics
2. stats_by_subject.csv - Statistics grouped by subject
3. stats_by_month.csv - Statistics grouped by month
4. stats_by_subject_month.csv - Statistics grouped by subject and month
5. session_count_by_subject.csv - Session counts per subject

Total Sessions: {len(all_durations)}
Total Duration: {sum(all_durations):.2f} seconds ({seconds_to_hms(sum(all_durations))})
Number of Subjects: {len(durations_by_subject)}
Number of Months: {len(durations_by_month)}

Subject Summary:
"""
    
    for subject in sorted(durations_by_subject.keys()):
        durs = durations_by_subject[subject]
        readme_content += f"  - {subject}: {len(durs)} sessions, {sum(durs):.2f} sec total\n"
    
    with open(dur_info_path / 'README.txt', 'w') as f:
        f.write(readme_content)
    
    print(f"\nStatistics saved to: {dur_info_dir}")
    print(f"  - overall_stats.csv")
    print(f"  - stats_by_subject.csv")
    print(f"  - stats_by_month.csv")
    print(f"  - stats_by_subject_month.csv")
    print(f"  - session_count_by_subject.csv")
    print(f"  - README.txt")


def create_duration_csv(base_dir, output_csv, metadata_csv=None, dur_info_dir=None):
    """Create CSV with session_id, subject, month, dur_sec, and dur_hms."""
    
    # Create output directory if it doesn't exist
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    sessions = []
    
    if metadata_csv and Path(metadata_csv).exists():
        # Use existing metadata CSV
        print(f"Reading metadata from {metadata_csv}")
        with open(metadata_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            seen_sessions = set()
            for row in reader:
                session_id = row['session_id']
                if session_id not in seen_sessions:
                    seen_sessions.add(session_id)
                    subject = row['subject']
                    month = row['month']
                    
                    wav_path = Path(base_dir) / subject / 'wav' / f"{session_id}.wav"
                    
                    if wav_path.exists():
                        duration = get_audio_duration(wav_path)
                        sessions.append({
                            'session_id': session_id,
                            'subject': subject,
                            'month': month,
                            'dur_sec': f"{duration:.3f}" if duration else '',
                            'dur_hms': seconds_to_hms(duration)
                        })
                        print(f"Processed: {session_id} - {duration:.3f} sec" if duration else f"Processed: {session_id} - no duration")
    else:
        # Scan directory structure
        print(f"Scanning directory: {base_dir}")
        base_path = Path(base_dir)
        
        for subject_dir in sorted(base_path.iterdir()):
            if subject_dir.is_dir():
                subject = subject_dir.name
                wav_dir = subject_dir / 'wav'
                
                if wav_dir.exists() and wav_dir.is_dir():
                    for wav_file in sorted(wav_dir.glob('*.wav')):
                        session_id = wav_file.stem
                        
                        # Extract month from session_id (e.g., kk001_1 -> 001 -> 1)
                        import re
                        match = re.match(r'^[a-z]+0*([0-9]+)_', session_id)
                        month = match.group(1) if match else '00'
                        
                        duration = get_audio_duration(wav_file)
                        
                        sessions.append({
                            'session_id': session_id,
                            'subject': subject,
                            'month': month,
                            'dur_sec': f"{duration:.3f}" if duration else '',
                            'dur_hms': seconds_to_hms(duration)
                        })
                        print(f"Processed: {session_id} - {duration:.3f} sec" if duration else f"Processed: {session_id} - no duration")
    
    # Write main CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['session_id', 'subject', 'month', 'dur_sec', 'dur_hms'])
        writer.writeheader()
        writer.writerows(sessions)
    
    print(f"\nMain CSV created: {output_csv}")
    print(f"Total sessions: {len(sessions)}")
    
    # Save statistics to separate directory
    if dur_info_dir:
        save_statistics(sessions, dur_info_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Create CSV with session duration information from NTT infant audio data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_duration_csv.py
  python create_duration_csv.py --output my_durations.csv
  python create_duration_csv.py --dur_info_dir my_stats
  python create_duration_csv.py --no_stats
        """
    )
    
    parser.add_argument(
        '--base_dir',
        type=str,
        default='/data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/processed/ntt_infant_data',
        help='Base directory containing subject folders (default: %(default)s)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/local/ntt_infant_duration.csv',
        help='Output CSV file path (default: %(default)s)'
    )
    
    parser.add_argument(
        '--metadata_csv',
        type=str,
        default='/data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/processed/all_kana_comment_token.csv',
        help='Existing metadata CSV file (optional, for faster processing)'
    )
    
    parser.add_argument(
        '--dur_info_dir',
        type=str,
        default='data/local/duration_info',
        help='Directory to save duration statistics CSV files (default: %(default)s)'
    )
    
    parser.add_argument(
        '--no_stats',
        action='store_true',
        help='Disable saving duration statistics'
    )
    
    args = parser.parse_args()
    
    create_duration_csv(
        args.base_dir, 
        args.output, 
        args.metadata_csv,
        dur_info_dir=None if args.no_stats else args.dur_info_dir
    )


if __name__ == '__main__':
    main()

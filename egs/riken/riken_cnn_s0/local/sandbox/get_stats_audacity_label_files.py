import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Dict
from sys import stdin

def parse_label_file(lines: List[str]) -> pd.DataFrame:
    """
    Parse Audacity label data with format: start_time end_time call_type
    
    Args:
        lines (List[str]): List of lines containing label data
    
    Returns:
        pd.DataFrame: DataFrame with columns ['start', 'end', 'call_type', 'duration']
    """
    data = []
    for line in lines:
        if line.strip():
            parts = line.strip().split()
            if len(parts) == 3:
                start, end, call_type = parts
                data.append([float(start), float(end), call_type])
    
    df = pd.DataFrame(data, columns=['start', 'end', 'call_type'])
    df['duration'] = df['end'] - df['start']
    return df

def calculate_audio_duration(files: List[str], duration: Optional[float], df: pd.DataFrame) -> float:
    """
    Calculate total audio duration. By default, sums up end times of all input files.
    If reading from stdin, uses the max end time from current data.
    
    Args:
        files (List[str]): List of input file paths
        duration (Optional[float]): User-provided duration value
        df (pd.DataFrame): DataFrame containing the current data
        
    Returns:
        float: Total audio duration in seconds
    """
    if duration is not None:
        return duration
    
    if not files:
        # For stdin, use max end time of current data
        return df['end'].max() if not df.empty else 0.0
    
    # For multiple files, sum up each file's max end time
    total_duration = 0.0
    for file in files:
        with open(file, 'r') as f:
            temp_df = parse_label_file(f.readlines())
            if not temp_df.empty:
                total_duration += temp_df['end'].max()
    
    return total_duration if total_duration > 0 else df['end'].max()

def calculate_statistics(df: pd.DataFrame, total_duration: float) -> Dict:
    """
    Calculate audio statistics.
    
    Args:
        df (pd.DataFrame): DataFrame with label data
        total_duration (float): Total audio duration
        
    Returns:
        Dict: Dictionary of calculated statistics
    """
    # Calculate duration statistics per call type
    duration_stats = df.groupby('call_type')['duration'].agg(['mean', 'std', 'min', 'max'])
    
    stats = {
        'total_vocal_duration': df['duration'].sum(),
        'total_audio_duration': total_duration,
        'vocal_ratio': df['duration'].sum() / total_duration,
        'call_counts': df['call_type'].value_counts().to_dict(),
        'call_durations': duration_stats.to_dict('index')
    }
    return stats

def plot_statistics(stats: Dict, save_path: Optional[str] = None, show: bool = False):
    """
    Generate and optionally save/show plots.
    
    Args:
        stats (Dict): Statistics dictionary
        save_path (Optional[str]): Path to save figures
        show (bool): Whether to display figures
    """
    # Plot call frequency
    plt.figure(figsize=(10, 6))
    plt.bar(stats['call_counts'].keys(), stats['call_counts'].values())
    plt.title('Call Type Frequency')
    plt.xlabel('Call Type')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    
    if save_path:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / 'call_frequency.png', bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()

    # Plot average durations
    plt.figure(figsize=(10, 6))
    mean_durations = {k: v['mean'] for k, v in stats['call_durations'].items()}
    plt.bar(mean_durations.keys(), mean_durations.values())
    plt.title('Average Call Duration by Type')
    plt.xlabel('Call Type')
    plt.ylabel('Duration (seconds)')
    plt.xticks(rotation=45)
    
    if save_path:
        plt.savefig(save_dir / 'call_durations.png', bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()

def main():
    """Main function to process label files and generate statistics."""
    parser = argparse.ArgumentParser(
        description='Analyze audio statistics from Audacity label files'
    )
    parser.add_argument(
        'files', 
        nargs='*', 
        help='Path to label file(s). If not provided, reads from stdin (stdin only supports a file)'
    )
    parser.add_argument(
        '--duration', 
        type=float,
        help='Total audio duration. If not provided, uses sum of max end times for multiple files or max end time for single file/stdin'
    )
    parser.add_argument(
        '--save-path',
        help='Directory to save figures'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display figures in pop-up windows'
    )
    args = parser.parse_args()

    # Process input
    if not args.files:
        # Reading from stdin
        lines = [line for line in stdin]
        if not lines:
            print("No input received from stdin")
            return
        df = parse_label_file(lines)
    else:
        dfs = []
        for file_path in args.files:
            with open(file_path, 'r') as f:
                df_temp = parse_label_file(f.readlines())
                dfs.append(df_temp)
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    if df.empty:
        print("No valid data found in input")
        return

    # Calculate total duration
    total_duration = calculate_audio_duration(args.files, args.duration, df)
    
    # Calculate statistics
    stats = calculate_statistics(df, total_duration)
    
    # Print summary
    print(f"Total vocal duration: {stats['total_vocal_duration']:.2f} seconds")
    print(f"Total audio duration: {stats['total_audio_duration']:.2f} seconds")
    print(f"Vocal ratio: {stats['vocal_ratio']:.2%}")
    print("\nCall type frequencies:")
    for call_type, count in sorted(stats['call_counts'].items()):
        print(f"{call_type}: {count}")

    if (args.save_path or args.show) and not df.empty:
        plot_statistics(stats, args.save_path, args.show)

if __name__ == "__main__":
    main()

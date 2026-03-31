import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def add_speaker_and_create_info(config_path, meta_speaker_path, tsne_path, output_dir):
    """
    Add speaker information from meta_all_speaker.csv and create info.csv and info_mae.csv

    Args:
        config_path: Path to config CSV file
        meta_speaker_path: Path to meta_all_speaker.csv with speaker information
        tsne_path: Path to t-SNE numpy array file
        output_dir: Directory for output files
    """
    # Load config CSV
    print(f"Loading config from: {config_path}")
    df_config = pd.read_csv(config_path)
    print(f"Config shape: {df_config.shape}")

    # Load meta_all_speaker CSV
    print(f"Loading speaker metadata from: {meta_speaker_path}")
    df_speaker = pd.read_csv(meta_speaker_path)
    print(f"Speaker metadata shape: {df_speaker.shape}")

    # Select only necessary columns from speaker metadata
    df_speaker_subset = df_speaker[['dataid', 'audioid', 'utt_id', 'utt_id_index', 'speaker']]

    # Merge config with speaker information
    print("Merging config with speaker information...")
    df_info = df_config.merge(
        df_speaker_subset,
        on=['dataid', 'audioid', 'utt_id', 'utt_id_index'],
        how='left'
    )

    # Check for missing speaker information
    missing_speaker = df_info['speaker'].isna().sum()
    if missing_speaker > 0:
        print(f"WARNING: {missing_speaker} rows have missing speaker information")

    # Reorder columns to match expected format
    # Original columns + speaker inserted before duration
    cols = list(df_config.columns)
    if 'duration' in cols:
        duration_idx = cols.index('duration')
        new_cols = cols[:duration_idx] + ['speaker'] + cols[duration_idx:]
    else:
        new_cols = cols + ['speaker']

    df_info = df_info[new_cols]

    # Save info.csv
    info_path = Path(output_dir) / "info.csv"
    print(f"Saving info.csv to: {info_path}")
    df_info.to_csv(info_path, index=False)
    print(f"Successfully created info.csv with shape: {df_info.shape}")

    # Load t-SNE results
    print(f"\nLoading t-SNE results from: {tsne_path}")
    tsne_data = np.load(tsne_path)
    print(f"t-SNE shape: {tsne_data.shape}")

    # Check if dimensions match
    if len(df_info) != tsne_data.shape[0]:
        raise ValueError(
            f"Dimension mismatch: info has {len(df_info)} rows, "
            f"t-SNE has {tsne_data.shape[0]} rows"
        )

    # Add t-SNE columns
    df_info_mae = df_info.copy()
    df_info_mae['tsne_1'] = tsne_data[:, 0]
    df_info_mae['tsne_2'] = tsne_data[:, 1]

    # Save info_mae.csv
    info_mae_path = Path(output_dir) / "info_mae.csv"
    print(f"Saving info_mae.csv to: {info_mae_path}")
    df_info_mae.to_csv(info_mae_path, index=False)
    print(f"Successfully created info_mae.csv with shape: {df_info_mae.shape}")

    # Show first few rows
    print("\nFirst few rows of info.csv:")
    print(df_info.head(3))
    print("\nFirst few rows of info_mae.csv:")
    print(df_info_mae.head(3))

    # Speaker distribution
    print("\nSpeaker distribution:")
    print(df_info['speaker'].value_counts())

def main():
    parser = argparse.ArgumentParser(
        description='Add speaker information and create info.csv and info_mae.csv'
    )

    parser.add_argument(
        '--config_path',
        type=str,
        default='exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa_data_ma/config_mae_pretrain_sa_data_ma.csv',
        help='Path to config CSV file'
    )

    parser.add_argument(
        '--meta_speaker_path',
        type=str,
        default='/data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/processed/metas_speaker_force_alignment_phone/meta_all_speaker.csv',
        help='Path to meta_all_speaker.csv with speaker information'
    )

    parser.add_argument(
        '--tsne_path',
        type=str,
        default='exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa_data_ma/tsne_result_mae_pretrain_sa_data_ma.npy',
        help='Path to t-SNE numpy array file'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa_data_ma',
        help='Directory for output files (info.csv and info_mae.csv)'
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create info.csv and info_mae.csv
    add_speaker_and_create_info(
        args.config_path,
        args.meta_speaker_path,
        args.tsne_path,
        args.output_dir
    )

if __name__ == '__main__':
    main()

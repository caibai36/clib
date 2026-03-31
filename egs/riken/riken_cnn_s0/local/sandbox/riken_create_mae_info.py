import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def create_mae_info(config_path, tsne_path, output_path):
    """
    Merge config CSV with t-SNE results to create info_mae.csv

    Args:
        config_path: Path to config CSV file
        tsne_path: Path to t-SNE numpy array file
        output_path: Path for output info_mae.csv file
    """
    # Load config CSV
    print(f"Loading config from: {config_path}")
    df_config = pd.read_csv(config_path)
    print(f"Config shape: {df_config.shape}")

    # Load t-SNE results
    print(f"Loading t-SNE results from: {tsne_path}")
    tsne_data = np.load(tsne_path)
    print(f"t-SNE shape: {tsne_data.shape}")

    # Check if dimensions match (accounting for header row in CSV)
    if len(df_config) != tsne_data.shape[0]:
        raise ValueError(
            f"Dimension mismatch: config has {len(df_config)} rows, "
            f"t-SNE has {tsne_data.shape[0]} rows"
        )

    # Add t-SNE columns
    df_config['tsne_1'] = tsne_data[:, 0]
    df_config['tsne_2'] = tsne_data[:, 1]

    # Save to output
    print(f"Saving to: {output_path}")
    df_config.to_csv(output_path, index=False)
    print(f"Successfully created info_mae.csv with shape: {df_config.shape}")

    # Show first few rows
    print("\nFirst few rows:")
    print(df_config.head(3))

def main():
    parser = argparse.ArgumentParser(
        description='Create info_mae.csv by merging config CSV with t-SNE results'
    )

    parser.add_argument(
        '--config_path',
        type=str,
        default='exp/mae_feat_dim_reduction/python_script_nas5/b1_f1_15weeks/config_b1_f1_15weeks.csv',
        help='Path to config CSV file (default: exp/mae_feat_dim_reduction/python_script_nas5/b1_f1_15weeks/config_b1_f1_15weeks.csv)'
    )

    parser.add_argument(
        '--tsne_path',
        type=str,
        default='exp/mae_feat_dim_reduction/python_script_nas5/b1_f1_15weeks/tsne_result_b1_f1_15weeks.npy',
        help='Path to t-SNE numpy array file (default: exp/mae_feat_dim_reduction/python_script_nas5/b1_f1_15weeks/tsne_result_b1_f1_15weeks.npy)'
    )

    parser.add_argument(
        '--output_path',
        type=str,
        default='exp/mae_feat_dim_reduction/python_script_nas5/b1_f1_15weeks/info_mae.csv',
        help='Path for output info_mae.csv file (default: exp/mae_feat_dim_reduction/python_script_nas5/b1_f1_15weeks/info_mae.csv)'
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create info_mae.csv
    create_mae_info(args.config_path, args.tsne_path, args.output_path)

if __name__ == '__main__':
    main()

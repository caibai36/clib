#!/usr/bin/env python3
"""
MAE Feature Dimensionality Reduction with Age Filtering for Marmoset Dataset

This script extends mae_dim_reduction.py with age filtering capabilities.

Usage:
    python local/mae_dim_reduction_filter_age.py \
        --info_csv exp/sandbox/exp1/info.csv \
        --mae_features exp/sandbox/exp1/mae_raw.npy \
        --output_dir exp/sandbox/exp1_filtered \
        --filter_age_col_name age_days \
        --begin_age 0 \
        --end_age 105
"""

import numpy as np
import pandas as pd
import os
import argparse
import sys
import time
import datetime
import pytz
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import sklearn

def print_timestamp():
    """Return current timestamp in Tokyo timezone."""
    return datetime.datetime.now(pytz.timezone('Asia/Tokyo')).strftime("%a %d %b %Y %I:%M:%S %p JST")

class Logger(object):
    """Redirect stdout to both console and log file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def filter_by_age(info_df, mae_features, age_col, begin_age, end_age):
    """
    Filter data by age range.

    Args:
        info_df: DataFrame with age information
        mae_features: Feature array
        age_col: Column name for age
        begin_age: Minimum age (inclusive)
        end_age: Maximum age (exclusive)

    Returns:
        tuple: (filtered_df, filtered_features, mask)
    """
    print("\n=== AGE FILTERING ===")
    print(f"Age column: {age_col}")
    print(f"Age range: [{begin_age}, {end_age})")
    print(f"Original size: {len(info_df)} samples")

    mask = (info_df[age_col] >= begin_age) & (info_df[age_col] < end_age)
    filtered_df = info_df[mask].reset_index(drop=True)
    filtered_features = mae_features[mask]

    print(f"Filtered size: {len(filtered_df)} samples")
    print(f"Removed: {len(info_df) - len(filtered_df)} samples")
    print(f"Age range in filtered data: [{filtered_df[age_col].min()}, {filtered_df[age_col].max()}]")

    return filtered_df, filtered_features, mask

def perform_dimensionality_reduction(features, pca_components=50, tsne_perplexity=30.0,
                                     tsne_iterations=1000, random_seed=42):
    """
    Perform PCA followed by t-SNE dimensionality reduction.
    """
    print("\n=== DIMENSIONALITY REDUCTION ===")
    print(f"scikit-learn version: {sklearn.__version__}")

    # Standardize features
    print("Standardizing features...")
    start_time = time.time()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    print(f"Standardization completed in {time.time() - start_time:.2f} seconds")

    # PCA
    print(f"Applying PCA with {pca_components} components...")
    start_time = time.time()
    pca = PCA(n_components=pca_components, random_state=random_seed)
    pca_result = pca.fit_transform(features_scaled)
    print(f"PCA completed in {time.time() - start_time:.2f} seconds")
    print(f"Explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")

    # t-SNE
    print(f"Applying t-SNE (perplexity={tsne_perplexity}, iterations={tsne_iterations})...")
    print(f"Started at: {print_timestamp()}")
    start_time = time.time()

    sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
    if sklearn_version >= (1, 2):
        tsne = TSNE(
            n_components=2,
            perplexity=tsne_perplexity,
            max_iter=tsne_iterations,
            random_state=random_seed,
            verbose=1
        )
    else:
        tsne = TSNE(
            n_components=2,
            perplexity=tsne_perplexity,
            n_iter=tsne_iterations,
            random_state=random_seed,
            verbose=1
        )

    tsne_result = tsne.fit_transform(pca_result)
    print(f"t-SNE completed in {time.time() - start_time:.2f} seconds")
    print(f"Finished at: {print_timestamp()}")

    return pca_result, tsne_result

def process_experiment(info_csv, mae_features_path, output_dir,
                       filter_age_col_name='age_days', begin_age=0, end_age=105,
                       pca_components=50, tsne_perplexity=30.0,
                       tsne_iterations=1000, random_seed=42):
    """
    Process one experiment with age filtering.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set up logging
    log_file = os.path.join(output_dir, "mae_reduction_filter_age.log")
    sys.stdout = Logger(log_file)

    print("=" * 60)
    print("MAE FEATURE DIMENSIONALITY REDUCTION (AGE FILTERED)")
    print("=" * 60)
    print(f"Started at: {print_timestamp()}")
    print(f"Random seed: {random_seed}")

    # Load data
    print("\n=== LOADING DATA ===")
    print(f"Loading info CSV: {info_csv}")
    info_df = pd.read_csv(info_csv)
    print(f"Info loaded: {len(info_df)} rows")

    print(f"Loading MAE features: {mae_features_path}")
    start_time = time.time()
    mae_features = np.load(mae_features_path)
    print(f"Features loaded in {time.time() - start_time:.2f} seconds")
    print(f"Features shape: {mae_features.shape}")

    # Verify shape match
    if mae_features.shape[0] != len(info_df):
        raise ValueError(f"Shape mismatch: features ({mae_features.shape[0]}) != info rows ({len(info_df)})")

    # Filter by age
    filtered_df, filtered_features, mask = filter_by_age(
        info_df, mae_features, filter_age_col_name, begin_age, end_age
    )

    # Perform dimensionality reduction
    pca_result, tsne_result = perform_dimensionality_reduction(
        filtered_features,
        pca_components=pca_components,
        tsne_perplexity=tsne_perplexity,
        tsne_iterations=tsne_iterations,
        random_seed=random_seed
    )

    # Save results
    print("\n=== SAVING RESULTS ===")

    # Save t-SNE as NPY
    tsne_output_path = os.path.join(output_dir, 'mae_tsne.npy')
    np.save(tsne_output_path, tsne_result)
    print(f"Saved t-SNE results to {tsne_output_path}")
    print(f"t-SNE shape: {tsne_result.shape}")

    # Append t-SNE to filtered info CSV
    filtered_df['tsne_1'] = tsne_result[:, 0]
    filtered_df['tsne_2'] = tsne_result[:, 1]

    info_mae_output_path = os.path.join(output_dir, 'info_mae.csv')
    filtered_df.to_csv(info_mae_output_path, index=False)
    print(f"Saved filtered info with t-SNE to {info_mae_output_path}")
    print(f"Added columns: tsne_1, tsne_2")

    # Print summary statistics
    print("\n=== SUMMARY ===")
    print(f"Input features: {mae_features.shape}")
    print(f"Age filter: {filter_age_col_name} in [{begin_age}, {end_age})")
    print(f"Filtered features: {filtered_features.shape}")
    print(f"PCA components: {pca_components}")
    print(f"t-SNE output: {tsne_result.shape}")
    print(f"t-SNE dim 1 range: [{tsne_result[:, 0].min():.2f}, {tsne_result[:, 0].max():.2f}]")
    print(f"t-SNE dim 2 range: [{tsne_result[:, 1].min():.2f}, {tsne_result[:, 1].max():.2f}]")

    print(f"\nCompleted at: {print_timestamp()}")
    print("=" * 60)

    # Reset stdout
    sys.stdout = sys.stdout.terminal
    print(f"\nProcessing complete!")
    print(f"Results saved to: {output_dir}")
    print(f"  - {tsne_output_path}")
    print(f"  - {info_mae_output_path}")
    print(f"Log saved to: {log_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Perform dimensionality reduction on MAE features with age filtering'
    )

    # Required arguments
    parser.add_argument('--info_csv', type=str, required=True,
                        help='Path to info CSV file')
    parser.add_argument('--mae_features', type=str, required=True,
                        help='Path to MAE features NPY file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')

    # Age filtering arguments
    parser.add_argument('--filter_age_col_name', type=str, default='age_days',
                        help='Column name for age filtering (default: age_days)')
    parser.add_argument('--begin_age', type=int, default=0,
                        help='Minimum age (inclusive, default: 0)')
    parser.add_argument('--end_age', type=int, default=105,
                        help='Maximum age (exclusive, default: 105)')

    # Optional parameters
    parser.add_argument('--pca_components', type=int, default=50,
                        help='Number of PCA components (default: 50)')
    parser.add_argument('--tsne_perplexity', type=float, default=30.0,
                        help='t-SNE perplexity parameter (default: 30.0)')
    parser.add_argument('--tsne_iterations', type=int, default=1000,
                        help='Number of t-SNE iterations (default: 1000)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.random_seed)

    # Process experiment
    process_experiment(
        args.info_csv,
        args.mae_features,
        args.output_dir,
        filter_age_col_name=args.filter_age_col_name,
        begin_age=args.begin_age,
        end_age=args.end_age,
        pca_components=args.pca_components,
        tsne_perplexity=args.tsne_perplexity,
        tsne_iterations=args.tsne_iterations,
        random_seed=args.random_seed
    )

if __name__ == "__main__":
    main()

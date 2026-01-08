#!/usr/bin/env python3
"""
MAE Feature Dimensionality Reduction for Marmoset Dataset

This script:
1. Loads MAE features and info CSV
2. Performs dimensionality reduction using PCA + t-SNE
3. Saves t-SNE results as separate file
4. Appends t-SNE coordinates to info CSV

Usage:
    python local/mae_dim_reduction.py \
        --info_csv exp/sandbox/exp1/info.csv \
        --mae_features exp/sandbox/exp1/mae_raw.npy \
        --output_dir exp/sandbox/exp1
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

def perform_dimensionality_reduction(features, pca_components=50, tsne_perplexity=30.0,
                                     tsne_iterations=1000, random_seed=42):
    """
    Perform PCA followed by t-SNE dimensionality reduction.

    Args:
        features: Input features array (N, D)
        pca_components: Number of PCA components
        tsne_perplexity: t-SNE perplexity parameter
        tsne_iterations: Number of t-SNE iterations
        random_seed: Random seed for reproducibility

    Returns:
        tuple: (pca_result, tsne_result)
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

    # t-SNE - use correct parameter name based on sklearn version
    print(f"Applying t-SNE (perplexity={tsne_perplexity}, iterations={tsne_iterations})...")
    print(f"Started at: {print_timestamp()}")
    start_time = time.time()

    # Check sklearn version to use correct parameter name
    sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:2]))
    if sklearn_version >= (1, 2):
        # Newer versions use max_iter
        tsne = TSNE(
            n_components=2,
            perplexity=tsne_perplexity,
            max_iter=tsne_iterations,
            random_state=random_seed,
            verbose=1
        )
    else:
        # Older versions use n_iter
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
                       pca_components=50, tsne_perplexity=30.0,
                       tsne_iterations=1000, random_seed=42):
    """
    Process one experiment: load data, perform reduction, save results.

    Args:
        info_csv: Path to info CSV file
        mae_features_path: Path to MAE features NPY file
        output_dir: Output directory
        pca_components: Number of PCA components
        tsne_perplexity: t-SNE perplexity
        tsne_iterations: t-SNE iterations
        random_seed: Random seed
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set up logging
    log_file = os.path.join(output_dir, "mae_reduction.log")
    sys.stdout = Logger(log_file)

    print("=" * 60)
    print("MAE FEATURE DIMENSIONALITY REDUCTION")
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

    # Perform dimensionality reduction
    pca_result, tsne_result = perform_dimensionality_reduction(
        mae_features,
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

    # Append t-SNE to info CSV
    info_df['tsne_1'] = tsne_result[:, 0]
    info_df['tsne_2'] = tsne_result[:, 1]

    info_mae_output_path = os.path.join(output_dir, 'info_mae.csv')
    info_df.to_csv(info_mae_output_path, index=False)
    print(f"Saved info with t-SNE to {info_mae_output_path}")
    print(f"Added columns: tsne_1, tsne_2")

    # Print summary statistics
    print("\n=== SUMMARY ===")
    print(f"Input features: {mae_features.shape}")
    print(f"PCA components: {pca_components}")
    print(f"t-SNE output: {tsne_result.shape}")
    print(f"t-SNE dim 1 range: [{tsne_result[:, 0].min():.2f}, {tsne_result[:, 0].max():.2f}]")
    print(f"t-SNE dim 2 range: [{tsne_result[:, 1].min():.2f}, {tsne_result[:, 1].max():.2f}]")

    # # Summary by phenotype
    # print("\n=== PHENOTYPE SUMMARY ===")
    # phenotype_counts = info_df.groupby('phenotype').size()
    # for pheno, count in phenotype_counts.items():
    #     print(f"{pheno}: {count} samples")

    # # Summary by subject
    # print("\n=== SUBJECT SUMMARY ===")
    # subject_counts = info_df.groupby(['subject_name', 'phenotype']).size()
    # for (subj, pheno), count in subject_counts.items():
    #     print(f"{subj} ({pheno}): {count} samples")

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
        description='Perform dimensionality reduction on MAE features'
    )

    # Required arguments
    parser.add_argument('--info_csv', type=str, required=True,
                        help='Path to info CSV file (e.g., exp/sandbox/exp1/info.csv)')
    parser.add_argument('--mae_features', type=str, required=True,
                        help='Path to MAE features NPY file (e.g., exp/sandbox/exp1/mae_raw.npy)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory (e.g., exp/sandbox/exp1)')

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
        pca_components=args.pca_components,
        tsne_perplexity=args.tsne_perplexity,
        tsne_iterations=args.tsne_iterations,
        random_seed=args.random_seed
    )

if __name__ == "__main__":
    main()

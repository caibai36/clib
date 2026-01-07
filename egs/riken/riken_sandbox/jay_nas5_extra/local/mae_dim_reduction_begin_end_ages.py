#!/usr/bin/env python3
"""
MAE Feature Dimensionality Reduction for Marmoset Dataset with Age Filtering

This script:
1. Loads MAE features and info CSV
2. Applies age filtering (optional)
3. Performs dimensionality reduction using PCA + t-SNE
4. Saves t-SNE results as separate file
5. Appends t-SNE coordinates to info CSV

Usage:
    # Process all ages
    python local/mae_dim_reduction_begin_end_ages.py \
        --info_csv exp/sandbox/exp1/info.csv \
        --mae_features exp/sandbox/exp1/mae_raw.npy \
        --output_dir exp/sandbox/exp1
    
    # Filter by age range with custom output names
    python local/mae_dim_reduction_begin_end_ages.py \
        --info_csv exp/sandbox/exp1/info.csv \
        --mae_features exp/sandbox/exp1/mae_raw.npy \
        --output_dir exp/sandbox/exp1 \
        --age_col age_days \
        --begin_age 0 \
        --end_age 10 \
        --mae_tsne mae_tsne_age0_10.npy \
        --info_mae info_mae_age0_10.csv
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

def apply_age_filter(info_df, features, age_col, begin_age, end_age):
    """
    Apply age filtering to info DataFrame and features.

    Args:
        info_df: Info DataFrame
        features: Features array
        age_col: Column name for age
        begin_age: Lower bound for age (inclusive), None means use min
        end_age: Upper bound for age (exclusive), None means use max+1

    Returns:
        tuple: (filtered_info_df, filtered_features, age_mask)
    """
    print("\n=== AGE FILTERING ===")

    # Check if age column exists
    if age_col not in info_df.columns:
        raise ValueError(f"Age column '{age_col}' not found in info CSV. Available columns: {list(info_df.columns)}")

    # Determine actual age range bounds
    if begin_age is None:
        begin_age = int(info_df[age_col].min())
        print(f"Using minimum age from data: {begin_age} days")
    else:
        print(f"Using specified begin_age: {begin_age} days")

    if end_age is None:
        # Add 1 to max to make range exclusive
        end_age = int(info_df[age_col].max()) + 1
        print(f"Using maximum age from data: {end_age-1} days")
    else:
        print(f"Using specified end_age: {end_age} days (exclusive)")

    # Create age mask
    print(f"Filtering ages: {begin_age} to {end_age-1} days (inclusive)")
    age_mask = (info_df[age_col] >= begin_age) & (info_df[age_col] < end_age)

    # Apply filtering
    filtered_info_df = info_df[age_mask].copy()
    filtered_features = features[age_mask]

    print(f"After filtering:")
    print(f"  - Features shape: {filtered_features.shape}")
    print(f"  - Info rows: {len(filtered_info_df)}")
    print(f"  - Kept {len(filtered_info_df) / len(info_df) * 100:.2f}% of original data")

    if len(filtered_info_df) == 0:
        raise ValueError(f"No data remaining after age filtering! Check your age range: [{begin_age}, {end_age})")

    return filtered_info_df, filtered_features, age_mask

def process_experiment(info_csv, mae_features_path, output_dir,
                       age_col=None, begin_age=None, end_age=None,
                       mae_tsne_name='mae_tsne.npy',
                       info_mae_name='info_mae.csv',
                       pca_components=50, tsne_perplexity=30.0,
                       tsne_iterations=1000, random_seed=42):
    """
    Process one experiment: load data, filter by age, perform reduction, save results.

    Args:
        info_csv: Path to info CSV file
        mae_features_path: Path to MAE features NPY file
        output_dir: Output directory
        age_col: Column name for age (if None, skip filtering)
        begin_age: Lower bound for age (inclusive), None means use min
        end_age: Upper bound for age (exclusive), None means use max+1
        mae_tsne_name: Output filename for t-SNE results (default: mae_tsne.npy)
        info_mae_name: Output filename for info with t-SNE (default: info_mae.csv)
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
    print("MAE FEATURE DIMENSIONALITY REDUCTION WITH AGE FILTERING")
    print("=" * 60)
    print(f"Started at: {print_timestamp()}")
    print(f"Random seed: {random_seed}")

    # Load data
    print("\n=== LOADING DATA ===")
    print(f"Loading info CSV: {info_csv}")
    info_df = pd.read_csv(info_csv)
    print(f"Info loaded: {len(info_df)} rows")
    print(f"Available columns: {list(info_df.columns)}")

    print(f"\nLoading MAE features: {mae_features_path}")
    start_time = time.time()
    mae_features = np.load(mae_features_path)
    print(f"Features loaded in {time.time() - start_time:.2f} seconds")
    print(f"Features shape: {mae_features.shape}")

    # Verify shape match
    if mae_features.shape[0] != len(info_df):
        raise ValueError(f"Shape mismatch: features ({mae_features.shape[0]}) != info rows ({len(info_df)})")

    # Apply age filtering if age_col is specified
    age_mask = None
    if age_col is not None:
        info_df, mae_features, age_mask = apply_age_filter(
            info_df, mae_features, age_col, begin_age, end_age
        )
    else:
        print("\n=== AGE FILTERING SKIPPED ===")
        print("No age_col specified, processing all data")

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

    # Save age mask if filtering was applied
    if age_mask is not None:
        mask_file = os.path.join(output_dir, 'age_mask.txt')
        np.savetxt(mask_file, age_mask.astype(int), fmt='%d')
        print(f"Saved age mask to {mask_file}")

    # Save filtered features
    features_output_path = os.path.join(output_dir, 'mae_filtered.npy')
    np.save(features_output_path, mae_features)
    print(f"Saved filtered MAE features to {features_output_path}")
    print(f"Filtered features shape: {mae_features.shape}")

    # Save t-SNE as NPY with custom filename
    tsne_output_path = os.path.join(output_dir, mae_tsne_name)
    np.save(tsne_output_path, tsne_result)
    print(f"Saved t-SNE results to {tsne_output_path}")
    print(f"t-SNE shape: {tsne_result.shape}")

    # Append t-SNE to info CSV
    info_df['tsne_1'] = tsne_result[:, 0]
    info_df['tsne_2'] = tsne_result[:, 1]

    # Save info with t-SNE using custom filename
    info_mae_output_path = os.path.join(output_dir, info_mae_name)
    info_df.to_csv(info_mae_output_path, index=False)
    print(f"Saved info with t-SNE to {info_mae_output_path}")
    print(f"Added columns: tsne_1, tsne_2")

    # Save filtered info (without t-SNE columns) as well
    info_filtered_path = os.path.join(output_dir, 'info_filtered.csv')
    info_df.drop(columns=['tsne_1', 'tsne_2']).to_csv(info_filtered_path, index=False)
    print(f"Saved filtered info (without t-SNE) to {info_filtered_path}")

    # Print summary statistics
    print("\n=== SUMMARY ===")
    print(f"Input features: {mae_features.shape}")
    print(f"PCA components: {pca_components}")
    print(f"t-SNE output: {tsne_result.shape}")
    print(f"t-SNE dim 1 range: [{tsne_result[:, 0].min():.2f}, {tsne_result[:, 0].max():.2f}]")
    print(f"t-SNE dim 2 range: [{tsne_result[:, 1].min():.2f}, {tsne_result[:, 1].max():.2f}]")

    # Age statistics if age_col is available
    if age_col is not None and age_col in info_df.columns:
        print(f"\n=== AGE STATISTICS ===")
        print(f"Age column: {age_col}")
        print(f"Age range in filtered data: {info_df[age_col].min():.1f} to {info_df[age_col].max():.1f} days")
        print(f"Mean age: {info_df[age_col].mean():.2f} days")
        print(f"Median age: {info_df[age_col].median():.2f} days")
        print(f"Total samples: {len(info_df)}")

    print(f"\n=== OUTPUT FILES ===")
    print(f"t-SNE file: {mae_tsne_name}")
    print(f"Info with t-SNE: {info_mae_name}")

    print(f"\nCompleted at: {print_timestamp()}")
    print("=" * 60)

    # Reset stdout
    sys.stdout = sys.stdout.terminal
    print(f"\nProcessing complete!")
    print(f"Results saved to: {output_dir}")
    print(f"  - {tsne_output_path}")
    print(f"  - {info_mae_output_path}")
    print(f"  - {features_output_path}")
    if age_mask is not None:
        print(f"  - {mask_file}")
    print(f"Log saved to: {log_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Perform dimensionality reduction on MAE features with optional age filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all ages with default output names
    python local/mae_dim_reduction_begin_end_ages.py \\
        --info_csv exp/sandbox/exp1/info.csv \\
        --mae_features exp/sandbox/exp1/mae_raw.npy \\
        --output_dir exp/sandbox/exp1

    # Filter by age range with custom output names
    python local/mae_dim_reduction_begin_end_ages.py \\
        --info_csv exp/sandbox/exp1/info.csv \\
        --mae_features exp/sandbox/exp1/mae_raw.npy \\
        --output_dir exp/sandbox/exp1 \\
        --age_col age_days \\
        --begin_age 0 \\
        --end_age 10 \\
        --mae_tsne mae_tsne_age0_10.npy \\
        --info_mae info_mae_age0_10.csv
        """
    )

    # Required arguments
    parser.add_argument('--info_csv', type=str, required=True,
                        help='Path to info CSV file (e.g., exp/sandbox/exp1/info.csv)')
    parser.add_argument('--mae_features', type=str, required=True,
                        help='Path to MAE features NPY file (e.g., exp/sandbox/exp1/mae_raw.npy)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory (e.g., exp/sandbox/exp1)')

    # Age filtering arguments
    parser.add_argument('--age_col', type=str, default=None,
                        help='Column name for age in CSV (default: None, skip filtering)')
    parser.add_argument('--begin_age', type=int, default=None,
                        help='Lower bound for age filtering (inclusive). None means use minimum age from data.')
    parser.add_argument('--end_age', type=int, default=None,
                        help='Upper bound for age filtering (exclusive). None means use maximum age from data + 1.')

    # Output filename arguments
    parser.add_argument('--mae_tsne', type=str, default='mae_tsne.npy',
                        help='Output filename for t-SNE results (default: mae_tsne.npy)')
    parser.add_argument('--info_mae', type=str, default='info_mae.csv',
                        help='Output filename for info CSV with t-SNE columns (default: info_mae.csv)')

    # Dimensionality reduction parameters
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

    # Validate age filtering arguments
    if args.begin_age is not None and args.end_age is not None:
        if args.begin_age >= args.end_age:
            raise ValueError(f"begin_age ({args.begin_age}) must be less than end_age ({args.end_age})")

    if (args.begin_age is not None or args.end_age is not None) and args.age_col is None:
        print("WARNING: begin_age or end_age specified but age_col is None. Age filtering will be skipped.")
        print("         To enable age filtering, specify --age_col parameter.")

    # Process experiment
    process_experiment(
        args.info_csv,
        args.mae_features,
        args.output_dir,
        age_col=args.age_col,
        begin_age=args.begin_age,
        end_age=args.end_age,
        mae_tsne_name=args.mae_tsne,
        info_mae_name=args.info_mae,
        pca_components=args.pca_components,
        tsne_perplexity=args.tsne_perplexity,
        tsne_iterations=args.tsne_iterations,
        random_seed=args.random_seed
    )

if __name__ == "__main__":
    main()

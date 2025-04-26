#!/usr/bin/env python3
"""
MAE Feature Dimensionality Reduction Script

This script loads MAE features and their config file, applies age filtering,
and performs dimensionality reduction using PCA, t-SNE, and UMAP.
Results are saved to specified output directories.

Author: Based on work by bin-wu
Date: March 25, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os
import argparse
import datetime
import pytz
import sys
import time
import umap

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

# # MAE feature
# (mlp) [bin-wu@s186 riken_cnn_s0]$(master *+) python local/riken_mae_pretrained_feature_extraction.py --save_reconstructions --pretrained_path exp/mae_pretraining/nas5_b2_mae_pretrained_48days/base/model_epoch_400_base_48days.pt --output_dir exp/mae_feat/division_nas5_b2_mae_pretrained_all_days
# (mlp) [bin-wu@s186 riken_cnn_s0]$(master *+) python local/riken_mae_pretrained_feature_extraction.py --gpu 2 --input_file exp/data/division_nas5_b2_mae_pretrained_all_days/train_input.npy  --save_reconstructions --pretrained_path exp/mae_pretraining/nas5_b2_mae_pretrained_48days/base/model_epoch_400_base_48days.pt --output_dir exp/mae_feat/division_nas5_b2_mae_pretrained_all_days

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process MAE features and perform dimensionality reduction.')
    parser.add_argument('--features_file', type=str, default="exp/mae_feat/division_nas5_b2_mae_pretrained_all_days/mae_feat_model_epoch_400_base_48days_dev_input.npy",
                        help='Path to the MAE features file')
    parser.add_argument('--config_file', type=str, default="exp/data/division_nas5_b2_mae_pretrained_48days/dev_config.csv",
                        help='Path to the config CSV file')
    parser.add_argument('--begin_age_days', type=int, default=None,
                        help='Lower bound for age filtering (inclusive)')
    parser.add_argument('--end_age_days', type=int, default=None,
                        help='Upper bound for age filtering (exclusive)')
    parser.add_argument('--tag', type=str, default="b2_dev_day153",
                        help='Tag for naming output files')
    parser.add_argument('--output_root', type=str, default="exp/mae_feat_dim_reduction/python_script",
                        help='Root directory for outputs')
    # Model parameters
    parser.add_argument('--pca_components', type=int, default=50,
                        help='Number of PCA components to use')
    parser.add_argument('--tsne_perplexity', type=float, default=30.0,
                        help='Perplexity parameter for t-SNE')
    parser.add_argument('--tsne_iterations', type=int, default=1000,
                        help='Number of iterations for t-SNE')
    parser.add_argument('--umap_neighbors', type=int, default=15,
                        help='Number of neighbors for UMAP')
    parser.add_argument('--umap_min_dist', type=float, default=0.1,
                        help='Minimum distance parameter for UMAP')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.random_seed)

    # Create output directory
    output_dir = os.path.join(args.output_root, args.tag)
    os.makedirs(output_dir, exist_ok=True)

    # Set up logging
    log_file = os.path.join(output_dir, f"info_{args.tag}.log")
    sys.stdout = Logger(log_file)

    print(f"=== MAE Feature Processing - {args.tag} ===")
    print(f"Started at: {print_timestamp()}")
    print(f"Random seed: {args.random_seed}")

    # ===================== LOAD DATA =====================
    print("\n=== LOADING DATA ===")
    print(f"Loading config file: {args.config_file}")
    config_df = pd.read_csv(args.config_file)
    print(f"Config loaded: {len(config_df)} rows")

    print(f"Loading features file: {args.features_file}")
    start_time = time.time()
    features = np.load(args.features_file)
    print(f"Features loaded in {time.time() - start_time:.2f} seconds")
    print(f"Features shape: {features.shape}")

    # Check for shape mismatch
    if features.shape[0] != len(config_df):
        print(f"WARNING: Features shape ({features.shape[0]}) doesn't match config rows ({len(config_df)})")
        if features.shape[0] > len(config_df):
            print(f"Truncating features to match config rows")
            features = features[:len(config_df)]
        else:
            print(f"Truncating config to match features rows")
            config_df = config_df.iloc[:features.shape[0]]

    # ===================== AGE FILTERING =====================
    print("\n=== AGE FILTERING ===")

    # Determine actual age range bounds - use min/max from data if None
    begin_age_days = args.begin_age_days
    end_age_days = args.end_age_days

    if begin_age_days is None:
        begin_age_days = int(config_df['age_days'].min())
        print(f"Using minimum age from data: {begin_age_days} days")

    if end_age_days is None:
        # Add 1 to max to make range exclusive
        end_age_days = int(config_df['age_days'].max()) + 1
        print(f"Using maximum age from data: {end_age_days-1} days")

    # Create age mask
    print(f"Filtering ages: {begin_age_days} to {end_age_days-1} days")
    age_mask = (config_df['age_days'] >= begin_age_days) & (config_df['age_days'] < end_age_days)

    # Apply filtering
    filtered_config_df = config_df[age_mask]
    filtered_features = features[age_mask]

    print(f"After filtering:")
    print(f"  - Features shape: {filtered_features.shape}")
    print(f"  - Config rows: {len(filtered_config_df)}")
    print(f"  - Kept {len(filtered_config_df) / len(config_df) * 100:.2f}% of original data")

    # Update working variables
    features = filtered_features
    config_df = filtered_config_df

    # ===================== DIMENSIONALITY REDUCTION =====================
    print("\n=== DIMENSIONALITY REDUCTION ===")

    # Standardize features
    print(f"Standardizing features...")
    start_time = time.time()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    print(f"Standardization completed in {time.time() - start_time:.2f} seconds")

    # PCA
    print(f"Applying PCA with {args.pca_components} components...")
    start_time = time.time()
    pca = PCA(n_components=args.pca_components)
    pca_result = pca.fit_transform(features_scaled)
    print(f"PCA completed in {time.time() - start_time:.2f} seconds")
    print(f"Explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")

    # t-SNE
    print(f"Applying t-SNE (perplexity={args.tsne_perplexity}, iterations={args.tsne_iterations})...")
    print(f"Started at: {print_timestamp()}")
    start_time = time.time()
    tsne = TSNE(n_components=2,
                perplexity=args.tsne_perplexity,
                n_iter=args.tsne_iterations,
                random_state=args.random_seed)
    tsne_result = tsne.fit_transform(pca_result)
    print(f"t-SNE completed in {time.time() - start_time:.2f} seconds")
    print(f"Finished at: {print_timestamp()}")

    # UMAP
    print(f"Applying UMAP (neighbors={args.umap_neighbors}, min_dist={args.umap_min_dist})...")
    print(f"Started at: {print_timestamp()}")
    start_time = time.time()
    umap_reducer = umap.UMAP(n_neighbors=args.umap_neighbors,
                            min_dist=args.umap_min_dist,
                            random_state=args.random_seed)
    umap_result = umap_reducer.fit_transform(pca_result)
    print(f"UMAP completed in {time.time() - start_time:.2f} seconds")
    print(f"Finished at: {print_timestamp()}")

    # ===================== SAVE RESULTS =====================
    print("\n=== SAVING RESULTS ===")

    # Define file paths
    mask_file = os.path.join(output_dir, f"mask_{args.tag}.txt")
    feature_file = os.path.join(output_dir, f"mae_feature_{args.tag}.npy")
    config_file_output = os.path.join(output_dir, f"config_{args.tag}.csv")
    tsne_file = os.path.join(output_dir, f"tsne_result_{args.tag}.npy")
    umap_file = os.path.join(output_dir, f"umap_result_{args.tag}.npy")

    # Save mask
    print(f"Saving mask to {mask_file}")
    np.savetxt(mask_file, age_mask.astype(int), fmt='%d')

    # Save filtered features
    print(f"Saving filtered features to {feature_file}")
    np.save(feature_file, features)

    # Save filtered config
    print(f"Saving filtered config to {config_file_output}")
    config_df.to_csv(config_file_output, index=False)

    # Save t-SNE results
    print(f"Saving t-SNE results to {tsne_file}")
    np.save(tsne_file, tsne_result)

    # Save UMAP results
    print(f"Saving UMAP results to {umap_file}")
    np.save(umap_file, umap_result)

    # ===================== SUMMARY =====================
    print("\n=== SUMMARY ===")
    print(f"Features: {features.shape}")
    print(f"Config: {len(config_df)} rows")
    print(f"Age range: {begin_age_days} to {end_age_days-1} days")
    print(f"PCA components: {args.pca_components}")
    print(f"t-SNE result: {tsne_result.shape}")
    print(f"UMAP result: {umap_result.shape}")
    print(f"All results saved to: {output_dir}")
    print(f"Completed at: {print_timestamp()}")

    # Reset stdout
    sys.stdout = sys.stdout.terminal
    print(f"Processing complete. Results saved to {output_dir}")
    print(f"Log saved to {log_file}")

if __name__ == "__main__":
    main()

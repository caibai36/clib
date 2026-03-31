#!/usr/bin/env python3
"""
MAE Feature to PCA50 Conversion Script

This script loads existing MAE feature .npy files and applies PCA dimension reduction
to 50 components using the exact same method as the original scripts.

Usage:
    python mae_raw2pca.py --input_file path/to/mae_feature.npy
    python mae_raw2pca.py --input_file path/to/mae_feature.npy --pca_components 50

Author: Based on work by bin-wu
Date: January 16, 2026
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description='Convert MAE features to PCA reduced dimensions.')
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to the MAE features .npy file')
    parser.add_argument('--pca_components', type=int, default=50,
                        help='Number of PCA components (default: 50)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output file path (default: same directory as input with mae_pca{N}.npy)')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.random_seed)

    # Determine output file path
    if args.output_file is None:
        input_dir = os.path.dirname(args.input_file)
        output_file = os.path.join(input_dir, f"mae_pca{args.pca_components}.npy")
    else:
        output_file = args.output_file

    print(f"=== MAE to PCA{args.pca_components} Conversion ===")
    print(f"Input file: {args.input_file}")
    print(f"Output file: {output_file}")
    print(f"Random seed: {args.random_seed}")

    # Load features
    print(f"\nLoading features...")
    start_time = time.time()
    features = np.load(args.input_file)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    print(f"Features shape: {features.shape}")

    # Check for NaN values
    print(f"\nChecking for NaN values...")
    nan_count = np.isnan(features).sum()
    if nan_count > 0:
        print(f"WARNING: Found {nan_count} NaN values")
        nan_mask = ~np.isnan(features).any(axis=1)
        samples_with_nan = (~nan_mask).sum()
        print(f"Filtering out {samples_with_nan} samples containing NaN values")
        features = features[nan_mask]
        print(f"After filtering: {features.shape}")
    else:
        print(f"No NaN values found")

    # Standardize features (same as original scripts)
    print(f"\nStandardizing features...")
    start_time = time.time()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    print(f"Standardization completed in {time.time() - start_time:.2f} seconds")

    # Apply PCA (same as original scripts)
    print(f"\nApplying PCA with {args.pca_components} components...")
    start_time = time.time()
    pca = PCA(n_components=args.pca_components, random_state=args.random_seed)
    pca_result = pca.fit_transform(features_scaled)
    print(f"PCA completed in {time.time() - start_time:.2f} seconds")
    print(f"PCA result shape: {pca_result.shape}")
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")

    # Save results
    print(f"\nSaving PCA results to {output_file}...")
    np.save(output_file, pca_result)
    print(f"Saved successfully!")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Input: {features.shape} -> Output: {pca_result.shape}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()

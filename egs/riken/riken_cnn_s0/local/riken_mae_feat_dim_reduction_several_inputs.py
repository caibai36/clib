#!/usr/bin/env python3
"""
MAE Feature Dimensionality Reduction Script for Multiple Inputs with Shared Embedding Space

This script loads multiple MAE features, applies age filtering to each,
merges them into a single dataset, performs dimensionality reduction,
then splits results back to individual datasets for comparison.

All files are saved in the same base directory with different tags.

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

def load_and_filter_dataset(features_file, config_file, subtag, begin_age_days, end_age_days):
    """Load and filter a single dataset, return filtered data and metadata."""
    
    print(f"\n--- Processing {subtag} ---")
    
    # Load data
    print(f"Loading config: {config_file}")
    config_df = pd.read_csv(config_file)
    print(f"Config loaded: {len(config_df)} rows")

    print(f"Loading features: {features_file}")
    start_time = time.time()
    features = np.load(features_file)
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

    # Determine age bounds
    actual_begin = begin_age_days if begin_age_days is not None else int(config_df['age_days'].min())
    actual_end = end_age_days if end_age_days is not None else int(config_df['age_days'].max()) + 1

    # Apply age filtering
    print(f"Filtering ages: {actual_begin} to {actual_end-1} days")
    age_mask = (config_df['age_days'] >= actual_begin) & (config_df['age_days'] < actual_end)
    
    filtered_config = config_df[age_mask].copy()
    filtered_features = features[age_mask]
    
    # Add dataset identifier to config
    filtered_config['dataset'] = subtag
    filtered_config['original_index'] = config_df.index[age_mask].values
    
    print(f"After filtering: {filtered_features.shape[0]} samples ({len(filtered_config)/len(config_df)*100:.2f}% kept)")
    
    return {
        'subtag': subtag,
        'features': filtered_features,
        'config': filtered_config,
        'mask': age_mask,
        'original_shape': features.shape,
        'original_config_len': len(config_df)
    }

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process multiple MAE features with shared embedding space.')
    parser.add_argument('--features_file', type=str, nargs='+', required=True,
                        help='Paths to the MAE features files')
    parser.add_argument('--config_file', type=str, nargs='+', required=True,
                        help='Paths to the config CSV files')
    parser.add_argument('--subtag', type=str, nargs='+', required=True,
                        help='Subtags for each dataset (e.g., b1_f1, b2_f1, etc.)')
    parser.add_argument('--begin_age_days', type=int, default=None,
                        help='Lower bound for age filtering (inclusive)')
    parser.add_argument('--end_age_days', type=int, default=None,
                        help='Upper bound for age filtering (exclusive)')
    parser.add_argument('--tag', type=str, default="bx_fx_15weeks",
                        help='Main tag for naming output directories and files')
    parser.add_argument('--output_root', type=str, default="exp/mae_feat_dim_reduction/python_script_nas5",
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

    # Validate input lengths
    if len(args.features_file) != len(args.config_file) or len(args.features_file) != len(args.subtag):
        print("Error: Number of features files, config files, and subtags must be equal")
        sys.exit(1)

    # Set random seed for reproducibility
    np.random.seed(args.random_seed)

    # Create base output directory
    # Base directory: args.output_root/tag/ (all files saved here)
    base_dir = os.path.join(args.output_root, args.tag)
    os.makedirs(base_dir, exist_ok=True)

    # Set up logging in the base directory
    log_file = os.path.join(base_dir, f"info_merged_{args.tag}.log")
    sys.stdout = Logger(log_file)

    print(f"=== MAE Feature Processing with Shared Embedding Space - {args.tag} ===")
    print(f"Started at: {print_timestamp()}")
    print(f"Random seed: {args.random_seed}")
    print(f"Number of datasets: {len(args.features_file)}")
    print(f"Subtags: {args.subtag}")
    print(f"Base output directory: {base_dir}")

    # ===================== STEP 1: LOAD AND FILTER EACH DATASET =====================
    print(f"\n{'='*80}")
    print("STEP 1: LOADING AND FILTERING INDIVIDUAL DATASETS")
    print(f"{'='*80}")
    
    datasets = []
    for features_file, config_file, subtag in zip(args.features_file, args.config_file, args.subtag):
        try:
            dataset = load_and_filter_dataset(
                features_file, config_file, subtag, 
                args.begin_age_days, args.end_age_days
            )
            datasets.append(dataset)
        except Exception as e:
            print(f"ERROR loading {subtag}: {str(e)}")
            sys.exit(1)

    # ===================== STEP 2: MERGE ALL FILTERED DATASETS =====================
    print(f"\n{'='*80}")
    print("STEP 2: MERGING ALL FILTERED DATASETS")
    print(f"{'='*80}")
    
    # Combine features and configs
    all_features = []
    all_configs = []
    dataset_boundaries = []  # Track where each dataset starts/ends in merged data
    
    current_idx = 0
    for dataset in datasets:
        start_idx = current_idx
        end_idx = current_idx + dataset['features'].shape[0]
        dataset_boundaries.append({
            'subtag': dataset['subtag'],
            'ind_tag': f"{args.tag}_{dataset['subtag']}",  # individual tag
            'start_idx': start_idx,
            'end_idx': end_idx,
            'count': dataset['features'].shape[0]
        })
        
        all_features.append(dataset['features'])
        all_configs.append(dataset['config'])
        current_idx = end_idx

    # Merge
    merged_features = np.vstack(all_features)
    merged_config = pd.concat(all_configs, ignore_index=True)
    
    print(f"Merged features shape: {merged_features.shape}")
    print(f"Merged config length: {len(merged_config)}")
    
    # Print dataset boundaries
    print("\nDataset boundaries in merged data:")
    for boundary in dataset_boundaries:
        print(f"  {boundary['subtag']} (ind_tag: {boundary['ind_tag']}): indices {boundary['start_idx']}:{boundary['end_idx']} ({boundary['count']} samples)")

    # ===================== STEP 3: DIMENSIONALITY REDUCTION ON MERGED DATA =====================
    print(f"\n{'='*80}")
    print("STEP 3: DIMENSIONALITY REDUCTION ON MERGED DATA")
    print(f"{'='*80}")

    # Standardize features
    print(f"Standardizing merged features...")
    start_time = time.time()
    scaler = StandardScaler()
    merged_features_scaled = scaler.fit_transform(merged_features)
    print(f"Standardization completed in {time.time() - start_time:.2f} seconds")

    # PCA
    print(f"Applying PCA with {args.pca_components} components...")
    start_time = time.time()
    pca = PCA(n_components=args.pca_components)
    merged_pca_result = pca.fit_transform(merged_features_scaled)
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
    merged_tsne_result = tsne.fit_transform(merged_pca_result)
    print(f"t-SNE completed in {time.time() - start_time:.2f} seconds")
    print(f"Finished at: {print_timestamp()}")

    # UMAP
    print(f"Applying UMAP (neighbors={args.umap_neighbors}, min_dist={args.umap_min_dist})...")
    print(f"Started at: {print_timestamp()}")
    start_time = time.time()
    umap_reducer = umap.UMAP(n_neighbors=args.umap_neighbors,
                            min_dist=args.umap_min_dist,
                            random_state=args.random_seed)
    merged_umap_result = umap_reducer.fit_transform(merged_pca_result)
    print(f"UMAP completed in {time.time() - start_time:.2f} seconds")
    print(f"Finished at: {print_timestamp()}")

    # ===================== STEP 4: SAVE MERGED RESULTS =====================
    print(f"\n{'='*80}")
    print("STEP 4: SAVING MERGED RESULTS")
    print(f"{'='*80}")
    
    # Save merged data in base directory
    np.save(os.path.join(base_dir, f"merged_features_{args.tag}.npy"), merged_features)
    np.save(os.path.join(base_dir, f"merged_pca_result_{args.tag}.npy"), merged_pca_result)
    np.save(os.path.join(base_dir, f"merged_tsne_result_{args.tag}.npy"), merged_tsne_result)
    np.save(os.path.join(base_dir, f"merged_umap_result_{args.tag}.npy"), merged_umap_result)
    merged_config.to_csv(os.path.join(base_dir, f"merged_config_{args.tag}.csv"), index=False)
    
    # Save dataset boundaries
    boundaries_df = pd.DataFrame(dataset_boundaries)
    boundaries_df.to_csv(os.path.join(base_dir, f"dataset_boundaries_{args.tag}.csv"), index=False)
    
    print(f"Merged results saved to: {base_dir}")

    # ===================== STEP 5: SAVE INDIVIDUAL RESULTS =====================
    print(f"\n{'='*80}")
    print("STEP 5: SAVING INDIVIDUAL RESULTS")
    print(f"{'='*80}")

    for i, boundary in enumerate(dataset_boundaries):
        subtag = boundary['subtag']
        ind_tag = boundary['ind_tag']  # individual tag: tag_subtag
        start_idx = boundary['start_idx']
        end_idx = boundary['end_idx']
        
        print(f"\nProcessing {subtag} (ind_tag: {ind_tag}, indices {start_idx}:{end_idx})")
        
        # Extract results for this dataset
        dataset_features = merged_features[start_idx:end_idx]
        dataset_config = merged_config.iloc[start_idx:end_idx].copy()
        dataset_pca = merged_pca_result[start_idx:end_idx]
        dataset_tsne = merged_tsne_result[start_idx:end_idx]
        dataset_umap = merged_umap_result[start_idx:end_idx]
        
        # Reset config index
        dataset_config.reset_index(drop=True, inplace=True)
        
        # Save individual results directly in base directory with ind_tag
        mask_file = os.path.join(base_dir, f"mask_{ind_tag}.txt")
        feature_file = os.path.join(base_dir, f"mae_feature_{ind_tag}.npy")
        config_file_output = os.path.join(base_dir, f"config_{ind_tag}.csv")
        pca_file = os.path.join(base_dir, f"pca_result_{ind_tag}.npy")
        tsne_file = os.path.join(base_dir, f"tsne_result_{ind_tag}.npy")
        umap_file = os.path.join(base_dir, f"umap_result_{ind_tag}.npy")
        
        # Save mask (from original dataset info)
        original_dataset = datasets[i]
        np.savetxt(mask_file, original_dataset['mask'].astype(int), fmt='%d')
        
        # Save other files
        np.save(feature_file, dataset_features)
        dataset_config.to_csv(config_file_output, index=False)
        np.save(pca_file, dataset_pca)
        np.save(tsne_file, dataset_tsne)
        np.save(umap_file, dataset_umap)
        
        print(f"  Files saved with ind_tag: {ind_tag}")
        print(f"  Features: {dataset_features.shape}")
        print(f"  t-SNE: {dataset_tsne.shape}")
        print(f"  UMAP: {dataset_umap.shape}")

    # ===================== FINAL SUMMARY =====================
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    print(f"Total datasets processed: {len(datasets)}")
    print(f"Age range: {args.begin_age_days} to {args.end_age_days-1} days")
    print(f"Merged dataset shape: {merged_features.shape}")
    print(f"PCA components: {args.pca_components}")
    print(f"t-SNE parameters: perplexity={args.tsne_perplexity}, iterations={args.tsne_iterations}")
    print(f"UMAP parameters: neighbors={args.umap_neighbors}, min_dist={args.umap_min_dist}")
    
    print(f"\nDataset breakdown:")
    for boundary in dataset_boundaries:
        print(f"  {boundary['subtag']} (ind_tag: {boundary['ind_tag']}): {boundary['count']} samples")
    
    print(f"\nAll files saved in base directory: {base_dir}")
    print(f"\nFile naming convention:")
    print(f"  Merged files: *_{args.tag}.* (e.g., merged_tsne_result_{args.tag}.npy)")
    print(f"  Individual files: *_{{ind_tag}}.* (e.g., mae_feature_{args.tag}_b1_f1.npy)")
    
    # Usage instructions
    print(f"\n{'='*80}")
    print("USAGE INSTRUCTIONS:")
    print(f"{'='*80}")
    print("# Load merged results:")
    print(f'tag = "{args.tag}"')
    print(f'base_dir = "{base_dir}"')
    print(f"merged_tsne = np.load(f\"{{base_dir}}/merged_tsne_result_{{tag}}.npy\")")
    print(f"merged_config = pd.read_csv(f\"{{base_dir}}/merged_config_{{tag}}.csv\")")
    print(f"boundaries = pd.read_csv(f\"{{base_dir}}/dataset_boundaries_{{tag}}.csv\")")
    print("")
    print("# Load individual results:")
    for boundary in dataset_boundaries:
        subtag = boundary['subtag']
        ind_tag = boundary['ind_tag']
        print(f"# {subtag}")
        print(f'tag = "{args.tag}"')
        print(f'subtag = "{subtag}"') 
        print(f'ind_tag = "{ind_tag}"  # {args.tag}_{subtag}')
        print(f'base_dir = "{base_dir}"')
        print(f"features = np.load(f\"{{base_dir}}/mae_feature_{{ind_tag}}.npy\")")
        print(f"config_df = pd.read_csv(f\"{{base_dir}}/config_{{ind_tag}}.csv\")")
        print(f"tsne_result = np.load(f\"{{base_dir}}/tsne_result_{{ind_tag}}.npy\")")
        print(f"umap_result = np.load(f\"{{base_dir}}/umap_result_{{ind_tag}}.npy\")")
        print("")

    print(f"\nCompleted at: {print_timestamp()}")

    # Reset stdout
    sys.stdout = sys.stdout.terminal
    print(f"Processing complete. Results saved to {base_dir}")
    print(f"Log saved to {log_file}")

if __name__ == "__main__":
    main()

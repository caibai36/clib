#!/usr/bin/env python3
"""
Simplified Stage Clustering Analysis for NTT Infant Data

Command-line tool for hierarchical clustering analysis with:
- Calinski-Harabasz index computation
- Dendrogram and stage sequence visualization
- Multiple feature modes: count_ratio, mae_tsne, count_ratio+mae_tsne

Usage:
    python local/ntt_clustering_analysis.py \
        --info_mae_csv /path/to/info_mae.csv \
        --output_dir /path/to/output \
        --data_name NTT_ma \
        --n_stages 3 \
        --feature_mode count_ratio+mae_tsne \
        --max_clusters 10
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score
import os
import argparse
from pathlib import Path

# Set style consistent with vis.py
sns.set_style('white')
sns.set_context('paper')


def create_combined_features(config_df, label_col='label', feature_mode='count_ratio+mae_tsne', 
                             age_unit='age_months', topk=3):
    """
    Create combined features based on feature_mode
    
    Parameters:
    -----------
    config_df : pd.DataFrame
        Input dataframe with labels and t-SNE coordinates
    label_col : str
        Column name for labels (default: 'label')
    feature_mode : str
        One of 'count_ratio', 'mae_tsne', 'count_ratio+mae_tsne'
    age_unit : str
        Age column name (default: 'age_months')
    topk : int
        Number of top call types to use (default: 3 for crying/fussing/laughter)
    """
    valid_modes = ["count_ratio", "mae_tsne", "count_ratio+mae_tsne"]
    if feature_mode not in valid_modes:
        raise ValueError(f"feature_mode must be one of {valid_modes}")
    
    # Define the key phone categories
    target_phones = ['<crying>', '<fussing>', '<laughter>']
    phone_names = ['crying', 'fussing', 'laughter']
    
    unique_ages = sorted(config_df[age_unit].unique())
    print(f"Processing {len(unique_ages)} age points from {len(config_df)} samples")
    print(f"Feature mode: {feature_mode}")
    
    combined_data = []
    
    for age in unique_ages:
        age_data = config_df[config_df[age_unit] == age]
        
        # Count each category
        counts = {}
        for phone in target_phones:
            counts[phone] = len(age_data[age_data[label_col] == phone])
        
        target_total = sum(counts.values())
        
        feature_vector = []
        
        # Add count ratios if needed
        if feature_mode in ['count_ratio', 'count_ratio+mae_tsne']:
            if target_total > 0:
                for phone in target_phones:
                    feature_vector.append(counts[phone] / target_total)
            else:
                feature_vector.extend([0.0] * len(target_phones))
        
        # Add t-SNE features if needed
        if feature_mode in ['mae_tsne', 'count_ratio+mae_tsne']:
            age_mask = config_df[age_unit] == age
            age_labels_data = config_df.loc[age_mask, label_col]
            
            for phone in target_phones:
                phone_mask = age_labels_data == phone
                if np.any(phone_mask):
                    # Get t-SNE coordinates for this phone at this age
                    phone_indices = config_df[age_mask].index[phone_mask]
                    tsne_x = config_df.loc[phone_indices, 'tsne_1'].mean()
                    tsne_y = config_df.loc[phone_indices, 'tsne_2'].mean()
                    feature_vector.extend([tsne_x, tsne_y])
                else:
                    feature_vector.extend([0.0, 0.0])
        
        # Add metadata
        feature_vector.extend([counts[phone] for phone in target_phones])
        feature_vector.extend([target_total, len(age_data), age])
        
        combined_data.append(feature_vector)
    
    # Build column names
    feature_columns = []
    if feature_mode in ['count_ratio', 'count_ratio+mae_tsne']:
        for name in phone_names:
            feature_columns.append(f'{name}_ratio')
    if feature_mode in ['mae_tsne', 'count_ratio+mae_tsne']:
        for name in phone_names:
            feature_columns.extend([f'{name}_tsne_x', f'{name}_tsne_y'])
    
    # Metadata columns
    for name in phone_names:
        feature_columns.append(f'{name}_count')
    feature_columns.extend(['target_total', 'sample_count', age_unit])
    
    combined_df = pd.DataFrame(combined_data, columns=feature_columns)
    
    print(f"Created feature matrix: {combined_df.shape}")
    print(f"Crying: {combined_df['crying_count'].sum():.0f}, "
          f"Fussing: {combined_df['fussing_count'].sum():.0f}, "
          f"Laughter: {combined_df['laughter_count'].sum():.0f}")
    
    return combined_df


def get_feature_columns(feature_mode):
    """Get the list of feature columns based on feature_mode"""
    phone_names = ['crying', 'fussing', 'laughter']
    feature_cols = []
    
    if feature_mode in ['count_ratio', 'count_ratio+mae_tsne']:
        for name in phone_names:
            feature_cols.append(f'{name}_ratio')
    
    if feature_mode in ['mae_tsne', 'count_ratio+mae_tsne']:
        for name in phone_names:
            feature_cols.extend([f'{name}_tsne_x', f'{name}_tsne_y'])
    
    return feature_cols


def compute_ch_scores(combined_df, feature_mode='count_ratio+mae_tsne', max_clusters=8):
    """
    Compute Calinski-Harabasz scores for different cluster numbers
    """
    feature_cols = get_feature_columns(feature_mode)
    
    X = combined_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    linkage_matrix = linkage(X_scaled, method='ward', metric='euclidean')
    
    cluster_range = range(2, min(max_clusters + 1, len(X_scaled)))
    ch_scores = []
    n_clusters_list = []
    
    for n_clusters in cluster_range:
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        if len(np.unique(cluster_labels)) == n_clusters:
            ch_score = calinski_harabasz_score(X_scaled, cluster_labels)
            ch_scores.append(ch_score)
            n_clusters_list.append(n_clusters)
    
    return n_clusters_list, ch_scores, linkage_matrix, X_scaled


def create_stage_visualization(combined_df, n_stages=3, data_name='NTT', 
                               feature_mode='count_ratio+mae_tsne',
                               age_unit='age_months',
                               sort_stage='avg_age',
                               figsize=(20, 6), output_dir=None):
    """
    Create 2-plot visualization: dendrogram and stage sequence
    Style consistent with vis.py
    """
    print(f"\n{'='*60}")
    print(f"CREATING {n_stages}-STAGE VISUALIZATION")
    print(f"{'='*60}")
    
    # Prepare features and clustering
    feature_cols = get_feature_columns(feature_mode)
    
    X = combined_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    linkage_matrix = linkage(X_scaled, method='ward', metric='euclidean')
    
    # Cut dendrogram
    cluster_labels = fcluster(linkage_matrix, n_stages, criterion='maxclust')
    combined_df['n_stages'] = cluster_labels
    
    # Analyze stages by average age
    stage_info = {}
    for stage in range(1, n_stages + 1):
        stage_data = combined_df[combined_df['n_stages'] == stage]
        if len(stage_data) == 0:
            continue
        
        ages = sorted(stage_data[age_unit].tolist())
        stage_info[stage] = {
            'avg_age': stage_data[age_unit].mean(),
            'min_age': stage_data[age_unit].min(),
            'max_age': stage_data[age_unit].max(),
            'ages': ages,
            'count': len(stage_data)
        }
    
    # Sort by specified criterion (default: avg_age)
    sorted_stages = sorted(stage_info.items(), key=lambda x: x[1][sort_stage])
    
    # Stage names and colors (consistent with vis.py)
    dev_colors = plt.cm.Set3(np.linspace(0, 1, n_stages))
    
    if n_stages == 2:
        dev_names = ['Early Stage', 'Late Stage']
    elif n_stages == 3:
        dev_names = ['Early Stage', 'Middle Stage', 'Late Stage']
    elif n_stages == 4:
        dev_names = ['Early Stage', 'Early-Middle Stage', 'Late-Middle Stage', 'Late Stage']
    else:
        dev_names = [f'Stage {i+1}' for i in range(n_stages)]
    
    # Map to developmental stages
    stage_mapping = {}
    for i, (original_stage, info) in enumerate(sorted_stages):
        stage_mapping[original_stage] = i + 1
        info['dev_stage'] = i + 1
        info['color'] = dev_colors[i]
        info['name'] = dev_names[i]
    
    combined_df['dev_stage'] = combined_df['n_stages'].map(stage_mapping)
    
    # Create 2-plot figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Dendrogram
    ax1 = axes[0]
    dendrogram(linkage_matrix, 
               labels=combined_df[age_unit].values,
               ax=ax1,
               leaf_rotation=45,
               leaf_font_size=8)
    
    # Calculate and draw cut line (matching vis.py logic)
    distances = linkage_matrix[:, 2]
    sorted_distances = np.sort(distances)
    if len(sorted_distances) >= n_stages:
        cut_height = sorted_distances[-(n_stages-1)] + 0.1
    else:
        cut_height = sorted_distances[-1] * 0.7
    
    ax1.axhline(y=cut_height, color='red', linestyle='--', linewidth=3, 
                label=f'{n_stages}-Stage Cut', zorder=10)
    
    # Title with feature mode info (matching vis.py style)
    title_suffix = f"\n{feature_mode.replace('_', ' ').title()}"
    dendrogram_title = f'Hierarchical Clustering - {n_stages} Developmental Stages ({data_name}){title_suffix}'
    ax1.set_title(dendrogram_title, fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'{age_unit}', fontsize=12)
    ax1.set_ylabel('Distance', fontsize=12)
    ax1.legend(fontsize=11)
    
    # Plot 2: Stage sequence
    ax2 = axes[1]
    for i, (original_stage, info) in enumerate(sorted_stages):
        stage_ages = info['ages']
        stage_nums = [info['dev_stage']] * len(stage_ages)
        ax2.scatter(stage_ages, stage_nums, 
                   s=80, alpha=0.7, color=info['color'], 
                   label=f"{info['name']} (n={info['count']})")
    
    # Title with feature mode info (matching vis.py style)
    sequence_title = f'{n_stages}-Stage Developmental Sequence ({data_name}){title_suffix}'
    ax2.set_title(sequence_title, fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'{age_unit}', fontsize=12)
    ax2.set_ylabel('Developmental Stage', fontsize=12)
    ax2.set_yticks(range(1, n_stages + 1))
    # Match vis.py format: "Stage Name (number)"
    ax2.set_yticklabels([f'{dev_names[i]} ({i+1})' for i in range(n_stages)])
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.92)
    
    if output_dir:
        output_path = os.path.join(output_dir, f'clustering_{feature_mode}_{n_stages}_stages.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved clustering plot to: {output_path}")
        plt.close()
    else:
        plt.show()
    
    return combined_df, sorted_stages


def print_stage_summary(combined_df, sorted_stages, age_unit='age_months'):
    """
    Print summary of stages (matching vis.py format)
    """
    n_stages = len(sorted_stages)
    print(f"\n{'='*80}")
    print(f"CLUSTERING RESULTS")
    print(f"{'='*80}")
    
    for i, (original_stage, info) in enumerate(sorted_stages):
        dev_stage = info['dev_stage']
        stage_data = combined_df[combined_df['dev_stage'] == dev_stage]
        
        print(f"\n{info['name']} (Stage {dev_stage}):")
        print(f"  Age range: {info['min_age']}-{info['max_age']} {age_unit}")
        print(f"  Number of time points: {info['count']}")
        print(f"  {age_unit.replace('_', ' ').title()}: {info['ages']}")
        
        # Print ratios if available
        if 'crying_ratio' in combined_df.columns:
            print(f"  Crying: {stage_data['crying_ratio'].mean():.3f}, "
                  f"Fussing: {stage_data['fussing_ratio'].mean():.3f}, "
                  f"Laughter: {stage_data['laughter_ratio'].mean():.3f}")


def main():
    parser = argparse.ArgumentParser(
        description='Simplified Stage Clustering Analysis for NTT Infant Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default feature mode
    python local/ntt_clustering_analysis.py \\
        --info_mae_csv /path/to/info_mae.csv \\
        --output_dir /path/to/output \\
        --data_name NTT_ma \\
        --n_stages 3

    # Using only count ratios
    python local/ntt_clustering_analysis.py \\
        --info_mae_csv /path/to/info_mae.csv \\
        --output_dir /path/to/output \\
        --data_name NTT_ma \\
        --n_stages 3 \\
        --feature_mode count_ratio

    # Using only MAE t-SNE features
    python local/ntt_clustering_analysis.py \\
        --info_mae_csv /path/to/info_mae.csv \\
        --output_dir /path/to/output \\
        --data_name NTT_ma \\
        --n_stages 3 \\
        --feature_mode mae_tsne
        """
    )

    # Required arguments
    parser.add_argument('--info_mae_csv', type=str, required=True,
                        help='Path to info_mae.csv file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for saving figures')
    parser.add_argument('--data_name', type=str, required=True,
                        help='Name of the dataset (e.g., NTT_ma)')

    # Clustering options
    parser.add_argument('--n_stages', type=int, default=3,
                        help='Number of developmental stages (default: 3)')
    parser.add_argument('--feature_mode', type=str, default='count_ratio+mae_tsne',
                        choices=['count_ratio', 'mae_tsne', 'count_ratio+mae_tsne'],
                        help='Feature mode for clustering (default: count_ratio+mae_tsne)')
    parser.add_argument('--max_clusters', type=int, default=10,
                        help='Maximum clusters for CH index (default: 10)')
    parser.add_argument('--age_unit', type=str, default='age_months',
                        help='Age column name (default: age_months)')
    parser.add_argument('--sort_stage', type=str, default='avg_age',
                        choices=['avg_age', 'min_age', 'max_age'],
                        help='Criterion for sorting developmental stages (default: avg_age)')
    parser.add_argument('--label_col', type=str, default=None,
                        help='Column name for labels (default: auto-detect "label" or "phone")')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("SIMPLIFIED STAGE CLUSTERING ANALYSIS")
    print("=" * 80)
    print(f"Dataset: {args.data_name}")
    print(f"Input CSV: {args.info_mae_csv}")
    print(f"Output directory: {args.output_dir}")
    print(f"Feature mode: {args.feature_mode}")
    print(f"Number of stages: {args.n_stages}")

    # Load data
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    config_df = pd.read_csv(args.info_mae_csv)
    print(f"Loaded {len(config_df)} samples")
    print(f"Columns: {list(config_df.columns)}")

    # Detect label column
    if args.label_col:
        label_col = args.label_col
    else:
        label_col = 'label' if 'label' in config_df.columns else 'phone'
    print(f"Using '{label_col}' column for labels")

    # Check required columns
    required_cols = [args.age_unit]
    if args.feature_mode in ['mae_tsne', 'count_ratio+mae_tsne']:
        required_cols.extend(['tsne_1', 'tsne_2'])
    
    missing_cols = [col for col in required_cols if col not in config_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"\nUnique {args.age_unit}: {sorted(config_df[args.age_unit].unique())}")

    # Create combined features
    print("\n" + "=" * 80)
    print("CREATING FEATURES")
    print("=" * 80)
    combined_df = create_combined_features(config_df, label_col=label_col, 
                                          feature_mode=args.feature_mode,
                                          age_unit=args.age_unit)

    # Compute CH scores
    print("\n" + "=" * 80)
    print("COMPUTING CALINSKI-HARABASZ SCORES")
    print("=" * 80)
    n_clusters_list, ch_scores, _, _ = compute_ch_scores(combined_df, 
                                                          feature_mode=args.feature_mode,
                                                          max_clusters=args.max_clusters)

    # Find optimal k
    optimal_k = n_clusters_list[np.argmax(ch_scores)]
    optimal_score = max(ch_scores)

    print(f"\nOptimal number of clusters (by CH index): {optimal_k}")
    print(f"CH score at optimal k: {optimal_score:.1f}")

    # Plot CH scores (matching vis.py style exactly)
    plt.figure(figsize=(10, 6))
    plt.plot(n_clusters_list, ch_scores, 'o-', color='green', linewidth=2, markersize=8)
    plt.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2,
               label=f'Optimal k={optimal_k}')

    if args.n_stages != optimal_k:
        plt.axvline(x=args.n_stages, color='blue', linestyle=':', linewidth=2,
                   label=f'Selected k={args.n_stages}')

    plt.text(optimal_k, optimal_score, f'  CH={optimal_score:.1f}',
            fontsize=10, va='bottom')

    # Title matching vis.py format: three lines with feature mode
    ch_title = f'Calinski-Harabasz Index (Higher = Better)\n{args.data_name}\n{args.feature_mode.replace("_", " ").title()}'
    plt.title(ch_title, fontsize=14, fontweight='bold')
    plt.xlabel('Number of Clusters', fontsize=12)
    plt.ylabel('CH Index', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()

    ch_output_path = os.path.join(args.output_dir, f'ch_index_{args.feature_mode}.png')
    plt.savefig(ch_output_path, dpi=150, bbox_inches='tight')
    print(f"Saved CH index plot to: {ch_output_path}")
    plt.close()

    print(f"\nCH Scores:")
    for n, score in zip(n_clusters_list, ch_scores):
        marker = " <-- OPTIMAL" if n == optimal_k else (" <-- SELECTED" if n == args.n_stages else "")
        print(f"  {n} clusters: {score:.2f}{marker}")

    # Create stage visualization
    print("\n" + "=" * 80)
    print(f"DEVELOPMENTAL STAGE ANALYSIS ({args.feature_mode.upper()})")
    print("=" * 80)
    combined_df_with_stages, sorted_stages = create_stage_visualization(
        combined_df, 
        n_stages=args.n_stages,
        data_name=args.data_name,
        feature_mode=args.feature_mode,
        age_unit=args.age_unit,
        sort_stage=args.sort_stage,
        output_dir=args.output_dir
    )

    # Print summary
    print_stage_summary(combined_df_with_stages, sorted_stages, age_unit=args.age_unit)

    # Save results
    output_csv = os.path.join(args.output_dir, 
                             f'clustering_results_{args.feature_mode}_{args.n_stages}_stages.csv')
    combined_df_with_stages.to_csv(output_csv, index=False)
    print(f"\n✓ Results saved to: {output_csv}")

    print("\n" + "=" * 80)
    print("ALL VISUALIZATIONS COMPLETE")
    print("=" * 80)
    print(f"\nOutput files in: {args.output_dir}")
    print(f"  - {ch_output_path}")
    print(f"  - {os.path.join(args.output_dir, f'clustering_{args.feature_mode}_{args.n_stages}_stages.png')}")
    print(f"  - {output_csv}")


if __name__ == "__main__":
    main()

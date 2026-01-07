#!/usr/bin/env python3
"""
Comprehensive MAE t-SNE Visualization for Marmoset Dataset

This script creates multiple visualization plots and saves them to the output directory:
1. t-SNE projections by age attribute
2. Calinski-Harabasz index analysis
3. Hierarchical clustering dendrogram and developmental stages

Usage:
    python local/visualize_mae_comprehensive.py \
        --info_mae_csv /path/to/info_mae.csv \
        --output_dir /path/to/output \
        --data_name b3_762F_763M_3201M \
        --label_option 2 \
        --attribute age_days \
        --n_cols 14 \
        --topk 5 \
        --n_stages 3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path
from typing import Tuple, Optional
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score

# Set style for plots
sns.set_style('white')
sns.set_context('paper')


def load_and_process_data(
    csv_path: str,
    label_option: int = 2,
    verbose: bool = True
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load data from CSV and process labels according to the specified option.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing the data
    label_option : int, default=2
        Label preprocessing option:
        1 - Keep original labels (including u-X labels)
        2 - Merge u-X labels with their corresponding X labels
        3 - Remove samples with u-X labels
    verbose : bool, default=True
        Whether to print processing information

    Returns:
    --------
    config_df : pd.DataFrame
        Processed dataframe with labels
    tsne_result : np.ndarray
        t-SNE coordinates extracted from the dataframe
    """

    # Load data
    config_df = pd.read_csv(csv_path)

    if verbose:
        print(f"Loaded data shape: {config_df.shape}")
        print(f"\nOriginal label distribution:")
        print(config_df['label'].value_counts())

    # Extract t-SNE results
    tsne_result = config_df[['tsne_1', 'tsne_2']].values

    if verbose:
        print(f"t-SNE shape: {tsne_result.shape}")

    # Process labels based on the selected option
    if label_option == 1:
        if verbose:
            print("\n[Option 1] Keeping original labels including uncertain (u-X) labels")

    elif label_option == 2:
        if verbose:
            print("\n[Option 2] Merging uncertain (u-X) labels with their corresponding certain labels")

        # Dictionary to map uncertain labels to certain labels
        uncertain_label_map = {
            'u-pp': 'pp', 'u-ct': 'ct', 'u-cr': 'cr', 'u-cp': 'cp',
            'u-ek': 'ek', 'u-ph': 'ph', 'u-tr': 'tr', 'u-ts': 'ts',
            'u-se': 'se', 'u-ok': 'ok', 'u-tw': 'tw'
        }

        config_df['label'] = config_df['label'].replace(uncertain_label_map)

        if verbose:
            print("\nLabel distribution after merging:")
            print(config_df['label'].value_counts())

    elif label_option == 3:
        if verbose:
            print("\n[Option 3] Removing samples with uncertain (u-X) labels")

        uncertain_mask = config_df['label'].str.startswith('u-')
        uncertain_count = uncertain_mask.sum()

        if verbose:
            print(f"Removing {uncertain_count} samples with uncertain labels")

        config_df = config_df[~uncertain_mask].reset_index(drop=True)
        tsne_result = tsne_result[~uncertain_mask.values]

        if verbose:
            print("\nLabel distribution after removing uncertain labels:")
            print(config_df['label'].value_counts())

    else:
        raise ValueError(f"Invalid label_option: {label_option}. Must be 1, 2, or 3.")

    return config_df, tsne_result


def visualize_labels_by_attribute_compact(tsne_result, config_df, attribute='age_weeks',
                                         values_to_show=None, n_cols=4, data_name='dataset',
                                         output_path=None):
    """
    Create compact subfigures of t-SNE plots for any attribute.
    """
    all_values = sorted(config_df[attribute].unique())
    all_labels = sorted(config_df['label'].unique())

    if values_to_show is None:
        values_to_show = all_values
    else:
        values_to_show = [v for v in values_to_show if v in all_values]

    n_rows = (len(values_to_show) + n_cols - 1) // n_cols

    # Create colormap
    if len(all_labels) <= 10:
        cmap = plt.cm.tab10
    elif len(all_labels) <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.viridis

    offset = 2
    color_dict = {label: cmap((i + offset) % len(all_labels) / len(all_labels))
                 for i, label in enumerate(all_labels)}

    x_min, x_max = tsne_result[:, 0].min(), tsne_result[:, 0].max()
    y_min, y_max = tsne_result[:, 1].min(), tsne_result[:, 1].max()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), constrained_layout=False)
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

    for i, value in enumerate(values_to_show):
        if i >= len(axes):
            break

        ax = axes[i]
        value_mask = config_df[attribute] == value

        if not np.any(value_mask):
            ax.text(0.5, 0.5, f"No data for {attribute}={value}", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{attribute}={value}')
            ax.axis('off')
            continue

        value_count = np.sum(value_mask)
        value_tsne = tsne_result[value_mask]
        value_labels = config_df.loc[value_mask, 'label']

        for label in all_labels:
            label_mask = value_labels == label
            if not np.any(label_mask):
                continue

            ax.scatter(
                value_tsne[label_mask, 0],
                value_tsne[label_mask, 1],
                c=[color_dict[label]],
                label=label,
                alpha=0.7,
                s=50,
                edgecolors='w',
                linewidth=0.5
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        if attribute == 'age_weeks':
            ax.set_title(f'Week {value} (n={value_count})')
        elif attribute == 'age_days':
            ax.set_title(f'Day {value} (n={value_count})')
        else:
            ax.set_title(f'{attribute}={value} (n={value_count})')

        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f't-SNE Projections by {attribute.replace("_", " ").title()} of {data_name}', fontsize=60)
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])

    overall_label_counts = config_df['label'].value_counts()
    total_count = len(config_df)

    handles = []
    labels = []

    for label in all_labels:
        if label in overall_label_counts.index:
            percentage = (overall_label_counts[label] / total_count) * 100
            scatter = axes[0].scatter([], [], c=[color_dict[label]], alpha=0.7, s=50,
                                     edgecolors='w', linewidth=0.5)
            handles.append(scatter)
            labels.append(f"{label} ({percentage:.1f}%)")

    perc = [float(s.split('(')[-1].replace('%)','')) for s in labels]
    order = np.argsort(perc)[::-1]
    handles = [handles[k] for k in order]
    labels  = [labels[k]  for k in order]

    fig.legend(handles, labels, loc='center right', title='Labels', fontsize=80, markerscale=8)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to: {output_path}")
        plt.close()
    else:
        plt.show()


def analyze_developmental_stages(config_df, tsne_result=None,
                                 topk=5, age_unit="age_days", n_stages=3,
                                 tag="dataset",
                                 feature_mode="count_ratio+mae_tsne",
                                 show_ch=True,
                                 max_clusters=10,
                                 title_suffix="",
                                 sort_stage="avg_age",
                                 output_dir=None):
    """
    Perform hierarchical clustering analysis on developmental stages.
    """
    valid_modes = ["count_ratio", "mae_tsne", "count_ratio+mae_tsne"]
    if feature_mode not in valid_modes:
        raise ValueError(f"feature_mode must be one of {valid_modes}")

    if feature_mode in ["mae_tsne", "count_ratio+mae_tsne"] and tsne_result is None:
        raise ValueError(f"tsne_result is required for feature_mode='{feature_mode}'")

    print(f"Loaded {len(config_df)} samples")
    print(f"Feature mode: {feature_mode}")

    call_counts = config_df['label'].value_counts()
    top_k_calls = call_counts.head(topk).index.tolist()
    print(f"Top {topk} calls: {top_k_calls}")

    unique_ages = sorted(config_df[age_unit].unique())
    combined_data = []
    age_labels = []

    for age in unique_ages:
        age_data = config_df[config_df[age_unit] == age]
        total_samples = len(age_data)

        if total_samples == 0:
            continue

        call_counts_age = {}
        for call in top_k_calls:
            call_counts_age[call] = len(age_data[age_data['label'] == call])

        target_total = sum(call_counts_age.values())
        feature_vector = []

        if feature_mode in ["count_ratio", "count_ratio+mae_tsne"]:
            if target_total > 0:
                for call in top_k_calls:
                    feature_vector.append(call_counts_age[call] / target_total)
            else:
                feature_vector.extend([0.0] * len(top_k_calls))

        if feature_mode in ["mae_tsne", "count_ratio+mae_tsne"]:
            age_mask = config_df[age_unit] == age
            age_tsne = tsne_result[age_mask]
            age_labels_data = config_df.loc[age_mask, 'label']

            for call in top_k_calls:
                call_mask = age_labels_data == call
                if np.any(call_mask):
                    call_tsne_mean = age_tsne[call_mask].mean(axis=0)
                    feature_vector.extend([call_tsne_mean[0], call_tsne_mean[1]])
                else:
                    feature_vector.extend([0.0, 0.0])

        combined_data.append(feature_vector)
        age_labels.append(age)

    feature_columns = []
    if feature_mode in ["count_ratio", "count_ratio+mae_tsne"]:
        for call in top_k_calls:
            feature_columns.append(f'{call}_ratio')
    if feature_mode in ["mae_tsne", "count_ratio+mae_tsne"]:
        for call in top_k_calls:
            feature_columns.extend([f'{call}_tsne_x', f'{call}_tsne_y'])

    combined_df = pd.DataFrame(combined_data, columns=feature_columns)
    combined_df[age_unit] = age_labels

    print(f"Created features for {len(combined_df)} age points")

    X = combined_df[feature_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    linkage_matrix = linkage(X_scaled, method='ward', metric='euclidean')

    # Calculate and plot CH index
    if show_ch:
        cluster_range = range(2, min(max_clusters + 1, len(X_scaled)))
        ch_scores = []

        for n_clusters in cluster_range:
            labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            ch_score = calinski_harabasz_score(X_scaled, labels)
            ch_scores.append(ch_score)

        optimal_k = cluster_range[np.argmax(ch_scores)]
        optimal_score = max(ch_scores)

        print(f"\nOptimal number of clusters (by CH index): {optimal_k}")
        print(f"CH score at optimal k: {optimal_score:.1f}")

        plt.figure(figsize=(10, 6))
        plt.plot(list(cluster_range), ch_scores, 'o-', color='green', linewidth=2, markersize=8)
        plt.axvline(x=optimal_k, color='red', linestyle='--', linewidth=2,
                   label=f'Optimal k={optimal_k}')

        if n_stages != optimal_k:
            plt.axvline(x=n_stages, color='blue', linestyle=':', linewidth=2,
                       label=f'Selected k={n_stages}')

        plt.text(optimal_k, optimal_score, f'  CH={optimal_score:.1f}',
                fontsize=10, va='bottom')

        ch_title = f'Calinski-Harabasz Index (Higher = Better)\n{tag}{title_suffix}'
        plt.title(ch_title, fontsize=14, fontweight='bold')
        plt.xlabel('Number of Clusters', fontsize=12)
        plt.ylabel('CH Index', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        plt.tight_layout()

        if output_dir:
            ch_path = os.path.join(output_dir, f'ch_index_{feature_mode}.png')
            plt.savefig(ch_path, dpi=150, bbox_inches='tight')
            print(f"Saved CH index plot to: {ch_path}")
            plt.close()
        else:
            plt.show()

    stage_clusters = fcluster(linkage_matrix, n_stages, criterion='maxclust')
    combined_df['n_stages'] = stage_clusters

    stage_info = {}
    for stage in range(1, n_stages + 1):
        stage_data = combined_df[combined_df['n_stages'] == stage]
        ages = sorted(stage_data[age_unit].tolist())
        stage_info[stage] = {
            'avg_age': stage_data[age_unit].mean(),
            'min_age': stage_data[age_unit].min(),
            'max_age': stage_data[age_unit].max(),
            'ages': ages,
            'count': len(stage_data)
        }

    sorted_stages = sorted(stage_info.items(), key=lambda x: x[1][sort_stage])

    stage_mapping = {}
    dev_colors = plt.cm.Set3(np.linspace(0, 1, n_stages))

    if n_stages == 2:
        dev_names = ['Early Stage', 'Late Stage']
    elif n_stages == 3:
        dev_names = ['Early Stage', 'Middle Stage', 'Late Stage']
    elif n_stages == 4:
        dev_names = ['Early Stage', 'Early-Middle Stage', 'Late-Middle Stage', 'Late Stage']
    else:
        dev_names = [f'Stage {i+1}' for i in range(n_stages)]

    for i, (original_stage, info) in enumerate(sorted_stages):
        stage_mapping[original_stage] = i + 1
        info['dev_stage'] = i + 1
        info['color'] = dev_colors[i]
        info['name'] = dev_names[i]

    combined_df['dev_stage'] = combined_df['n_stages'].map(stage_mapping)

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    ax1 = axes[0]
    dendrogram(linkage_matrix,
               labels=combined_df[age_unit].values,
               ax=ax1,
               leaf_rotation=45,
               leaf_font_size=8)

    distances = linkage_matrix[:, 2]
    sorted_distances = np.sort(distances)
    if len(sorted_distances) >= n_stages:
        cut_height = sorted_distances[-(n_stages-1)] + 0.1
    else:
        cut_height = sorted_distances[-1] * 0.7

    ax1.axhline(y=cut_height, color='red', linestyle='--', linewidth=3,
               label=f'{n_stages}-Stage Cut', zorder=10)

    dendrogram_title = f'Hierarchical Clustering - {n_stages} Developmental Stages ({tag}){title_suffix}'
    ax1.set_title(dendrogram_title, fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'{age_unit}', fontsize=12)
    ax1.set_ylabel('Distance', fontsize=12)
    ax1.legend(fontsize=11)

    ax2 = axes[1]
    for i, (original_stage, info) in enumerate(sorted_stages):
        stage_ages = info['ages']
        stage_nums = [info['dev_stage']] * len(stage_ages)
        ax2.scatter(stage_ages, stage_nums,
                   s=80, alpha=0.7, color=info['color'],
                   label=f"{info['name']} (n={info['count']})")

    sequence_title = f'{n_stages}-Stage Developmental Sequence ({tag}){title_suffix}'
    ax2.set_title(sequence_title, fontsize=14, fontweight='bold')
    ax2.set_xlabel(f'{age_unit}', fontsize=12)
    ax2.set_ylabel('Developmental Stage', fontsize=12)
    ax2.set_yticks(range(1, n_stages + 1))
    ax2.set_yticklabels([f'{dev_names[i]} ({i+1})' for i in range(n_stages)])
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(right=0.92)

    if output_dir:
        cluster_path = os.path.join(output_dir, f'clustering_{feature_mode}.png')
        plt.savefig(cluster_path, dpi=150, bbox_inches='tight')
        print(f"Saved clustering plot to: {cluster_path}")
        plt.close()
    else:
        plt.show()

    # Print results
    print("\n" + "=" * 80)
    print(f"CLUSTERING RESULTS")
    print("=" * 80)
    for i, (original_stage, info) in enumerate(sorted_stages):
        print(f"\n{info['name']} (Stage {info['dev_stage']}):")
        print(f"  Age range: {info['min_age']}-{info['max_age']} {age_unit}")
        print(f"  Number of time points: {info['count']}")
        print(f"  Days: {info['ages']}")

    return combined_df, sorted_stages, linkage_matrix


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive MAE t-SNE Visualization'
    )

    # Required arguments
    parser.add_argument('--info_mae_csv', type=str, required=True,
                        help='Path to info_mae.csv file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for saving figures')
    parser.add_argument('--data_name', type=str, required=True,
                        help='Name of the dataset (e.g., b3_762F_763M_3201M)')

    # Data processing options
    parser.add_argument('--label_option', type=int, default=2, choices=[1, 2, 3],
                        help='Label processing: 1=keep all, 2=merge u-X, 3=remove u-X (default: 2)')

    # Visualization options
    parser.add_argument('--attribute', type=str, default='age_days',
                        help='Attribute to group by (default: age_days)')
    parser.add_argument('--n_cols', type=int, default=14,
                        help='Number of columns in grid layout (default: 14)')

    # Clustering options
    parser.add_argument('--topk', type=int, default=5,
                        help='Number of top call types to analyze (default: 5)')
    parser.add_argument('--n_stages', type=int, default=3,
                        help='Number of developmental stages (default: 3)')
    parser.add_argument('--feature_mode', type=str, default='count_ratio+mae_tsne',
                        choices=['count_ratio', 'mae_tsne', 'count_ratio+mae_tsne'],
                        help='Feature mode for clustering (default: count_ratio+mae_tsne)')
    parser.add_argument('--max_clusters', type=int, default=10,
                        help='Maximum clusters for CH index (default: 10)')
    parser.add_argument('--sort_stage', type=str, default='avg_age',
                        choices=['avg_age', 'min_age', 'max_age'],
                        help='Criterion for sorting developmental stages (default: avg_age)')
    

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("COMPREHENSIVE MAE t-SNE VISUALIZATION")
    print("=" * 80)
    print(f"Dataset: {args.data_name}")
    print(f"Input CSV: {args.info_mae_csv}")
    print(f"Output directory: {args.output_dir}")

    # Load and process data
    print("\n" + "=" * 80)
    print("LOADING AND PROCESSING DATA")
    print("=" * 80)
    config_df, tsne_result = load_and_process_data(
        csv_path=args.info_mae_csv,
        label_option=args.label_option,
        verbose=True
    )

    print(f"\nUnique {args.attribute}: {sorted(config_df[args.attribute].unique())}")

    # Generate t-SNE projection visualization
    print("\n" + "=" * 80)
    print("GENERATING t-SNE PROJECTIONS")
    print("=" * 80)
    tsne_output_path = os.path.join(args.output_dir, f'tsne_by_{args.attribute}.png')
    visualize_labels_by_attribute_compact(
        tsne_result=tsne_result,
        config_df=config_df,
        attribute=args.attribute,
        values_to_show=None,
        n_cols=args.n_cols,
        data_name=args.data_name,
        output_path=tsne_output_path
    )

    # Generate clustering analysis
    print("\n" + "=" * 80)
    print(f"DEVELOPMENTAL STAGE ANALYSIS ({args.feature_mode.upper()})")
    print("=" * 80)

    title_suffix = f"\n{args.feature_mode.replace('_', ' ').title()}"
    combined_df, sorted_stages, linkage_matrix = analyze_developmental_stages(
        config_df=config_df,
        tsne_result=tsne_result,
        topk=args.topk,
        age_unit=args.attribute,
        n_stages=args.n_stages,
        tag=args.data_name,
        feature_mode=args.feature_mode,
        show_ch=True,
        max_clusters=args.max_clusters,
        title_suffix=title_suffix,
        sort_stage=args.sort_stage,
        output_dir=args.output_dir
    )

    # Save combined results
    results_csv_path = os.path.join(args.output_dir, f'clustering_results_{args.feature_mode}.csv')
    combined_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved clustering results to: {results_csv_path}")

    print("\n" + "=" * 80)
    print("ALL VISUALIZATIONS COMPLETE")
    print("=" * 80)
    print(f"\nOutput files in: {args.output_dir}")
    print(f"  - {tsne_output_path}")
    print(f"  - {os.path.join(args.output_dir, f'ch_index_{args.feature_mode}.png')}")
    print(f"  - {os.path.join(args.output_dir, f'clustering_{args.feature_mode}.png')}")
    print(f"  - {results_csv_path}")


if __name__ == "__main__":
    main()

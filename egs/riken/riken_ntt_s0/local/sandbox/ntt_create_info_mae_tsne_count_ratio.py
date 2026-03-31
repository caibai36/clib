#!/usr/bin/env python3
"""
Create info CSV with MAE t-SNE coordinates, counts, and ratios for phone labels.
"""

import os
import argparse
import numpy as np
import pandas as pd
import yaml


def load_kana_to_phone(yaml_path):
    """Load kana_to_phone dictionary from YAML file
    
    Args:
        yaml_path: Path to the YAML file containing kana to phone mappings
        
    Returns:
        dict: Dictionary mapping kana to phone representations
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        kana_dict = yaml.safe_load(f)
    return kana_dict


def convert_kana_to_phone(kana_text, remove_spaces=True, kana_to_phone=None):
    """Convert Japanese kana to phonetic representation based on the kana2phone mapping
    
    Args:
        kana_text: Kana text to convert
        remove_spaces: Whether to remove spaces in the phone representation
        kana_to_phone: Dictionary mapping kana to phone representations
        
    Returns:
        str: Phone representation
    """
    if kana_to_phone is None:
        return kana_text
    
    # Handle special labels (non-kana)
    if kana_text.startswith('<') and kana_text.endswith('>'):
        return kana_text  # Keep special labels as is
    
    # Handle empty strings
    if not kana_text or kana_text.strip() == '':
        return ''
    
    # Convert kana to phone
    if kana_text in kana_to_phone:
        phone = kana_to_phone[kana_text]
        # Remove spaces if requested
        if remove_spaces:
            phone = phone.replace(' ', '')
        return phone
    else:
        # For unknown kana, return as is
        return kana_text


def process_labels(config_df, option='only_kana', label_col='label', custom_labels=None):
    """
    Process labels based on the specified option.
    
    Args:
        config_df: Input DataFrame
        option: Processing option ('all', 'all_long_vowel_merged', 'only_special', 
                'only_kana', 'only_kana_long_vowel_merged')
        label_col: Name of the label column
        custom_labels: List of custom labels to filter, or 'all' for all labels
        
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    processed_df = config_df.copy()
    
    # Handle custom labels
    if custom_labels is not None:
        if custom_labels == 'all':
            # Keep all labels, copy to phone column
            processed_df['phone'] = processed_df[label_col]
            return processed_df
        else:
            # Filter to only specified custom labels
            mask = processed_df[label_col].isin(custom_labels)
            processed_df = processed_df[mask]
            processed_df['phone'] = processed_df[label_col]
            return processed_df
    
    # Standard processing options
    if option == 'all':
        # Keep everything as is
        pass
    
    elif option == 'all_long_vowel_merged':
        # Merge long vowels: アー -> ア, XXXー -> XXX
        processed_df[label_col] = processed_df[label_col].str.replace('ー+$', '', regex=True)
    
    elif option == 'only_special':
        # Keep only special labels (those with < >)
        mask = processed_df[label_col].str.contains('<.*>', regex=True, na=False)
        processed_df = processed_df[mask]
    
    elif option == 'only_kana':
        # Keep only kana characters (exclude special labels)
        mask = ~processed_df[label_col].str.contains('<.*>', regex=True, na=False)
        processed_df = processed_df[mask]
    
    elif option == 'only_kana_long_vowel_merged':
        # Keep only kana and merge long vowels
        mask = ~processed_df[label_col].str.contains('<.*>', regex=True, na=False)
        processed_df = processed_df[mask]
        processed_df[label_col] = processed_df[label_col].str.replace('ー+$', '', regex=True)
    
    else:
        raise ValueError(f"Unknown option: {option}")
    
    return processed_df


def process_label_ntt(
    config_df_path='exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa_data_ma/info.csv',
    tsne_result_path='exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa_data_ma/mae_tsne.npy',
    kana2phone_yaml='/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_ntt_s0/conf/dict/ntt_kana2phone.yaml',
    output_file=None,
    output_dir='exp/sandbox/distribution/ma/only_kana_long_vowel_merged',
    option='only_kana',
    remove_spaces_in_phone=True,
    label_col_name='label',
    speaker_col_name='speaker',
    custom_labels=None
):
    """
    Main function to process NTT labels and create combined count ratio CSV.
    
    Args:
        config_df_path: Path to the config CSV file
        tsne_result_path: Path to the t-SNE result .npy file (optional)
        kana2phone_yaml: Path to the kana to phone YAML mapping file
        output_file: Output file path (if None, auto-generated)
        output_dir: Output directory
        option: Processing option ('only_kana', 'all', 'all_long_vowel_merged', 
                'only_special', 'only_kana_long_vowel_merged')
        remove_spaces_in_phone: Whether to remove spaces in phone representation
        label_col_name: Name of the label column in the config DataFrame
        speaker_col_name: Name of the speaker column (if None, skip speaker-level ratio)
        custom_labels: List of custom labels to filter (e.g., ['<laughter>', '<cry>']),
                       or 'all' to keep all labels and copy to phone column
    
    Returns:
        pd.DataFrame: Combined DataFrame with counts, ratios, and t-SNE coordinates
    """
    
    print("=" * 80)
    print("PROCESSING NTT LABELS")
    print("=" * 80)
    
    # Load data
    print(f"\nLoading data...")
    print(f"  Config: {config_df_path}")
    config_df = pd.read_csv(config_df_path)
    
    # Load t-SNE results if provided
    tsne_result = None
    has_tsne = False
    if tsne_result_path and os.path.exists(tsne_result_path):
        print(f"  t-SNE: {tsne_result_path}")
        tsne_result = np.load(tsne_result_path)
        has_tsne = True
        print(f"Loaded: config {len(config_df)} rows, t-SNE {tsne_result.shape}")
    else:
        print(f"Loaded: config {len(config_df)} rows")
        print(f"  Note: t-SNE results not found, skipping t-SNE columns")
    
    # Load kana to phone mapping
    print(f"\nLoading kana to phone mapping from: {kana2phone_yaml}")
    kana_to_phone = load_kana_to_phone(kana2phone_yaml)
    
    # Process labels
    print(f"\nProcessing labels with option: {option}")
    print(f"  Remove spaces in phone: {remove_spaces_in_phone}")
    print(f"  Label column: {label_col_name}")
    print(f"  Speaker column: {speaker_col_name}")
    if custom_labels:
        print(f"  Custom labels: {custom_labels}")
    
    config_df_processed = process_labels(
        config_df, 
        option=option, 
        label_col=label_col_name,
        custom_labels=custom_labels
    )
    
    # Convert kana to phone representation (skip if custom_labels is set)
    if custom_labels is None:
        config_df_processed['phone'] = config_df_processed[label_col_name].apply(
            lambda x: convert_kana_to_phone(x, remove_spaces=remove_spaces_in_phone, kana_to_phone=kana_to_phone)
        )
    
    print(f"  Original samples: {len(config_df)}")
    print(f"  Processed samples: {len(config_df_processed)}")
    
    # Show label distribution
    print(f"\nLabel distribution (top 20):")
    print(config_df_processed[label_col_name].value_counts().head(20))
    
    print(f"\nPhone representation distribution (top 20):")
    print(config_df_processed['phone'].value_counts().head(20))
    
    # Update t-SNE results if samples were filtered
    if has_tsne and len(config_df_processed) != len(config_df):
        remaining_indices = config_df_processed.index.tolist()
        tsne_result = tsne_result[remaining_indices]
        print(f"\nUpdated t-SNE shape: {tsne_result.shape}")
    
    # Reset index
    config_df_processed = config_df_processed.reset_index(drop=True)
    
    # Create combined count ratio CSV
    print("\n" + "=" * 80)
    print("CREATING COMBINED COUNT RATIO CSV")
    print("=" * 80)
    
    # Determine grouping columns
    has_speaker = speaker_col_name and speaker_col_name in config_df_processed.columns
    if has_speaker:
        group_cols = ['age_months', speaker_col_name, 'phone']
        total_group_cols = ['age_months', speaker_col_name]
    else:
        group_cols = ['age_months', 'phone']
        total_group_cols = ['age_months']
        print("\nNote: No speaker column found, computing ratios per age_months only")
    
    # Group by age_months, (speaker), and phone to get counts
    grouped = config_df_processed.groupby(group_cols).size().reset_index(name='count')
    
    # Calculate total count per age_months (and speaker) for ratio calculation
    total_counts = config_df_processed.groupby(total_group_cols).size().reset_index(name='total_count')
    
    # Merge with total counts
    grouped = grouped.merge(total_counts, on=total_group_cols, how='left')
    
    # Calculate ratio
    grouped['ratio'] = grouped['count'] / grouped['total_count']
    
    print(f"\nGrouped data shape: {grouped.shape}")
    
    # Calculate mean t-SNE coordinates if available
    if has_tsne:
        print("\nCalculating mean t-SNE coordinates for each group...")
        tsne_means = []
        for idx, row in grouped.iterrows():
            mask = (config_df_processed['age_months'] == row['age_months']) & \
                   (config_df_processed['phone'] == row['phone'])
            
            if has_speaker:
                mask = mask & (config_df_processed[speaker_col_name] == row[speaker_col_name])
            
            indices = config_df_processed[mask].index.tolist()
            mean_tsne_x = tsne_result[indices, 0].mean()
            mean_tsne_y = tsne_result[indices, 1].mean()
            
            tsne_data = {
                'age_months': row['age_months'],
                'phone': row['phone'],
                'tsne_x': mean_tsne_x,
                'tsne_y': mean_tsne_y
            }
            if has_speaker:
                tsne_data[speaker_col_name] = row[speaker_col_name]
            
            tsne_means.append(tsne_data)
            
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(grouped)} groups...")
        
        tsne_means_df = pd.DataFrame(tsne_means)
        
        # Merge t-SNE coordinates with grouped data
        merge_cols = [col for col in group_cols if col in tsne_means_df.columns]
        grouped = grouped.merge(tsne_means_df, on=merge_cols, how='left')
    
    # Pivot the data to create wide format
    print("\nPivoting data to wide format...")
    
    index_cols = ['age_months'] + ([speaker_col_name] if has_speaker else [])
    
    pivot_count = grouped.pivot_table(
        index=index_cols,
        columns='phone',
        values='count',
        fill_value=0
    ).add_suffix('_count')
    
    pivot_ratio = grouped.pivot_table(
        index=index_cols,
        columns='phone',
        values='ratio',
        fill_value=0.0
    ).add_suffix('_ratio')
    
    pivot_tables = [pivot_count, pivot_ratio]
    
    if has_tsne:
        pivot_tsne_x = grouped.pivot_table(
            index=index_cols,
            columns='phone',
            values='tsne_x',
            fill_value=0.0
        ).add_suffix('_tsne_x')
        
        pivot_tsne_y = grouped.pivot_table(
            index=index_cols,
            columns='phone',
            values='tsne_y',
            fill_value=0.0
        ).add_suffix('_tsne_y')
        
        pivot_tables.extend([pivot_tsne_x, pivot_tsne_y])
    
    # Combine all pivot tables
    combined_df = pd.concat(pivot_tables, axis=1)
    
    # Add total_count column
    combined_df = combined_df.merge(total_counts, on=index_cols, how='left')
    
    # Reset index
    combined_df = combined_df.reset_index()
    
    # Reorder columns
    unique_phones = sorted([p for p in config_df_processed['phone'].unique() if p != ''])
    
    column_order = ['age_months']
    if has_speaker:
        column_order.append(speaker_col_name)
    
    for phone in unique_phones:
        column_order.append(f'{phone}_count')
        column_order.append(f'{phone}_ratio')
        if has_tsne:
            column_order.append(f'{phone}_tsne_x')
            column_order.append(f'{phone}_tsne_y')
    
    column_order.append('total_count')
    
    # Filter to only existing columns
    column_order = [col for col in column_order if col in combined_df.columns]
    combined_df = combined_df[column_order]
    
    print(f"\nFinal combined dataframe shape: {combined_df.shape}")
    print(f"Sample of final data:")
    print(combined_df.head())
    
    # Save to CSV
    if output_file is None:
        tsne_suffix = '_tsne' if has_tsne else ''
        output_file = os.path.join(output_dir, f'info_mae{tsne_suffix}_count_ratio_{option}.csv')
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    combined_df.to_csv(output_file, index=False)
    
    print(f"\n✓ CSV saved to: {output_file}")
    print(f"  Rows: {len(combined_df)}")
    print(f"  Columns: {len(combined_df.columns)}")
    
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    
    return combined_df


def main():
    parser = argparse.ArgumentParser(
        description='Process NTT labels and create combined count ratio CSV'
    )
    
    parser.add_argument(
        '--config_df',
        default='exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa_data_ma/info.csv',
        help='Path to config CSV file'
    )
    
    parser.add_argument(
        '--tsne_result',
        default='exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa_data_ma/mae_tsne.npy',
        help='Path to t-SNE result .npy file (optional)'
    )
    
    parser.add_argument(
        '--kana2phone_yaml',
        default='/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_ntt_s0/conf/dict/ntt_kana2phone.yaml',
        help='Path to kana to phone YAML mapping file'
    )
    
    parser.add_argument(
        '--output_file',
        default=None,
        help='Output file path (auto-generated if not specified)'
    )
    
    parser.add_argument(
        '--output_dir',
        default='exp/sandbox/distribution/ma/only_kana_long_vowel_merged',
        help='Output directory'
    )
    
    parser.add_argument(
        '--option',
        default='only_kana',
        choices=['all', 'all_long_vowel_merged', 'only_special', 'only_kana', 'only_kana_long_vowel_merged'],
        help='Label processing option'
    )
    
    parser.add_argument(
        '--remove_spaces_in_phone',
        action='store_true',
        default=True,
        help='Remove spaces in phone representation'
    )
    
    parser.add_argument(
        '--no_remove_spaces_in_phone',
        action='store_false',
        dest='remove_spaces_in_phone',
        help='Keep spaces in phone representation'
    )
    
    parser.add_argument(
        '--label_col_name',
        default='label',
        help='Name of the label column'
    )
    
    parser.add_argument(
        '--speaker_col_name',
        default='speaker',
        help='Name of the speaker column'
    )
    
    parser.add_argument(
        '--custom_labels',
        nargs='+',
        default=None,
        help='Custom labels to filter (e.g., <laughter> <cry>), or "all" for all labels'
    )
    
    args = parser.parse_args()
    
    # Handle custom_labels 'all' case
    custom_labels = args.custom_labels
    if custom_labels and len(custom_labels) == 1 and custom_labels[0] == 'all':
        custom_labels = 'all'
    
    process_label_ntt(
        config_df_path=args.config_df,
        tsne_result_path=args.tsne_result,
        kana2phone_yaml=args.kana2phone_yaml,
        output_file=args.output_file,
        output_dir=args.output_dir,
        option=args.option,
        remove_spaces_in_phone=args.remove_spaces_in_phone,
        label_col_name=args.label_col_name,
        speaker_col_name=args.speaker_col_name,
        custom_labels=custom_labels
    )


if __name__ == '__main__':
    main()

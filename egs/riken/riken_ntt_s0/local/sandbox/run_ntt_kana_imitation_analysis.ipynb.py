#!/usr/bin/env python
# coding: utf-8
#
# python -u local/sandbox/run_ntt_kana_imitation_analysis.ipynb.py \
#     --data_name infant2_ma_family3 \
#     --mae_base_dir /work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_ntt_s0/exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa_data_ma \
#     --output_base_dir /work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_ntt_s0/exp/sandbox/imitation_analysis \
#     --kana2phone_yaml /work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_ntt_s0/conf/dict/ntt_kana2phone.yaml \
#     --distinctive_feature_csv /work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_ntt_s0/conf/dict/distinctive_feature.csv \
#     |& tee logs/imitation_analysis_infant2_ma_family3.log

import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy import stats
from collections import Counter, defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')
sns.set_context('paper')

# ========================== ARGUMENT PARSER =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description='NTT Imitation Analysis - Multiple Feature Types and Groupings',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output paths
    parser.add_argument('--data_name', type=str, required=True,
                        help='Data name (e.g., infant2_ma_family3)')
    parser.add_argument('--mae_base_dir', type=str, required=True,
                        help='Base directory containing MAE features (mae_raw.npy, mae_pca50.npy, mae_tsne.npy)')
    parser.add_argument('--output_base_dir', type=str, required=True,
                        help='Base output directory')

    # Dictionary paths
    parser.add_argument('--kana2phone_yaml', type=str, required=True,
                        help='Path to kana to phone YAML mapping')
    parser.add_argument('--distinctive_feature_csv', type=str, required=True,
                        help='Path to distinctive feature CSV')

    # Analysis parameters
    parser.add_argument('--age_col', type=str, default='age_months',
                        choices=['age_weeks', 'age_days', 'age_months'],
                        help='Age column to use')
    parser.add_argument('--threshold_sec', type=float, default=1.0,
                        help='Response detection threshold in seconds')
    parser.add_argument('--topk', type=int, default=1000,
                        help='Top K units to analyze (1000 includes all)')
    parser.add_argument('--label_option', type=str, default='only_kana',
                        choices=['only_kana', 'all'],
                        help='Label processing option')

    return parser.parse_args()

# ========================== UTILITY FUNCTIONS =========================
def load_kana_to_phone(yaml_path):
    """Load kana_to_phone dictionary from YAML file"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def convert_kana_to_phone(kana_text, kana_to_phone):
    """Convert Japanese kana to phonetic representation (remove spaces like original)"""
    if kana_text.startswith('<') and kana_text.endswith('>'):
        return kana_text
    if not kana_text or kana_text.strip() == '':
        return ''
    if kana_text in kana_to_phone:
        return kana_to_phone[kana_text].replace(' ', '')  # Remove spaces like original
    return kana_text

def process_labels(config_df, option='only_kana'):
    """Process labels based on the specified option

    Returns processed dataframe AND boolean mask for filtering
    """
    if option == 'only_kana':
        mask = ~config_df['label'].str.contains('<.*>', regex=True)
    else:
        mask = pd.Series([True] * len(config_df), index=config_df.index)

    processed_df = config_df[mask].copy()
    return processed_df, mask

def load_distinctive_features(csv_path):
    """Load distinctive feature mappings"""
    df = pd.read_csv(csv_path)

    mappings = {
        'distinctive_feature': {},
        'distinctive_feature_simplified': {},
        'place_of_articulation': {},
    }

    for _, row in df.iterrows():
        phone = row['Phoneme']
        mappings['distinctive_feature'][phone] = row['Distinctive_feature']
        mappings['distinctive_feature_simplified'][phone] = row['Distinctive_feature_simplified']

        # Place of articulation - can have multiple values
        place = row['Place_of_articulation']
        if pd.notna(place) and place != '-':
            mappings['place_of_articulation'][phone] = place.split(';')
        else:
            mappings['place_of_articulation'][phone] = []

    return mappings

# ========================== MAIN EXECUTION =========================
args = parse_args()

print("="*80)
print("NTT IMITATION ANALYSIS - MULTIPLE FEATURES & GROUPINGS")
print("="*80)
print(f"\nConfiguration:")
print(f"  Data name: {args.data_name}")
print(f"  MAE base dir: {args.mae_base_dir}")
print(f"  Output base dir: {args.output_base_dir}")
print(f"  Age column: {args.age_col}")
print(f"  Response threshold: {args.threshold_sec} seconds")
print(f"  Top K: {args.topk}")
print(f"  Label option: {args.label_option}")
print("="*80 + "\n")

# ========================== LOAD DATA =========================
print("Loading data...")

# Load distinctive features
feature_mappings = load_distinctive_features(args.distinctive_feature_csv)
print(f"✓ Loaded distinctive feature mappings")

# Load kana to phone mapping
kana_to_phone_dict = load_kana_to_phone(args.kana2phone_yaml)
print(f"✓ Loaded {len(kana_to_phone_dict)} kana to phone mappings")

# Load config CSV (info.csv already has speaker column!)
config_csv = os.path.join(args.mae_base_dir, 'info.csv')
config_df_full = pd.read_csv(config_csv)
print(f"✓ Loaded config: {len(config_df_full)} rows")
print(f"  Columns: {config_df_full.columns.tolist()}")

# Check speaker column
if 'speaker' in config_df_full.columns:
    print(f"✓ Speaker column found in info.csv")
    print(f"  Speaker distribution:")
    print(config_df_full['speaker'].value_counts())
else:
    print(f"ERROR: Speaker column not found in info.csv!")
    sys.exit(1)

# Load MAE features - all three types (BEFORE any filtering)
mae_features_full = {}
for feat_type in ['mae_raw', 'mae_pca50', 'mae_tsne']:
    feat_path = os.path.join(args.mae_base_dir, f'{feat_type}.npy')
    mae_features_full[feat_type] = np.load(feat_path)
    print(f"✓ Loaded {feat_type}: shape {mae_features_full[feat_type].shape}")

# Verify alignment
assert len(config_df_full) == mae_features_full['mae_raw'].shape[0], \
    f"Mismatch: config has {len(config_df_full)} rows but features have {mae_features_full['mae_raw'].shape[0]}"
print(f"✓ Verified alignment: {len(config_df_full)} samples in both config and features")

# Process labels (kana only by default) - GET MASK
config_df, filter_mask = process_labels(config_df_full, args.label_option)
print(f"✓ After filtering ({args.label_option}): {len(config_df)} rows")

# Apply same mask to features - CRITICAL FOR ALIGNMENT
mae_features = {}
for feat_type in mae_features_full:
    mae_features[feat_type] = mae_features_full[feat_type][filter_mask.values]
    print(f"✓ Filtered {feat_type} to shape: {mae_features[feat_type].shape}")

# Verify alignment again
assert len(config_df) == mae_features['mae_raw'].shape[0], \
    f"Mismatch after filtering: config has {len(config_df)} rows but features have {mae_features['mae_raw'].shape[0]}"
print(f"✓ Verified alignment after filtering: {len(config_df)} samples")

# Reset index for clean processing
config_df = config_df.reset_index(drop=True)

# Convert kana to phone - COMPACT VERSION (like original, no spaces)
config_df['phone'] = config_df['label'].apply(
    lambda x: convert_kana_to_phone(x, kana_to_phone_dict)
)

print(f"\n✓ Converted kana to phones (compact, no spaces)")
print(f"  Example conversions:")
for i in range(min(5, len(config_df))):
    row = config_df.iloc[i]
    print(f"    {row['label']:10s} → {row['phone']:10s}")

# Create phone_with_spaces for mapping to distinctive features
def get_phone_with_spaces(label, kana_to_phone):
    """Get phone representation WITH spaces preserved"""
    if label.startswith('<') and label.endswith('>'):
        return label
    if not label or label.strip() == '':
        return ''
    if label in kana_to_phone:
        return kana_to_phone[label]  # Keep spaces
    return label

config_df['phone_with_spaces'] = config_df['label'].apply(
    lambda x: get_phone_with_spaces(x, kana_to_phone_dict)
)

print(f"\n✓ Created phone_with_spaces for distinctive feature mapping")
print(f"  Examples:")
for i in range(min(5, len(config_df))):
    row = config_df.iloc[i]
    if ' ' in row['phone_with_spaces']:
        print(f"    {row['label']:10s} → '{row['phone_with_spaces']:10s}' (splits to: {row['phone_with_spaces'].split()})")

# Build kana to distinctive feature mapping
print("\nBuilding kana to distinctive feature mappings...")

def map_kana_to_features(phone_compact, phone_with_spaces, feature_mappings):
    """Map a kana (compact phone) to its distinctive features

    Returns dict with lists of features for each grouping type
    """
    # Split phone_with_spaces to get individual phones
    individual_phones = phone_with_spaces.split()

    result = {
        'kana': [phone_compact],  # For kana grouping, use compact version
        'distinctive_feature': [],
        'distinctive_feature_simplified': [],
        'place_of_articulation': []
    }

    # Collect features from all individual phones
    for phone in individual_phones:
        # Distinctive feature
        df = feature_mappings['distinctive_feature'].get(phone, phone)
        if df not in result['distinctive_feature']:
            result['distinctive_feature'].append(df)

        # Distinctive feature simplified
        dfs = feature_mappings['distinctive_feature_simplified'].get(phone, phone)
        if dfs not in result['distinctive_feature_simplified']:
            result['distinctive_feature_simplified'].append(dfs)

        # Place of articulation (can be multiple per phone)
        places = feature_mappings['place_of_articulation'].get(phone, [phone])
        for place in places:
            if place not in result['place_of_articulation']:
                result['place_of_articulation'].append(place)

    return result

# Apply mapping to each unique kana
kana_to_features_map = {}
for _, row in config_df.drop_duplicates('phone').iterrows():
    kana = row['phone']
    features = map_kana_to_features(kana, row['phone_with_spaces'], feature_mappings)
    kana_to_features_map[kana] = features

print(f"✓ Built mappings for {len(kana_to_features_map)} unique kana")
print(f"\nExample mappings:")
for i, (kana, features) in enumerate(list(kana_to_features_map.items())[-5:-1]):
    print(f"  Kana '{kana}':")
    print(f"    → DF: {features['distinctive_feature']}")
    print(f"    → DFS: {features['distinctive_feature_simplified']}")
    print(f"    → Place: {features['place_of_articulation']}")

# ========================== CREATE UTTERANCE-LEVEL DATA =========================
print("\n" + "="*80)
print("CREATING UTTERANCE-LEVEL DATA")
print("="*80)

def create_utt_level_data(config_df, features_dict, age_col, threshold_sec):
    """Create utterance-level aggregated data"""

    utt_data_dict = {}

    for feat_type, features in features_dict.items():
        print(f"\nProcessing {feat_type}...")

        # Create utterance-level features by averaging
        utt_features = pd.DataFrame(features)
        utt_features['utt_id'] = config_df['utt_id']
        utt_features = utt_features.groupby('utt_id').mean().reset_index()

        # Create utterance-level configuration
        utt_config = config_df.groupby('utt_id').agg({
            'dataid': 'first',
            'audioid': 'first',
            age_col: 'first',
            'begin_sec': 'min',
            'end_sec': 'max',
            'label': lambda x: ' '.join(x),
            'phone': lambda x: ' '.join(x),
            'speaker': 'first'
        }).reset_index()

        utt_config['duration'] = utt_config['end_sec'] - utt_config['begin_sec']

        # Merge features
        utt_data = utt_config.merge(utt_features, on='utt_id')

        # Detect responses
        utt_data['has_response'] = False
        utt_data['initiator'] = None
        utt_data['responder'] = None
        utt_data['responder_utterance_id'] = None

        # Sort by audio and time
        for audioid in utt_data['audioid'].unique():
            audio_mask = utt_data['audioid'] == audioid
            audio_data = utt_data[audio_mask].sort_values('begin_sec')
            utt_data.loc[audio_mask] = audio_data

        utt_data = utt_data.reset_index(drop=True)

        # Detect responses
        for audioid in utt_data['audioid'].unique():
            audio_mask = utt_data['audioid'] == audioid
            audio_indices = utt_data[audio_mask].index.tolist()

            for i, current_idx in enumerate(audio_indices[:-1]):
                current_row = utt_data.loc[current_idx]
                current_end = current_row['end_sec']
                current_speaker = current_row['speaker']

                for next_idx in audio_indices[i+1:]:
                    next_row = utt_data.loc[next_idx]
                    next_begin = next_row['begin_sec']
                    next_speaker = next_row['speaker']

                    if next_begin > current_end:
                        if (next_begin - current_end <= threshold_sec) and (current_speaker != next_speaker):
                            utt_data.at[current_idx, 'has_response'] = True
                            utt_data.at[current_idx, 'initiator'] = current_speaker
                            utt_data.at[current_idx, 'responder'] = next_speaker
                            utt_data.at[current_idx, 'responder_utterance_id'] = next_row['utt_id']
                        break

        utt_data_dict[feat_type] = utt_data
        print(f"  ✓ {len(utt_data)} utterances, {utt_data['has_response'].sum()} with response")

    return utt_data_dict

utt_data_dict = create_utt_level_data(config_df, mae_features, args.age_col, args.threshold_sec)

age_values = sorted(utt_data_dict['mae_tsne'][args.age_col].unique())
print(f"\n✓ Age values: {age_values}")

# ========================== IMITATION ANALYSIS FUNCTIONS =========================

def analyze_imitation_by_kana(utt_data, config_df, features, kana, age, parent, age_col):
    """
    Analyze imitation for a specific kana (phone-level unit)
    Returns kana-level distances that will be grouped by features later

    This matches the original script's logic exactly
    """
    age_data = utt_data[utt_data[age_col] == age]

    # Get parent -> child pairs
    parent_child_pairs = age_data[
        (age_data['speaker'] == parent) &
        (age_data['responder'] == 'child')
    ]

    results = {
        'kana': kana,
        'age': age,
        'parent': parent,
        'pair_details': []
    }

    # Pre-compute baseline: all child kana at this age
    all_child_kana_rows = config_df[
        (config_df[age_col] == age) &
        (config_df['speaker'] == 'child') &
        (config_df['phone'] == kana)
    ]

    if len(all_child_kana_rows) == 0:
        return results

    child_kana_indices = all_child_kana_rows.index.tolist()
    child_kana_mean = features[child_kana_indices].mean(axis=0)

    # Analyze each pair
    for _, parent_utt in parent_child_pairs.iterrows():
        # Check if parent utterance contains the target kana
        if kana not in parent_utt['phone'].split():
            continue

        # Get child response utterance
        child_resp_utt_id = parent_utt['responder_utterance_id']
        child_resp = utt_data[utt_data['utt_id'] == child_resp_utt_id]

        if len(child_resp) == 0:
            continue

        child_resp = child_resp.iloc[0]

        # Check if child response contains the target kana
        if kana not in child_resp['phone'].split():
            continue

        # Get kana-level features
        parent_kana_rows = config_df[
            (config_df['utt_id'] == parent_utt['utt_id']) &
            (config_df['phone'] == kana)
        ]

        child_resp_kana_rows = config_df[
            (config_df['utt_id'] == child_resp_utt_id) &
            (config_df['phone'] == kana)
        ]

        if len(parent_kana_rows) == 0 or len(child_resp_kana_rows) == 0:
            continue

        parent_kana_indices = parent_kana_rows.index.tolist()
        child_resp_kana_indices = child_resp_kana_rows.index.tolist()

        parent_kana_mean = features[parent_kana_indices].mean(axis=0)
        child_resp_kana_mean = features[child_resp_kana_indices].mean(axis=0)

        # Calculate distances
        imitation_dist = euclidean(parent_kana_mean, child_resp_kana_mean)
        baseline_dist = euclidean(parent_kana_mean, child_kana_mean)

        # Store pair details
        results['pair_details'].append({
            'kana': kana,
            'imitation_distance': imitation_dist,
            'baseline_distance': baseline_dist
        })

    return results

def group_kana_results_by_type(kana_results, grouping_type, kana_to_features_map):
    """
    Group kana-level results by the specified grouping type

    This allows the same kana's distances to contribute to multiple groups
    """
    grouped_results = defaultdict(lambda: {
        'imitation_distances': [],
        'baseline_distances': []
    })

    for result in kana_results:
        if len(result['pair_details']) == 0:
            continue

        kana = result['kana']
        age = result['age']
        parent = result['parent']

        # Get groups for this kana
        if kana not in kana_to_features_map:
            continue

        groups = kana_to_features_map[kana][grouping_type]

        # Add distances to each group this kana belongs to
        for group in groups:
            key = (group, age, parent)
            for pair in result['pair_details']:
                grouped_results[key]['imitation_distances'].append(pair['imitation_distance'])
                grouped_results[key]['baseline_distances'].append(pair['baseline_distance'])

    return grouped_results

def get_top_units_for_grouping(kana_to_features_map, grouping_type, topk=1000):
    """Get the most frequent units for a given grouping type"""
    all_units = []

    for kana, features in kana_to_features_map.items():
        all_units.extend(features[grouping_type])

    unit_counts = Counter(all_units)
    # Filter out special tokens and empty strings
    unit_counts_filtered = {u: c for u, c in unit_counts.items()
                           if not str(u).startswith('<') and str(u) != '' and pd.notna(u)}

    top_units = sorted(unit_counts_filtered.items(), key=lambda x: x[1], reverse=True)[:topk]
    return [u for u, _ in top_units], dict(top_units)

def calculate_corrected_pvalues(pvalues):
    """Calculate Bonferroni and Benjamini-Hochberg corrections"""
    n = len(pvalues)
    if n == 0:
        return np.array([]), np.array([])

    pvalues_arr = np.array(pvalues)

    # Bonferroni
    bonferroni = np.minimum(pvalues_arr * n, 1.0)

    # Benjamini-Hochberg (FDR)
    sorted_indices = np.argsort(pvalues_arr)
    sorted_pvalues = pvalues_arr[sorted_indices]

    bh_corrected = np.zeros(n)
    for i in range(n-1, -1, -1):
        bh_corrected[sorted_indices[i]] = min(sorted_pvalues[i] * n / (i + 1),
                                               1.0 if i == n-1 else bh_corrected[sorted_indices[i+1]])

    return bonferroni, bh_corrected

# ========================== RUN ANALYSIS FOR ALL COMBINATIONS =========================

print("\n" + "="*80)
print("RUNNING IMITATION ANALYSIS")
print("="*80)

grouping_types = ['kana', 'distinctive_feature', 'distinctive_feature_simplified', 'place_of_articulation']

for feat_type in ['mae_raw', 'mae_pca50', 'mae_tsne']:
    print(f"\n{'='*80}")
    print(f"FEATURE TYPE: {feat_type}")
    print(f"{'='*80}")

    # Create output directory
    output_dir = os.path.join(args.output_base_dir, args.data_name, feat_type)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    utt_data = utt_data_dict[feat_type]
    features = mae_features[feat_type]

    # ========================== STEP 1: Calculate kana-level distances (ONCE) =========================
    print(f"\n{'─'*80}")
    print("STEP 1: Calculating kana-level distances (same for all groupings)")
    print(f"{'─'*80}")

    # Get all unique kana (compact phone strings)
    all_kana = sorted(kana_to_features_map.keys())
    print(f"Analyzing {len(all_kana)} unique kana")

    # Calculate distances for each kana (this is done ONCE)
    kana_results = []

    for kana_idx, kana in enumerate(all_kana):
        if (kana_idx + 1) % 10 == 0:
            print(f"  Processing kana {kana_idx + 1}/{len(all_kana)}: {kana}")

        for age in age_values:
            for parent in ['mother', 'father']:
                result = analyze_imitation_by_kana(
                    utt_data, config_df, features,
                    kana, age, parent, args.age_col
                )
                if len(result['pair_details']) > 0:
                    kana_results.append(result)

    total_pairs = sum(len(r['pair_details']) for r in kana_results)
    print(f"✓ Calculated distances for {len(kana_results)} kana-age-parent combinations")
    print(f"✓ Total pairs analyzed: {total_pairs}")

    # ========================== STEP 2: Group by different types =========================
    for grouping_type in grouping_types:
        print(f"\n{'─'*80}")
        print(f"STEP 2: Grouping kana distances by: {grouping_type}")
        print(f"{'─'*80}")

        # Get top units for this grouping
        top_units, unit_counts = get_top_units_for_grouping(kana_to_features_map, grouping_type, args.topk)
        print(f"\nTop units for {grouping_type} (showing top 20):")
        for unit in top_units[:20]:
            count = unit_counts[unit]
            print(f"  {unit:30s}: {count:5d} kana map to this")
        if len(top_units) > 20:
            print(f"  ... and {len(top_units) - 20} more")
        print(f"\nTotal units to analyze: {len(top_units)}")

        # Group kana results by this grouping type
        print(f"\nGrouping kana-level distances by {grouping_type}...")
        grouped_results = group_kana_results_by_type(kana_results, grouping_type, kana_to_features_map)
        print(f"✓ Created {len(grouped_results)} group-age-parent combinations")

        # Show example groupings
        print(f"\nExample groupings for {grouping_type}:")
        example_kana = list(all_kana)[:5]
        for kana in example_kana:
            groups = kana_to_features_map[kana][grouping_type]
            print(f"  Kana '{kana}' → {grouping_type}: {groups}")

        # Show which kana contribute to each group (for first few groups)
        print(f"\nExample: Which kana contribute to each {grouping_type}:")
        for unit in top_units[:3]:
            contributing_kana = [k for k, f in kana_to_features_map.items() if unit in f[grouping_type]]
            print(f"  {grouping_type} '{unit}': {len(contributing_kana)} kana")
            print(f"    Examples: {contributing_kana[:10]}")

        # ========================== STEP 3: Statistical analysis =========================
        print(f"\n{'─'*80}")
        print(f"STEP 3: Statistical analysis for {grouping_type}")
        print(f"{'─'*80}")

        significance_results = []

        for unit in top_units:
            for parent in ['mother', 'father']:
                # Collect all distances for this unit-parent combination across all ages
                imitation_dists = []
                baseline_dists = []

                for age in age_values:
                    key = (unit, age, parent)
                    if key in grouped_results:
                        imitation_dists.extend(grouped_results[key]['imitation_distances'])
                        baseline_dists.extend(grouped_results[key]['baseline_distances'])

                if len(imitation_dists) < 3:
                    continue

                imitation_arr = np.array(imitation_dists)
                baseline_arr = np.array(baseline_dists)

                # Statistical tests
                t_stat, t_pval = stats.ttest_rel(baseline_arr, imitation_arr)
                wilcoxon_stat, wilcoxon_pval = stats.wilcoxon(baseline_arr, imitation_arr)

                # Effect size
                diff = baseline_arr - imitation_arr
                cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

                # Means
                mean_imitation = np.mean(imitation_arr)
                mean_baseline = np.mean(baseline_arr)
                mean_diff = mean_baseline - mean_imitation

                # Convergence percentage
                convergence_count = np.sum(diff > 0)
                convergence_pct = 100 * convergence_count / len(diff)

                significance_results.append({
                    'Unit': unit,
                    'Parent': parent,
                    'N': len(imitation_dists),
                    'Mean_Response': mean_imitation,
                    'Mean_Baseline': mean_baseline,
                    'Mean_Diff': mean_diff,
                    't_stat': t_stat,
                    'p_value': t_pval,
                    'p_value_wilcoxon': wilcoxon_pval,
                    'Cohens_d': cohens_d,
                    'Conv_%': convergence_pct
                })

        if len(significance_results) == 0:
            print("⚠ No significant resultsfor this grouping, skipping...")
            continue

        sig_df = pd.DataFrame(significance_results)
        print(f"✓ Statistical tests completed for {len(sig_df)} unit-parent combinations")

        # ========================== STEP 4: Multiple comparison corrections =========================
        print(f"\nApplying multiple comparison corrections...")

        # Calculate corrected p-values
        bonferroni_t, bh_t = calculate_corrected_pvalues(sig_df['p_value'].values)
        bonferroni_w, bh_w = calculate_corrected_pvalues(sig_df['p_value_wilcoxon'].values)

        sig_df['p_value_bonferroni'] = bonferroni_t
        sig_df['p_value_BH_FDR'] = bh_t
        sig_df['p_value_wilcoxon_bonferroni'] = bonferroni_w
        sig_df['p_value_wilcoxon_BH_FDR'] = bh_w

        # Reorder columns
        column_order = [
            'Unit', 'Parent', 'N',
            'Mean_Response', 'Mean_Baseline', 'Mean_Diff',
            't_stat', 'p_value', 'p_value_bonferroni', 'p_value_BH_FDR',
            'p_value_wilcoxon', 'p_value_wilcoxon_bonferroni', 'p_value_wilcoxon_BH_FDR',
            'Cohens_d', 'Conv_%'
        ]
        sig_df = sig_df[column_order]

        # Save results
        output_csv = os.path.join(output_dir, f'imitation_detailed_results_{grouping_type}.csv')
        sig_df.to_csv(output_csv, index=False)
        print(f"✓ Saved: {output_csv}")

        # Print summary
        print(f"\n{'─'*40}")
        print(f"SUMMARY for {grouping_type}:")
        print(f"{'─'*40}")
        print(f"  Total units tested: {len(sig_df)}")
        print(f"  Significant (p < 0.05, uncorrected t-test): {sig_df['p_value'].lt(0.05).sum()}")
        print(f"  Significant (Bonferroni corrected): {sig_df['p_value_bonferroni'].lt(0.05).sum()}")
        print(f"  Significant (BH FDR corrected): {sig_df['p_value_BH_FDR'].lt(0.05).sum()}")
        print(f"  Significant (Wilcoxon uncorrected): {sig_df['p_value_wilcoxon'].lt(0.05).sum()}")
        print(f"  Mean Cohen's d: {sig_df['Cohens_d'].mean():.3f}")
        print(f"  Mean convergence effect: {sig_df['Mean_Diff'].mean():.2f}")

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE")
print(f"Results saved to: {args.output_base_dir}/{args.data_name}/")
print("="*80)

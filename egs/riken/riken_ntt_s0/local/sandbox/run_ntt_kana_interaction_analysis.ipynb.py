#!/usr/bin/env python
# coding: utf-8
#
# python -u local/sandbox/run_ntt_kana_interaction_analysis.ipynb.py \
#     --input_csv /work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_ntt_s0/exp/mae_feat_dim_reduction/python_script/mae_pretrain_sa_data_ma/info_mae.csv \
#     --output_dir /work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_ntt_s0/exp/sandbox/interaction_analysis/infant2_ma_family3 \
#     --data_name "infant2_ma_family3" \
#     --kana_csv /data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/processed/all_kana_comment_token_speaker.csv \
#     --kana2phone_yaml /work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_ntt_s0/conf/dict/ntt_kana2phone.yaml \
#     |& tee logs/interaction_analysis_infant2_ma_family3.log

import argparse
import os
import sys
import yaml

# ========================== ARGUMENT PARSER =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Human NTT Interaction Analysis - MAE Feature Space',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output paths
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Path to input CSV file (info_mae.csv)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for all results')

    # Data identification
    parser.add_argument('--data_name', type=str, required=True,
                        help='Data name for identification (e.g., infant2_ma_family3)')

    # NTT-specific paths
    parser.add_argument('--kana_csv', type=str,
                        default='/data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/processed/all_kana_comment_token_speaker.csv',
                        help='Path to kana speaker CSV file')
    parser.add_argument('--kana2phone_yaml', type=str,
                        default='/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_ntt_s0/conf/dict/ntt_kana2phone.yaml',
                        help='Path to kana to phone YAML mapping')

    # Analysis parameters
    parser.add_argument('--age_col', type=str, default='age_months',
                        choices=['age_weeks', 'age_days', 'age_months'],
                        help='Age column to use for analysis')
    parser.add_argument('--label_option', type=str, default='only_kana',
                        choices=['all', 'all_long_vowel_merged', 'only_special', 'only_kana', 'only_kana_long_vowel_merged'],
                        help='Label processing option')
    parser.add_argument('--threshold_sec', type=float, default=1.0,
                        help='Response detection threshold in seconds')
    parser.add_argument('--top_n_labels', type=int, default=12,
                        help='Number of top labels for imitation analysis')

    # Plotting parameters
    parser.add_argument('--font_size', type=int, default=16,
                        help='Base font size for plots')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for saved figures')

    return parser.parse_args()

# ========================== MAIN EXECUTION =========================

args = parse_args()

# Set global variables from args
age_col = args.age_col
label_option = args.label_option
threshold_sec = args.threshold_sec
input_csv_path = args.input_csv
output_dir = args.output_dir
data_name = args.data_name
kana_csv_path = args.kana_csv
kana2phone_yaml_path = args.kana2phone_yaml

# Validate input file exists
if not os.path.exists(input_csv_path):
    print(f"ERROR: Input CSV file not found: {input_csv_path}")
    sys.exit(1)

# Create output directory
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("HUMAN NTT INTERACTION ANALYSIS - MAE FEATURE SPACE")
print("="*80)
print(f"\nConfiguration:")
print(f"  Data name: {data_name}")
print(f"  Input CSV: {input_csv_path}")
print(f"  Output directory: {output_dir}")
print(f"  Age column: {age_col}")
print(f"  Label processing option: {label_option}")
print(f"  Response threshold: {threshold_sec} seconds")
print(f"  Top N labels for analysis: {args.top_n_labels}")
print(f"  Kana CSV: {kana_csv_path}")
print(f"  Kana2Phone YAML: {kana2phone_yaml_path}")
print("="*80 + "\n")

# Setup matplotlib
import os
os.environ['PATH'] = '/project/nakamura-lab08/Work/bin-wu/.local/texlive/2018/bin/x86_64-linux:' + os.environ['PATH']
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style('white')
sns.set_context('paper')

import matplotlib
from math import sqrt
SPINE_COLOR = 'gray'

def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting."""
    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9

    if fig_height is None:
        golden_mean = (np.sqrt(5)-1.0)/2.0
        fig_height = fig_width*golden_mean

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + str(fig_height) +
              "so will reduce to" + str(MAX_HEIGHT_INCHES) + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    font_size=16
    params = {'backend': 'ps',
              'axes.labelsize': font_size,
              'axes.titlesize': font_size,
              'legend.fontsize': font_size,
              'legend.title_fontsize': font_size,
              'xtick.labelsize': font_size,
              'ytick.labelsize': font_size,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif',
              'errorbar.capsize': 4
    }

    matplotlib.rcParams.update(params)


def format_axes(ax):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax

font_size=16
latexify()

# ========================== LOAD KANA TO PHONE MAPPING =========================
print("\n" + "="*80)
print("LOADING KANA TO PHONE MAPPING")
print("="*80)

def load_kana_to_phone(yaml_path):
    """Load kana_to_phone dictionary from YAML file"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        kana_dict = yaml.safe_load(f)
    return kana_dict

kana_to_phone_dict = load_kana_to_phone(kana2phone_yaml_path)
print(f"\nLoaded {len(kana_to_phone_dict)} kana to phone mappings")

def convert_kana_to_phone(kana_text, remove_spaces=True, kana_to_phone=None):
    """Convert Japanese kana to phonetic representation"""
    if kana_to_phone is None:
        kana_to_phone = kana_to_phone_dict

    if kana_text.startswith('<') and kana_text.endswith('>'):
        return kana_text

    if not kana_text or kana_text.strip() == '':
        return ''

    if kana_text in kana_to_phone:
        phone = kana_to_phone[kana_text]
        if remove_spaces:
            phone = phone.replace(' ', '')
        return phone
    else:
        return kana_text

# ========================== LABEL PROCESSING =========================
def process_labels(config_df, option='only_kana'):
    """Process labels based on the specified option"""
    processed_df = config_df.copy()

    if option == 'all':
        pass
    elif option == 'all_long_vowel_merged':
        processed_df['label'] = processed_df['label'].str.replace('¡¼$', '', regex=True)
        processed_df['label'] = processed_df['label'].str.replace('¡¼¡¼$', '', regex=True)
    elif option == 'only_special':
        mask = processed_df['label'].str.contains('<.*>', regex=True)
        processed_df = processed_df[mask]
    elif option == 'only_kana':
        mask = ~processed_df['label'].str.contains('<.*>', regex=True)
        processed_df = processed_df[mask]
    elif option == 'only_kana_long_vowel_merged':
        mask = ~processed_df['label'].str.contains('<.*>', regex=True)
        processed_df = processed_df[mask]
        processed_df['label'] = processed_df['label'].str.replace('¡¼$', '', regex=True)
        processed_df['label'] = processed_df['label'].str.replace('¡¼¡¼$', '', regex=True)

    return processed_df

# ========================== LOAD NTT DATA =========================
print("\n" + "="*80)
print("LOADING NTT DATA")
print("="*80)

# Determine base directory from input CSV path
base_dir = os.path.dirname(input_csv_path)

# Try to find config CSV
import glob
possible_configs = glob.glob(os.path.join(base_dir, 'config_*.csv'))
if possible_configs:
    config_csv = possible_configs[0]
else:
    print(f"ERROR: Config CSV not found in {base_dir}")
    sys.exit(1)

print(f"\nLoading config CSV from: {config_csv}")
config_df = pd.read_csv(config_csv)

# Load info_mae CSV
info_mae_df = pd.read_csv(input_csv_path)

print(f"\nLoaded config data: {len(config_df)} rows")
print(f"Loaded info_mae data: {len(info_mae_df)} rows")
print(f"\nColumns in config_df: {config_df.columns.tolist()}")
print(f"Columns in info_mae_df: {info_mae_df.columns.tolist()}")

# Check if t-SNE coordinates are in info_mae or need to load separately
tsne_cols = [col for col in info_mae_df.columns if 'tsne' in col.lower()]
if len(tsne_cols) >= 2:
    print(f"\nFound t-SNE columns in info_mae: {tsne_cols}")
    tsne_result = info_mae_df[tsne_cols[:2]].values
else:
    # Load from separate file
    tsne_files = glob.glob(os.path.join(base_dir, 'tsne_result_*.npy'))
    if tsne_files:
        tsne_result = np.load(tsne_files[0])
        print(f"\nLoaded t-SNE from: {tsne_files[0]}")
    else:
        print(f"ERROR: t-SNE results not found")
        sys.exit(1)

print(f"\nt-SNE result shape: {tsne_result.shape}")
print(f"Config DataFrame shape: {config_df.shape}")

# ========================== DATA PREPROCESSING =========================
print("\n" + "="*80)
print("NTT LABEL PREPROCESSING")
print("="*80)

print(f"\nSelected label processing option: {label_option}")
config_df_processed = process_labels(config_df, label_option)

# Convert kana to phone representation
config_df_processed['phone'] = config_df_processed['label'].apply(
    lambda x: convert_kana_to_phone(x, remove_spaces=True)
)

print(f"\nOriginal samples: {len(config_df)}")
print(f"Processed samples: {len(config_df_processed)}")

print("\nLabel distribution after processing:")
print(config_df_processed['label'].value_counts().head(20))

print("\nPhone representation distribution:")
print(config_df_processed['phone'].value_counts().head(20))

# Update t-SNE if samples were filtered
if len(config_df_processed) != len(config_df):
    remaining_indices = config_df_processed.index.tolist()
    tsne_result = tsne_result[remaining_indices]
    print(f"\nUpdated t-SNE shape: {tsne_result.shape}")

config_df_processed = config_df_processed.reset_index(drop=True)

# ========================== APPEND SPEAKER INFORMATION =========================
print("\n" + "="*80)
print("APPENDING SPEAKER INFORMATION")
print("="*80)

speaker_df = pd.read_csv(kana_csv_path)
print(f"\nLoaded speaker data: {len(speaker_df)} rows")
print(f"Speaker distribution:")
print(speaker_df['speaker'].value_counts())

# Merge speaker column
config_df_processed = config_df_processed.merge(
    speaker_df[['session', 'speaker']],
    left_on='utt_id',
    right_on='session',
    how='left'
).drop(columns=['session'])

print(f"\nAfter merge: {len(config_df_processed)} rows")
print(f"Missing speaker values: {config_df_processed['speaker'].isna().sum()}")
print("\nSpeaker distribution in config_df_processed:")
print(config_df_processed['speaker'].value_counts())

# ========================== CREATE UTTERANCE-LEVEL DATA =========================
print("\n" + "="*80)
print("CREATING UTTERANCE-LEVEL DATA")
print("="*80)

# Create utterance-level t-SNE by averaging phone-level features
utt_tsne_result = pd.DataFrame(tsne_result, columns=['tsne_1', 'tsne_2'])
utt_tsne_result['utt_id'] = config_df_processed['utt_id']
utt_tsne_result = utt_tsne_result.groupby('utt_id')[['tsne_1', 'tsne_2']].mean().reset_index()

# Create utterance-level configuration dataframe
utt_config_df_processed = config_df_processed.groupby('utt_id').agg({
    'dataid': 'first',
    'audioid': 'first',
    age_col: 'first',
    'begin_sec': 'min',
    'end_sec': 'max',
    'label': lambda x: ' '.join(x),
    'phone': lambda x: ' '.join(x),
    'speaker': 'first'
}).reset_index()

utt_config_df_processed['duration'] = utt_config_df_processed['end_sec'] - utt_config_df_processed['begin_sec']

print(f"utt_tsne_result.shape: {utt_tsne_result.shape}")
print(f"utt_config_df_processed.shape: {utt_config_df_processed.shape}")
print("\nutt_config_df_processed.head():")
print(utt_config_df_processed.head())

# ========================== RESPONSE DETECTION =========================
print("\n" + "="*80)
print("DETECTING CONVERSATIONAL RESPONSES")
print("="*80)
print(f"Threshold: {threshold_sec} seconds")

# Assert utterances are sorted
for audioid in utt_config_df_processed['audioid'].unique():
    audio_data = utt_config_df_processed[utt_config_df_processed['audioid'] == audioid]
    if not audio_data['begin_sec'].is_monotonic_increasing:
        utt_config_df_processed.loc[
            utt_config_df_processed['audioid'] == audioid, :
        ] = audio_data.sort_values('begin_sec')
        print(f"Sorted utterances for {audioid}")

print("? All audio files have utterances sorted by begin_sec")

# Initialize columns
utt_config_df_processed['has_response'] = False
utt_config_df_processed['initiator'] = None
utt_config_df_processed['responder'] = None
utt_config_df_processed['responder_utterance_id'] = None

# Reset index
utt_config_df_processed = utt_config_df_processed.reset_index(drop=True)

# Process each audio file separately
for audioid in utt_config_df_processed['audioid'].unique():
    audio_mask = utt_config_df_processed['audioid'] == audioid
    audio_indices = utt_config_df_processed[audio_mask].index.tolist()

    for i, current_idx in enumerate(audio_indices[:-1]):
        current_row = utt_config_df_processed.loc[current_idx]
        current_end = current_row['end_sec']
        current_speaker = current_row['speaker']

        for next_idx in audio_indices[i+1:]:
            next_row = utt_config_df_processed.loc[next_idx]
            next_begin = next_row['begin_sec']
            next_speaker = next_row['speaker']

            if next_begin > current_end:
                if (next_begin - current_end <= threshold_sec) and (current_speaker != next_speaker):
                    utt_config_df_processed.at[current_idx, 'has_response'] = True
                    utt_config_df_processed.at[current_idx, 'initiator'] = current_speaker
                    utt_config_df_processed.at[current_idx, 'responder'] = next_speaker
                    utt_config_df_processed.at[current_idx, 'responder_utterance_id'] = next_row['utt_id']
                break

print(f"\nResponse statistics:")
print(f"Total utterances with response: {utt_config_df_processed['has_response'].sum()}")
print(f"Percentage: {100 * utt_config_df_processed['has_response'].mean():.2f}%")

print("\nResponse breakdown by initiator¢ªresponder:")
response_breakdown = utt_config_df_processed[utt_config_df_processed['has_response']].groupby(
    ['initiator', 'responder']
).size()
print(response_breakdown)

# ========================== MERGE WITH t-SNE FOR ANALYSIS =========================
print("\n" + "="*80)
print("MERGING DATA FOR ANALYSIS")
print("="*80)

utt_data = utt_config_df_processed.merge(utt_tsne_result, on='utt_id')

print(f"utt_data.shape: {utt_data.shape}")
print("\nColumns in utt_data:")
print(utt_data.columns.tolist())

age_values = sorted(utt_data[age_col].unique())
print(f"\n{age_col} values available: {age_values}")
print(f"Number of unique {age_col} values: {len(age_values)}")

print("\n" + "="*80)
print("DATA PREPARATION COMPLETE - READY FOR INTERACTION ANALYSIS")
print("="*80)

# Summary statistics
print("\nSummary Statistics:")
print(f"Total utterances: {len(utt_data)}")
print(f"Total audio files: {utt_data['audioid'].nunique()}")
print(f"Age range: {min(age_values)} - {max(age_values)} {age_col.replace('age_', '')}")
print(f"Total turn-taking pairs: {utt_data['has_response'].sum()}")
print(f"\nUtterances by speaker:")
print(utt_data['speaker'].value_counts())
print(f"\nUtterances by label type:")
print(utt_data['label'].value_counts().head(15))

# ========================== SAVE PROCESSED DATA =========================
print("\n" + "="*80)
print("SAVING PROCESSED DATA TO OUTPUT DIRECTORY")
print("="*80)

output_csv_config = os.path.join(output_dir, "utt_config_processed.csv")
output_csv_tsne = os.path.join(output_dir, "utt_tsne_result.csv")
output_csv_merged = os.path.join(output_dir, "utt_data_merged.csv")

utt_config_df_processed.to_csv(output_csv_config, index=False)
utt_tsne_result.to_csv(output_csv_tsne, index=False)
utt_data.to_csv(output_csv_merged, index=False)

print(f"? Saved utt_config_processed.csv to: {output_csv_config}")
print(f"? Saved utt_tsne_result.csv to: {output_csv_tsne}")
print(f"? Saved utt_data_merged.csv to: {output_csv_merged}")

output_npy_tsne = os.path.join(output_dir, "tsne_result.npy")
output_npy_config = os.path.join(output_dir, "config_processed.csv")

np.save(output_npy_tsne, tsne_result)
config_df_processed.to_csv(output_npy_config, index=False)

print(f"? Saved tsne_result.npy to: {output_npy_tsne}")
print(f"? Saved config_processed.csv to: {output_npy_config}")

# Save configuration info
import json
config_info = {
    'data_name': data_name,
    'age_col': age_col,
    'label_option': label_option,
    'threshold_sec': threshold_sec,
    'input_csv': input_csv_path,
    'output_dir': output_dir,
    'n_utterances': len(utt_data),
    'n_audio_files': utt_data['audioid'].nunique(),
    'age_range': f"{min(age_values)} - {max(age_values)}",
    'n_turn_pairs': int(utt_data['has_response'].sum())
}

config_json_path = os.path.join(output_dir, "analysis_config.json")
with open(config_json_path, 'w') as f:
    json.dump(config_info, f, indent=2)
print(f"? Saved analysis_config.json to: {config_json_path}")

print("\n" + "="*80)
print("ALL DATA SAVED SUCCESSFULLY")
print(f"Output directory: {output_dir}")
print("="*80)

# ========================== TRAJECTORY ANALYSIS =========================
print("\n" + "="*80)
print("VOCAL TRAJECTORY ANALYSIS BY COMMUNICATION STATE")
print("="*80)

import matplotlib.cm as cm
from matplotlib.patheffects import withStroke

# Calculate mean t-SNE positions for each communication state by age
trajectory_data = {
    'no_response': {'ages': [], 'tsne_1': [], 'tsne_2': [], 'counts': []},
    'resp_mother': {'ages': [], 'tsne_1': [], 'tsne_2': [], 'counts': []},
    'resp_father': {'ages': [], 'tsne_1': [], 'tsne_2': [], 'counts': []}
}

for age in age_values:
    age_data = utt_data[utt_data[age_col] == age]
    child_utts = age_data[age_data['speaker'] == 'child']

    # No response
    no_resp = child_utts[~child_utts['has_response']]
    if len(no_resp) > 0:
        trajectory_data['no_response']['ages'].append(age)
        trajectory_data['no_response']['tsne_1'].append(no_resp['tsne_1'].mean())
        trajectory_data['no_response']['tsne_2'].append(no_resp['tsne_2'].mean())
        trajectory_data['no_response']['counts'].append(len(no_resp))

    # Response by mother
    resp_mother = child_utts[child_utts['responder'] == 'mother']
    if len(resp_mother) > 0:
        trajectory_data['resp_mother']['ages'].append(age)
        trajectory_data['resp_mother']['tsne_1'].append(resp_mother['tsne_1'].mean())
        trajectory_data['resp_mother']['tsne_2'].append(resp_mother['tsne_2'].mean())
        trajectory_data['resp_mother']['counts'].append(len(resp_mother))

    # Response by father
    resp_father = child_utts[child_utts['responder'] == 'father']
    if len(resp_father) > 0:
        trajectory_data['resp_father']['ages'].append(age)
        trajectory_data['resp_father']['tsne_1'].append(resp_father['tsne_1'].mean())
        trajectory_data['resp_father']['tsne_2'].append(resp_father['tsne_2'].mean())
        trajectory_data['resp_father']['counts'].append(len(resp_father))

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Define styles for each communication state
state_styles = {
    'no_response': {
        'color': 'gray',
        'marker': 'o',
        'label': 'No Response',
        'title': 'No Response'
    },
    'resp_mother': {
        'color': 'green',
        'marker': 's',
        'label': 'Responded by Mother',
        'title': 'Response by Mother'
    },
    'resp_father': {
        'color': 'blue',
        'marker': '^',
        'label': 'Responded by Father',
        'title': 'Response by Father'
    }
}

# Age normalization for coloring
age_min, age_max = min(age_values), max(age_values)
norm = plt.Normalize(vmin=age_min, vmax=age_max)

# Plot each communication state separately
for idx, (state, data) in enumerate(trajectory_data.items()):
    ax = axes[idx]
    style = state_styles[state]

    if len(data['ages']) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
               transform=ax.transAxes, fontsize=14)
        ax.set_title(f"Child: {style['title']} (n=0)",
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.grid(True, alpha=0.3)
        continue

    ages = np.array(data['ages'])
    tsne_1 = np.array(data['tsne_1'])
    tsne_2 = np.array(data['tsne_2'])
    counts = np.array(data['counts'])

    total_count = sum(counts)

    # Plot trajectory line
    ax.plot(tsne_1, tsne_2, color=style['color'], linewidth=2.5,
            alpha=0.5, linestyle='-', zorder=1)

    # Add arrows to show direction
    if len(ages) > 1:
        for i in range(len(ages) - 1):
            ax.annotate('', xy=(tsne_1[i+1], tsne_2[i+1]),
                       xytext=(tsne_1[i], tsne_2[i]),
                       arrowprops=dict(arrowstyle='->', color=style['color'],
                                     lw=1.5, alpha=0.6), zorder=2)

    # Scatter plot with age-based colors
    scatter = ax.scatter(tsne_1, tsne_2, c=ages, cmap='viridis',
                        s=250, marker=style['marker'],
                        alpha=0.8, edgecolors='black', linewidth=2,
                        norm=norm, zorder=3)

    # Add age labels on each dot
    for i, (age, x, y, count) in enumerate(zip(ages, tsne_1, tsne_2, counts)):
        text = ax.annotate(f'{int(age)}', (x, y),
                          fontsize=9, fontweight='bold',
                          ha='center', va='center',
                          color='black', zorder=4)
        text.set_path_effects([withStroke(linewidth=2.5, foreground='white')])

    ax.set_xlabel('t-SNE 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('t-SNE 2', fontsize=12, fontweight='bold')
    ax.set_title(f"Child: {style['title']} (n={int(total_count)})",
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

# Add colorbar for age
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
sm = cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label(f'Age ({age_col.replace("age_", "")})', fontsize=12, fontweight='bold')

plt.suptitle(f'Child Utterance Trajectories by Communication State (Mean t-SNE positions)\n{data_name}',
            fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 0.92, 0.96])
plt.savefig(os.path.join(output_dir, 'child_communication_state_trajectories.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# Combined view: All three states on one plot
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

for state, data in trajectory_data.items():
    if len(data['ages']) == 0:
        continue

    style = state_styles[state]
    ages = np.array(data['ages'])
    tsne_1 = np.array(data['tsne_1'])
    tsne_2 = np.array(data['tsne_2'])
    counts = np.array(data['counts'])

    # Plot trajectory line
    ax.plot(tsne_1, tsne_2, color=style['color'], linewidth=2.5,
            alpha=0.5, linestyle='-', zorder=1, label=f"{style['label']} (trajectory)")

    # Add arrows
    if len(ages) > 1:
        for i in range(len(ages) - 1):
            ax.annotate('', xy=(tsne_1[i+1], tsne_2[i+1]),
                       xytext=(tsne_1[i], tsne_2[i]),
                       arrowprops=dict(arrowstyle='->', color=style['color'],
                                     lw=1.5, alpha=0.6), zorder=2)

    # Scatter plot
    scatter = ax.scatter(tsne_1, tsne_2, c=style['color'],
                        s=200, marker=style['marker'],
                        alpha=0.7, edgecolors='black', linewidth=2,
                        zorder=3, label=f"{style['label']} (points)")

    # Add age labels
    for age, x, y in zip(ages, tsne_1, tsne_2):
        text = ax.annotate(f'{int(age)}', (x, y),
                          fontsize=8, fontweight='bold',
                          ha='center', va='center',
                          color='white', zorder=4)
        text.set_path_effects([withStroke(linewidth=2, foreground='black')])

ax.set_xlabel('t-SNE 1', fontsize=13, fontweight='bold')
ax.set_ylabel('t-SNE 2', fontsize=13, fontweight='bold')
ax.set_title(f'Child Utterance Trajectories: All Communication States Combined\n{data_name}',
            fontsize=15, fontweight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'child_communication_state_trajectories_combined.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# Print trajectory statistics
print("\n" + "="*80)
print("CHILD UTTERANCE TRAJECTORY STATISTICS")
print("="*80)

# Collect trajectory statistics for CSV export
trajectory_stats_rows = []

for state, data in trajectory_data.items():
    style = state_styles[state]
    print(f"\n{style['label'].upper()}:")
    if len(data['ages']) == 0:
        print("  No data available")
        continue

    print(f"  Number of {age_col} with data: {len(data['ages'])}")
    print(f"  Total utterances: {sum(data['counts'])}")
    print(f"  {age_col.replace('age_', '').capitalize()}-by-{age_col.replace('age_', '')} positions:")
    for age, t1, t2, count in zip(data['ages'], data['tsne_1'],
                                   data['tsne_2'], data['counts']):
        print(f"    {age_col.replace('age_', '').capitalize()} {age:2d}: t-SNE=({t1:7.2f}, {t2:7.2f}), n={count:4d}")

        # Add to CSV data
        trajectory_stats_rows.append({
            'data_name': data_name,
            'communication_state': style['label'],
            'age': age,
            'tsne_1': t1,
            'tsne_2': t2,
            'n_utterances': count
        })

    # Calculate trajectory length
    if len(data['ages']) > 1:
        total_distance = 0
        for i in range(len(data['ages']) - 1):
            dist = np.sqrt((data['tsne_1'][i+1] - data['tsne_1'][i])**2 +
                          (data['tsne_2'][i+1] - data['tsne_2'][i])**2)
            total_distance += dist
        print(f"  Total trajectory distance: {total_distance:.2f}")
        print(f"  Average distance per {age_col.replace('age_', '')}: {total_distance / (len(data['ages']) - 1):.2f}")

# Save trajectory statistics to CSV
trajectory_stats_df = pd.DataFrame(trajectory_stats_rows)
trajectory_stats_csv = os.path.join(output_dir, 'trajectory_statistics.csv')
trajectory_stats_df.to_csv(trajectory_stats_csv, index=False)
print(f"\n? Saved trajectory statistics to: {trajectory_stats_csv}")

print("\n" + "="*80)

# ========================== RESPONSE RATE ANALYSIS =========================
print("\n" + "="*80)
print("RESPONSE RATE ANALYSIS")
print("="*80)

# Calculate response rates by speaker and age
response_data = []

for speaker in ['child', 'mother', 'father']:
    speaker_rates = []
    speaker_counts = []
    speaker_totals = []

    for age in age_values:
        age_data = utt_data[utt_data[age_col] == age]
        speaker_utts = age_data[age_data['speaker'] == speaker]

        total = len(speaker_utts)
        with_resp = speaker_utts['has_response'].sum()
        resp_rate = 100 * speaker_utts['has_response'].mean() if total > 0 else 0

        speaker_rates.append(resp_rate)
        speaker_counts.append(with_resp)
        speaker_totals.append(total)

    response_data.append({
        'speaker': speaker,
        'rates': speaker_rates,
        'counts': speaker_counts,
        'totals': speaker_totals
    })

# Create comprehensive response rate visualization
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3,
                      left=0.08, right=0.95, top=0.95, bottom=0.05)

# Plot 1: Response rate over time by speaker
ax1 = fig.add_subplot(gs[0, 0])
colors = {'child': 'red', 'mother': 'green', 'father': 'blue'}

for data in response_data:
    speaker = data['speaker']
    ax1.plot(age_values, data['rates'], marker='o', linewidth=2, markersize=8,
            color=colors[speaker], label=speaker.capitalize(), alpha=0.7)

ax1.set_xlabel(f'Age ({age_col.replace("age_", "")})', fontsize=12, fontweight='bold')
ax1.set_ylabel('Response Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title(f'Response Rate Development by Speaker\n{data_name}', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, max([max(d['rates']) for d in response_data if d['rates']]) * 1.1)

# Plot 2: Total utterances over time by speaker
ax2 = fig.add_subplot(gs[0, 1])

for data in response_data:
    speaker = data['speaker']
    ax2.plot(age_values, data['totals'], marker='s', linewidth=2, markersize=8,
            color=colors[speaker], label=speaker.capitalize(), alpha=0.7)

ax2.set_xlabel(f'Age ({age_col.replace("age_", "")})', fontsize=12, fontweight='bold')
ax2.set_ylabel('Total Utterances', fontsize=12, fontweight='bold')
ax2.set_title(f'Total Utterances by Speaker Over Time\n{data_name}', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Plot 3: Stacked bar chart - utterances with/without response
ax3 = fig.add_subplot(gs[1, 0])

width = 0.25
x = np.arange(len(age_values))

for idx, data in enumerate(response_data):
    speaker = data['speaker']
    no_resp = np.array(data['totals']) - np.array(data['counts'])
    with_resp = np.array(data['counts'])

    ax3.bar(x + idx * width, no_resp, width, label=f'{speaker.capitalize()} - No resp',
           color=colors[speaker], alpha=0.3)
    ax3.bar(x + idx * width, with_resp, width, bottom=no_resp,
           label=f'{speaker.capitalize()} - Has resp',
           color=colors[speaker], alpha=0.8)

ax3.set_xlabel(f'Age ({age_col.replace("age_", "")})', fontsize=12, fontweight='bold')
ax3.set_ylabel('Number of Utterances', fontsize=12, fontweight='bold')
ax3.set_title(f'Utterances with/without Response by Speaker\n{data_name}', fontsize=14, fontweight='bold')
ax3.set_xticks(x + width)
ax3.set_xticklabels(age_values, rotation=45, ha='right')
ax3.legend(fontsize=9, ncol=2, loc='upper left')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Response rate comparison table
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('tight')
ax4.axis('off')

# Create table data
display_ages = age_values[::max(1, len(age_values)//10)]  # Show subset if too many
table_data = [[f'{age_col.replace("age_", "").capitalize()}'] + [str(a) for a in display_ages]]

for data in response_data:
    speaker = data['speaker']
    row = [speaker.capitalize()]
    for i, age in enumerate(age_values):
        if age in display_ages:
            rate = data['rates'][i]
            count = data['counts'][i]
            total = data['totals'][i]
            row.append(f'{rate:.1f}%\n({count}/{total})')
    table_data.append(row)

col_width = min(0.15, 0.9 / len(display_ages))
table = ax4.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.12] + [col_width] * len(display_ages))
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2)

# Color code the header
for i in range(len(table_data[0])):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color code speaker rows
for idx, data in enumerate(response_data):
    table[(idx+1, 0)].set_facecolor(colors[data['speaker']])
    table[(idx+1, 0)].set_text_props(weight='bold', color='white')

ax4.set_title(f'Response Rate Summary Table\n{data_name}', fontsize=14, fontweight='bold', pad=20)

# Plot 5: Overall statistics bar chart
ax5 = fig.add_subplot(gs[2, :])

categories = ['Overall\nResponse Rate (%)', 'Total\nUtterances', 'Utterances\nwith Response']
x_pos = np.arange(len(categories))
width = 0.25

for idx, data in enumerate(response_data):
    speaker = data['speaker']
    overall_rate = 100 * sum(data['counts']) / sum(data['totals']) if sum(data['totals']) > 0 else 0
    values = [overall_rate, sum(data['totals']), sum(data['counts'])]

    ax5.bar(x_pos + idx * width, values, width, label=speaker.capitalize(),
            color=colors[speaker], alpha=0.7)

ax5.set_ylabel('Value', fontsize=12, fontweight='bold')
ax5.set_title(f'Overall Statistics Across All {age_col.replace("age_", "")}s\n{data_name}', fontsize=14, fontweight='bold')
ax5.set_xticks(x_pos + width)
ax5.set_xticklabels(categories, fontsize=11)
ax5.legend(fontsize=11)
ax5.grid(True, alpha=0.3, axis='y')

plt.savefig(os.path.join(output_dir, 'response_rate_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()

# Print summary statistics
print("\n" + "="*80)
print("RESPONSE RATE SUMMARY")
print("="*80)

# Collect response rate data for CSV
response_rate_rows = []

for data in response_data:
    speaker = data['speaker']
    print(f"\n{speaker.upper()}:")
    print(f"  Average response rate across all {age_col.replace('age_', '')}s: {np.mean(data['rates']):.2f}%")
    print(f"  Total utterances: {sum(data['totals'])}")
    print(f"  Total with response: {sum(data['counts'])}")
    print(f"  Overall response rate: {100 * sum(data['counts']) / sum(data['totals']):.2f}%")
    print(f"  {age_col.replace('age_', '').capitalize()}-by-{age_col.replace('age_', '')}:")
    for age, rate, count, total in zip(age_values, data['rates'], data['counts'], data['totals']):
        print(f"    {age_col.replace('age_', '').capitalize()} {age:2d}: {rate:5.1f}% ({count:4d}/{total:4d})")

        # Add to CSV data
        response_rate_rows.append({
            'data_name': data_name,
            'speaker': speaker,
            'age': age,
            'response_rate_pct': rate,
            'n_with_response': count,
            'n_total': total
        })

# Save response rate summary to CSV
response_rate_df = pd.DataFrame(response_rate_rows)
response_rate_csv = os.path.join(output_dir, 'response_rate_summary.csv')
response_rate_df.to_csv(response_rate_csv, index=False)
print(f"\n? Saved response rate summary to: {response_rate_csv}")

print("\n" + "="*80)

# Additional plot: Response breakdown by responder type
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.12, wspace=0.25)

for speaker_idx, speaker in enumerate(['child', 'mother', 'father']):
    ax = axes[speaker_idx]

    # Get response breakdown for each age
    responder_data = {}

    for age in age_values:
        age_data = utt_data[utt_data[age_col] == age]
        speaker_utts = age_data[age_data['speaker'] == speaker]
        has_resp = speaker_utts[speaker_utts['has_response']]

        if len(has_resp) > 0:
            responder_counts = has_resp['responder'].value_counts()
            for responder, count in responder_counts.items():
                if responder not in responder_data:
                    responder_data[responder] = [0] * len(age_values)
                responder_data[responder][age_values.index(age)] = count

    # Plot stacked bar chart
    bottom = np.zeros(len(age_values))
    responder_colors = {'child': 'red', 'mother': 'green', 'father': 'blue'}

    for responder in ['child', 'mother', 'father']:
        if responder in responder_data:
            values = responder_data[responder]
            ax.bar(age_values, values, bottom=bottom, label=f'¢ª {responder.capitalize()}',
                   color=responder_colors.get(responder, 'gray'), alpha=0.7)
            bottom += values

    ax.set_xlabel(f'Age ({age_col.replace("age_", "")})', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Responses Received', fontsize=11, fontweight='bold')
    ax.set_title(f'{speaker.capitalize()} - Who Responds?\n{data_name}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Rotate x-axis labels if many ages
    if len(age_values) > 15:
        ax.tick_params(axis='x', rotation=45)

plt.savefig(os.path.join(output_dir, 'response_breakdown_by_responder.png'),
            dpi=150, bbox_inches='tight')
plt.close()

print("\n" + "="*80)
print("RESPONSE RATE ANALYSIS COMPLETE")
print("="*80)

# ========================== IMITATION ANALYSIS SETUP =========================
print("\n" + "="*80)
print("PREPARING FOR IMITATION ANALYSIS")
print("="*80)

from scipy.spatial.distance import euclidean
from collections import Counter

def get_top_phones(df, top_n=12):
    """Get the most frequent phones"""
    all_phones = []
    for phones_str in df['phone'].dropna():
        all_phones.extend(phones_str.split())

    phone_counts = Counter(all_phones)
    # Filter out special tokens
    phone_counts_filtered = {p: c for p, c in phone_counts.items()
                            if not p.startswith('<')}

    # Sort by count and get top N
    top_phones = sorted(phone_counts_filtered.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_phones = [p for p, _ in top_phones]

    return top_phones

# Get top phones for analysis
top_phones = get_top_phones(config_df_processed, top_n=args.top_n_labels)
print(f"\nAnalyzing top {len(top_phones)} phones: {top_phones}")

# Show distribution
print("\nPhone distribution:")
for phone in top_phones:
    all_phones = []
    for phones_str in config_df_processed['phone'].dropna():
        all_phones.extend(phones_str.split())
    count = all_phones.count(phone)
    print(f"  /{phone}/: {count}")

print("\n" + "="*80)
print("READY FOR FORWARD IMITATION ANALYSIS (Parent¢ªChild)")
print("="*80)

# ========================== FORWARD IMITATION ANALYSIS =========================
print("\n" + "="*80)
print("FORWARD IMITATION ANALYSIS: Parent->Child Convergence")
print("="*80)

def analyze_imitation_by_phone(utt_data, config_df, tsne_result, phone, age, parent='mother'):
    """
    Analyze if child responses are closer to parent initiations than general child vocalizations
    OPTIMIZED VERSION
    """
    age_data = utt_data[utt_data[age_col] == age]

    # Get parent ¢ª child turn-taking pairs
    parent_child_pairs = age_data[
        (age_data['speaker'] == parent) &
        (age_data['responder'] == 'child')
    ]

    results = {
        'phone': phone,
        'age': age,
        'parent': parent,
        'imitation_distances': [],
        'baseline_distances': [],
        'pair_count': 0
    }

    # Pre-compute baseline once
    all_child_phone_rows = config_df[
        (config_df[age_col] == age) &
        (config_df['speaker'] == 'child') &
        (config_df['phone'] == phone)
    ]

    if len(all_child_phone_rows) == 0:
        return results

    child_phone_indices = all_child_phone_rows.index.tolist()
    child_phone_tsne_mean = tsne_result[child_phone_indices].mean(axis=0)

    for _, parent_utt in parent_child_pairs.iterrows():
        # Check if parent utterance contains the target phone
        if phone not in parent_utt['phone'].split():
            continue

        # Get child response utterance
        child_resp_utt_id = parent_utt['responder_utterance_id']
        child_resp = utt_data[utt_data['utt_id'] == child_resp_utt_id]

        if len(child_resp) == 0:
            continue

        child_resp = child_resp.iloc[0]

        # Check if child response contains the target phone
        if phone not in child_resp['phone'].split():
            continue

        # Get phone-level t-SNE features
        parent_phone_rows = config_df[
            (config_df['utt_id'] == parent_utt['utt_id']) &
            (config_df['phone'] == phone)
        ]

        child_resp_phone_rows = config_df[
            (config_df['utt_id'] == child_resp_utt_id) &
            (config_df['phone'] == phone)
        ]

        if len(parent_phone_rows) == 0 or len(child_resp_phone_rows) == 0:
            continue

        # Get mean t-SNE for this phone
        parent_phone_indices = parent_phone_rows.index.tolist()
        child_resp_phone_indices = child_resp_phone_rows.index.tolist()

        parent_phone_tsne = tsne_result[parent_phone_indices].mean(axis=0)
        child_resp_phone_tsne = tsne_result[child_resp_phone_indices].mean(axis=0)

        # Distance: parent phone to child response phone (imitation distance)
        imitation_dist = euclidean(parent_phone_tsne, child_resp_phone_tsne)

        # Distance: parent phone to mean child phone (baseline distance - pre-computed)
        baseline_dist = euclidean(parent_phone_tsne, child_phone_tsne_mean)

        results['imitation_distances'].append(imitation_dist)
        results['baseline_distances'].append(baseline_dist)
        results['pair_count'] += 1

    return results

# Run analysis for top phones across all ages
print(f"\nAnalyzing top {len(top_phones)} phones: {top_phones}")

all_results = []

for phone in top_phones:
    print(f"\nProcessing phone: /{phone}/")
    for age in age_values:
        for parent in ['mother', 'father']:
            result = analyze_imitation_by_phone(
                utt_data, config_df_processed, tsne_result,
                phone, age, parent
            )
            if result['pair_count'] > 0:
                all_results.append(result)
                print(f"  {age_col.replace('age_', '').capitalize()} {age}, {parent}: {result['pair_count']} pairs")

print(f"\nTotal results collected: {len(all_results)}")

# Process results into trajectory data
phone_trajectories = {}

for phone in top_phones:
    phone_trajectories[phone] = {
        'mother': {'ages': [], 'imitation': [], 'baseline': [], 'diff': [], 'ratio': [], 'counts': []},
        'father': {'ages': [], 'imitation': [], 'baseline': [], 'diff': [], 'ratio': [], 'counts': []}
    }

for result in all_results:
    phone = result['phone']
    parent = result['parent']
    age = result['age']

    if len(result['imitation_distances']) > 0:
        imitation_mean = np.mean(result['imitation_distances'])
        baseline_mean = np.mean(result['baseline_distances'])
        diff = baseline_mean - imitation_mean  # Positive = imitation effect
        ratio = imitation_mean / baseline_mean if baseline_mean > 0 else 1.0

        phone_trajectories[phone][parent]['ages'].append(age)
        phone_trajectories[phone][parent]['imitation'].append(imitation_mean)
        phone_trajectories[phone][parent]['baseline'].append(baseline_mean)
        phone_trajectories[phone][parent]['diff'].append(diff)
        phone_trajectories[phone][parent]['ratio'].append(ratio)
        phone_trajectories[phone][parent]['counts'].append(result['pair_count'])

print("\n" + "="*80)
print("FORWARD IMITATION DATA COLLECTED")
print("="*80)

# Save forward imitation trajectories to CSV
forward_traj_rows = []
for phone in top_phones:
    for parent in ['mother', 'father']:
        data = phone_trajectories[phone][parent]
        if len(data['ages']) == 0:
            continue
        for age, imit, base, diff, ratio, count in zip(
            data['ages'], data['imitation'], data['baseline'],
            data['diff'], data['ratio'], data['counts']
        ):
            forward_traj_rows.append({
                'data_name': data_name,
                'phone': phone,
                'parent': parent,
                'age': age,
                'imitation_distance': imit,
                'baseline_distance': base,
                'difference': diff,
                'ratio': ratio,
                'n_pairs': count
            })

forward_traj_df = pd.DataFrame(forward_traj_rows)
forward_traj_csv = os.path.join(output_dir, 'forward_imitation_trajectories.csv')
forward_traj_df.to_csv(forward_traj_csv, index=False)
print(f"? Saved forward imitation trajectories to: {forward_traj_csv}")

# ========================== VISUALIZATION: PARENT¢ªCHILD IMITATION =========================
print("\n" + "="*80)
print("VISUALIZING PARENT->CHILD IMITATION EFFECT")
print("="*80)

n_phones = len(top_phones)
n_cols = 4
n_rows = int(np.ceil(n_phones / n_cols))

# Visualization 1: Absolute distance trajectories - MOTHER
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
axes = axes.flatten() if n_phones > 1 else [axes]

for idx, phone in enumerate(top_phones):
    ax = axes[idx]
    data = phone_trajectories[phone]['mother']

    if len(data['ages']) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_title(f'/{phone}/ (n=0)', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'Age ({age_col.replace("age_", "")})', fontsize=10)
        ax.set_ylabel('t-SNE Distance', fontsize=10)
        ax.grid(True, alpha=0.3)
        continue

    ages = data['ages']
    total_count = sum(data['counts'])

    # Response = GREEN (solid line, circle)
    ax.plot(ages, data['imitation'],
            marker='o', linewidth=2.5, markersize=10,
            color='green', linestyle='-', alpha=0.8,
            label='Response')

    # Baseline = BLUE (dashed line, square)
    ax.plot(ages, data['baseline'],
            marker='s', linewidth=2.5, markersize=10,
            color='blue', linestyle='--', alpha=0.8,
            label='Baseline')

    ax.set_xlabel(f'Age ({age_col.replace("age_", "")})', fontsize=10, fontweight='bold')
    ax.set_ylabel('t-SNE Distance', fontsize=10, fontweight='bold')
    ax.set_title(f'/{phone}/ (n={total_count})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(n_phones, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle(f'Mother->Child: Imitation Effect by Phone\n(Green=Response, Blue=Baseline)\n{data_name}',
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'imitation_distance_mother_by_phone.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# Visualization 2: Absolute distance trajectories - FATHER
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
axes = axes.flatten() if n_phones > 1 else [axes]

for idx, phone in enumerate(top_phones):
    ax = axes[idx]
    data = phone_trajectories[phone]['father']

    if len(data['ages']) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_title(f'/{phone}/ (n=0)', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'Age ({age_col.replace("age_", "")})', fontsize=10)
        ax.set_ylabel('t-SNE Distance', fontsize=10)
        ax.grid(True, alpha=0.3)
        continue

    ages = data['ages']
    total_count = sum(data['counts'])

    # Response = GREEN (solid line, triangle)
    ax.plot(ages, data['imitation'],
            marker='^', linewidth=2.5, markersize=10,
            color='green', linestyle='-', alpha=0.8,
            label='Response')

    # Baseline = BLUE (dashed line, diamond)
    ax.plot(ages, data['baseline'],
            marker='D', linewidth=2.5, markersize=10,
            color='blue', linestyle='--', alpha=0.8,
            label='Baseline')

    ax.set_xlabel(f'Age ({age_col.replace("age_", "")})', fontsize=10, fontweight='bold')
    ax.set_ylabel('t-SNE Distance', fontsize=10, fontweight='bold')
    ax.set_title(f'/{phone}/ (n={total_count})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(n_phones, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle(f'Father->Child: Imitation Effect by Phone\n(Green=Response, Blue=Baseline)\n{data_name}',
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'imitation_distance_father_by_phone.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# Visualization 3: Imitation effect (difference) - MOTHER
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
axes = axes.flatten() if n_phones > 1 else [axes]

for idx, phone in enumerate(top_phones):
    ax = axes[idx]
    data = phone_trajectories[phone]['mother']

    if len(data['ages']) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_title(f'/{phone}/ (n=0)', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'Age ({age_col.replace("age_", "")})', fontsize=10)
        ax.set_ylabel('Imitation Effect', fontsize=10)
        ax.grid(True, alpha=0.3)
        continue

    ages = data['ages']
    total_count = sum(data['counts'])

    ax.plot(ages, data['diff'],
            marker='o', linewidth=2.5, markersize=10,
            color='black', linestyle='-', alpha=0.7)

    # Color individual points
    for age, diff in zip(ages, data['diff']):
        color = 'green' if diff > 0 else 'red'
        ax.scatter(age, diff, s=100, color=color, alpha=0.7, zorder=3,
                  edgecolors='black', linewidths=1.5)

    # Shaded regions
    ax.fill_between(ages, 0, data['diff'],
                    where=np.array(data['diff']) > 0,
                    color='green', alpha=0.2, label='Convergence')
    ax.fill_between(ages, 0, data['diff'],
                    where=np.array(data['diff']) < 0,
                    color='red', alpha=0.2, label='Divergence')

    ax.axhline(y=0, color='black', linestyle=':', linewidth=1.5, alpha=0.5)

    ax.set_xlabel(f'Age ({age_col.replace("age_", "")})', fontsize=10, fontweight='bold')
    ax.set_ylabel('Imitation Effect\n(Baseline - Response)', fontsize=10, fontweight='bold')
    ax.set_title(f'/{phone}/ (n={total_count})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(n_phones, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle(f'Mother->Child: Imitation Effect by Phone\n(Positive=Convergence, Negative=Divergence)\n{data_name}',
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'imitation_effect_mother_by_phone.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# Visualization 4: Imitation effect (difference) - FATHER
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
axes = axes.flatten() if n_phones > 1 else [axes]

for idx, phone in enumerate(top_phones):
    ax = axes[idx]
    data = phone_trajectories[phone]['father']

    if len(data['ages']) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_title(f'/{phone}/ (n=0)', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'Age ({age_col.replace("age_", "")})', fontsize=10)
        ax.set_ylabel('Imitation Effect', fontsize=10)
        ax.grid(True, alpha=0.3)
        continue

    ages = data['ages']
    total_count = sum(data['counts'])

    ax.plot(ages, data['diff'],
            marker='^', linewidth=2.5, markersize=10,
            color='black', linestyle='-', alpha=0.7)

    # Color individual points
    for age, diff in zip(ages, data['diff']):
        color = 'green' if diff > 0 else 'red'
        ax.scatter(age, diff, s=100, color=color, alpha=0.7, marker='^',
                  zorder=3, edgecolors='black', linewidths=1.5)

    # Shaded regions
    ax.fill_between(ages, 0, data['diff'],
                    where=np.array(data['diff']) > 0,
                    color='green', alpha=0.2, label='Convergence')
    ax.fill_between(ages, 0, data['diff'],
                    where=np.array(data['diff']) < 0,
                    color='red', alpha=0.2, label='Divergence')

    ax.axhline(y=0, color='black', linestyle=':', linewidth=1.5, alpha=0.5)

    ax.set_xlabel(f'Age ({age_col.replace("age_", "")})', fontsize=10, fontweight='bold')
    ax.set_ylabel('Imitation Effect\n(Baseline - Response)', fontsize=10, fontweight='bold')
    ax.set_title(f'/{phone}/ (n={total_count})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(n_phones, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle(f'Father->Child: Imitation Effect by Phone\n(Positive=Convergence, Negative=Divergence)\n{data_name}',
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'imitation_effect_father_by_phone.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# Print summary statistics
print("\n" + "="*80)
print("FORWARD IMITATION EFFECT SUMMARY (PARENT->CHILD)")
print("="*80)

for phone in top_phones:
    print(f"\n/{phone}/:")

    for parent in ['mother', 'father']:
        data = phone_trajectories[phone][parent]

        if len(data['ages']) == 0:
            continue

        print(f"  {parent.capitalize()}->Child:")

        avg_imitation = np.mean(data['imitation'])
        avg_baseline = np.mean(data['baseline'])
        avg_diff = np.mean(data['diff'])
        avg_ratio = np.mean(data['ratio'])

        print(f"    Average imitation distance: {avg_imitation:.2f}")
        print(f"    Average baseline distance: {avg_baseline:.2f}")
        print(f"    Average effect (baseline - imitation): {avg_diff:.2f}")
        print(f"    Average ratio (imitation / baseline): {avg_ratio:.3f}")
        print(f"    Convergence: {'YES' if avg_ratio < 1.0 else 'NO'}")
        print(f"    Total pairs analyzed: {sum(data['counts'])}")

        # Count ages with convergence
        convergence_ages = sum(1 for r in data['ratio'] if r < 1.0)
        print(f"    {age_col.replace('age_', '').capitalize()}s showing convergence: {convergence_ages}/{len(data['ages'])}")

print("\n" + "="*80)

print("\n? Forward imitation analysis complete")
print(f"? Results saved to: {output_dir}")

# ========================== STATISTICAL SIGNIFICANCE TESTING FORWARD IMITATION =========================
print("\n" + "="*80)
print("STATISTICAL SIGNIFICANCE TESTING: PARENT->CHILD IMITATION")
print("="*80)

from scipy import stats

def test_imitation_significance_detailed(all_results, top_phones):
    """
    Test if imitation distances are significantly different from baseline distances
    """

    significance_results = []

    for phone in top_phones:
        for parent in ['mother', 'father']:
            # Collect all paired distances across all ages
            imitation_dists = []
            baseline_dists = []

            for result in all_results:
                if result['phone'] == phone and result['parent'] == parent:
                    if len(result['imitation_distances']) > 0:
                        imitation_dists.extend(result['imitation_distances'])
                        baseline_dists.extend(result['baseline_distances'])

            if len(imitation_dists) < 3:  # Need minimum samples
                continue

            # Convert to arrays
            imitation_arr = np.array(imitation_dists)
            baseline_arr = np.array(baseline_dists)

            # Paired t-test
            t_stat, t_pval = stats.ttest_rel(baseline_arr, imitation_arr)

            # Wilcoxon signed-rank test
            wilcoxon_stat, wilcoxon_pval = stats.wilcoxon(baseline_arr, imitation_arr)

            # Effect size: Cohen's d
            diff = baseline_arr - imitation_arr
            cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

            # Mean values
            mean_imitation = np.mean(imitation_arr)
            mean_baseline = np.mean(baseline_arr)
            mean_diff = mean_baseline - mean_imitation

            # Percentage showing convergence
            convergence_count = np.sum(diff > 0)
            convergence_pct = 100 * convergence_count / len(diff)

            significance_results.append({
                'Phone': phone,
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

    return pd.DataFrame(significance_results)

# Run the analysis
sig_results_df = test_imitation_significance_detailed(all_results, top_phones)

# Create significance table
def create_significance_table(df, alpha=0.05):
    """
    Create a formatted table with significance highlighting
    """

    fig, ax = plt.subplots(figsize=(16, len(df) * 0.4 + 1))
    ax.axis('tight')
    ax.axis('off')

    # Prepare data for table
    table_data = []

    # Header
    headers = ['Phone', 'Parent', 'N', 'Mean\nResponse', 'Mean\nBaseline', 'Mean\nDiff',
               't-stat', 'p-value\n(t-test)', 'p-value\n(Wilcoxon)', "Cohen's\nd", 'Conv.\n%']
    table_data.append(headers)

    # Data rows
    for _, row in df.iterrows():
        table_row = [
            f"/{row['Phone']}/",
            row['Parent'].capitalize(),
            f"{row['N']}",
            f"{row['Mean_Response']:.2f}",
            f"{row['Mean_Baseline']:.2f}",
            f"{row['Mean_Diff']:.2f}",
            f"{row['t_stat']:.2f}",
            f"{row['p_value']:.4f}",
            f"{row['p_value_wilcoxon']:.4f}",
            f"{row['Cohens_d']:.3f}",
            f"{row['Conv_%']:.1f}%"
        ]
        table_data.append(table_row)

    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.10, 0.08, 0.06, 0.09, 0.09, 0.09, 0.08, 0.10, 0.10, 0.08, 0.08])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', fontsize=10)

    # Highlight significant rows
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        # Determine if significant
        sig_t = row['p_value'] < alpha
        sig_w = row['p_value_wilcoxon'] < alpha
        positive_effect = row['Mean_Diff'] > 0

        # Color code the row
        if sig_t and sig_w and positive_effect:
            row_color = '#C6E0B4'  # Light green - significant convergence
        elif sig_t and positive_effect:
            row_color = '#E2EFDA'  # Very light green - t-test significant only
        elif positive_effect:
            row_color = '#F2F2F2'  # Light gray - positive but not significant
        else:
            row_color = '#FCE4D6'  # Light orange - no convergence

        for col in range(len(headers)):
            table[(idx, col)].set_facecolor(row_color)

        # Bold significant p-values
        if sig_t:
            table[(idx, 7)].set_text_props(weight='bold')
        if sig_w:
            table[(idx, 8)].set_text_props(weight='bold')

    # Add legend
    legend_text = (
        "Highlighting Key:\n"
        "¢£ Dark Green: Both tests significant (p < 0.05) with positive convergence\n"
        "¢£ Light Green: t-test significant with positive convergence\n"
        "¢£ Gray: Positive effect but not significant\n"
        "¢£ Orange: No convergence effect\n"
        "Bold p-values: Significant (p < 0.05)"
    )

    ax.text(0.5, -0.05, legend_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', horizontalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(f'Statistical Significance of Acoustic Convergence (Parent->Child Imitation)\n{data_name}',
                fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(os.path.join(output_dir, 'imitation_significance_table.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

# Create the table
create_significance_table(sig_results_df, alpha=0.05)

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY OF STATISTICAL TESTS - PARENT->CHILD IMITATION")
print("="*80)

for _, row in sig_results_df.iterrows():
    print(f"\n/{row['Phone']}/ - {row['Parent'].capitalize()} Imitation:")
    print(f"  N pairs: {row['N']}")
    print(f"  Mean response distance: {row['Mean_Response']:.2f}")
    print(f"  Mean baseline distance: {row['Mean_Baseline']:.2f}")
    print(f"  Mean convergence effect: {row['Mean_Diff']:.2f}")
    print(f"  Cohen's d (effect size): {row['Cohens_d']:.3f}")
    print(f"  Convergence prevalence: {row['Conv_%']:.1f}%")
    print(f"  Paired t-test: t={row['t_stat']:.3f}, p={row['p_value']:.4f} {'***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else 'ns'}")
    print(f"  Wilcoxon test: p={row['p_value_wilcoxon']:.4f} {'***' if row['p_value_wilcoxon'] < 0.001 else '**' if row['p_value_wilcoxon'] < 0.01 else '*' if row['p_value_wilcoxon'] < 0.05 else 'ns'}")

# Summary statistics
print("\n" + "="*80)
print("OVERALL SUMMARY")
print("="*80)

print(f"\nTotal phone types tested: {len(sig_results_df)}")
print(f"Significant by t-test (p < 0.05): {sig_results_df['p_value'].lt(0.05).sum()} ({100*sig_results_df['p_value'].lt(0.05).mean():.1f}%)")
print(f"Significant by Wilcoxon (p < 0.05): {sig_results_df['p_value_wilcoxon'].lt(0.05).sum()} ({100*sig_results_df['p_value_wilcoxon'].lt(0.05).mean():.1f}%)")
print(f"Mean Cohen's d across all phones: {sig_results_df['Cohens_d'].mean():.3f}")
print(f"Mean convergence effect: {sig_results_df['Mean_Diff'].mean():.2f}")

# Save results
sig_results_df.to_csv(os.path.join(output_dir, 'imitation_detailed_results.csv'), index=False)
print(f"\n? Saved detailed results to: {os.path.join(output_dir, 'imitation_detailed_results.csv')}")

print("\n" + "="*80)

# ========================== REVERSE IMITATION ANALYSIS =========================
print("\n" + "="*80)
print("REVERSE IMITATION ANALYSIS: Child->Parent Convergence")
print("="*80)

def analyze_reverse_imitation_by_phone(utt_data, config_df, tsne_result, phone, age, parent='mother'):
    """
    Analyze if PARENT responses are closer to CHILD initiations than general parent vocalizations
    OPTIMIZED VERSION
    """
    age_data = utt_data[utt_data[age_col] == age]

    # Get child -> parent turn-taking pairs
    child_parent_pairs = age_data[
        (age_data['speaker'] == 'child') &
        (age_data['responder'] == parent)
    ]

    results = {
        'phone': phone,
        'age': age,
        'parent': parent,
        'imitation_distances': [],  # Now: child to parent response
        'baseline_distances': [],   # Now: child to parent general
        'pair_count': 0
    }

    # Pre-compute baseline once
    all_parent_phone_rows = config_df[
        (config_df[age_col] == age) &
        (config_df['speaker'] == parent) &
        (config_df['phone'] == phone)
    ]

    if len(all_parent_phone_rows) == 0:
        return results

    parent_phone_indices = all_parent_phone_rows.index.tolist()
    parent_phone_tsne_mean = tsne_result[parent_phone_indices].mean(axis=0)

    for _, child_utt in child_parent_pairs.iterrows():
        # Check if child utterance contains the target phone
        if phone not in child_utt['phone'].split():
            continue

        # Get parent response utterance
        parent_resp_utt_id = child_utt['responder_utterance_id']
        parent_resp = utt_data[utt_data['utt_id'] == parent_resp_utt_id]

        if len(parent_resp) == 0:
            continue

        parent_resp = parent_resp.iloc[0]

        # Check if parent response contains the target phone
        if phone not in parent_resp['phone'].split():
            continue

        # Get phone-level t-SNE features
        child_phone_rows = config_df[
            (config_df['utt_id'] == child_utt['utt_id']) &
            (config_df['phone'] == phone)
        ]

        parent_resp_phone_rows = config_df[
            (config_df['utt_id'] == parent_resp_utt_id) &
            (config_df['phone'] == phone)
        ]

        if len(child_phone_rows) == 0 or len(parent_resp_phone_rows) == 0:
            continue

        # Get mean t-SNE for this phone
        child_phone_indices = child_phone_rows.index.tolist()
        parent_resp_phone_indices = parent_resp_phone_rows.index.tolist()

        child_phone_tsne = tsne_result[child_phone_indices].mean(axis=0)
        parent_resp_phone_tsne = tsne_result[parent_resp_phone_indices].mean(axis=0)

        # Distance: child phone to parent response phone (imitation distance)
        imitation_dist = euclidean(child_phone_tsne, parent_resp_phone_tsne)

        # Distance: child phone to mean parent phone (baseline distance - pre-computed)
        baseline_dist = euclidean(child_phone_tsne, parent_phone_tsne_mean)

        results['imitation_distances'].append(imitation_dist)
        results['baseline_distances'].append(baseline_dist)
        results['pair_count'] += 1

    return results

# Run reverse analysis for top phones across all ages
print(f"\nAnalyzing top {len(top_phones)} phones: {top_phones}")

reverse_all_results = []

for phone in top_phones:
    print(f"\nProcessing phone: /{phone}/")
    for age in age_values:
        for parent in ['mother', 'father']:
            result = analyze_reverse_imitation_by_phone(
                utt_data, config_df_processed, tsne_result,
                phone, age, parent
            )
            if result['pair_count'] > 0:
                reverse_all_results.append(result)
                print(f"  {age_col.replace('age_', '').capitalize()} {age}, {parent}: {result['pair_count']} pairs")

print(f"\nTotal results collected: {len(reverse_all_results)}")

# Process results into trajectory data
reverse_phone_trajectories = {}

for phone in top_phones:
    reverse_phone_trajectories[phone] = {
        'mother': {'ages': [], 'imitation': [], 'baseline': [], 'diff': [], 'ratio': [], 'counts': []},
        'father': {'ages': [], 'imitation': [], 'baseline': [], 'diff': [], 'ratio': [], 'counts': []}
    }

for result in reverse_all_results:
    phone = result['phone']
    parent = result['parent']
    age = result['age']

    if len(result['imitation_distances']) > 0:
        imitation_mean = np.mean(result['imitation_distances'])
        baseline_mean = np.mean(result['baseline_distances'])
        diff = baseline_mean - imitation_mean  # Positive = parent convergence to child
        ratio = imitation_mean / baseline_mean if baseline_mean > 0 else 1.0

        reverse_phone_trajectories[phone][parent]['ages'].append(age)
        reverse_phone_trajectories[phone][parent]['imitation'].append(imitation_mean)
        reverse_phone_trajectories[phone][parent]['baseline'].append(baseline_mean)
        reverse_phone_trajectories[phone][parent]['diff'].append(diff)
        reverse_phone_trajectories[phone][parent]['ratio'].append(ratio)
        reverse_phone_trajectories[phone][parent]['counts'].append(result['pair_count'])

print("\n" + "="*80)
print("REVERSE IMITATION DATA COLLECTED")
print("="*80)

# Save data
import pickle
reverse_traj_path = os.path.join(output_dir, 'reverse_imitation_trajectories.pkl')
with open(reverse_traj_path, 'wb') as f:
    pickle.dump(reverse_phone_trajectories, f)
print(f"? Saved reverse imitation trajectories (pickle) to: {reverse_traj_path}")

# Also save as CSV
reverse_traj_rows = []
for phone in top_phones:
    for parent in ['mother', 'father']:
        data = reverse_phone_trajectories[phone][parent]
        if len(data['ages']) == 0:
            continue
        for age, imit, base, diff, ratio, count in zip(
            data['ages'], data['imitation'], data['baseline'],
            data['diff'], data['ratio'], data['counts']
        ):
            reverse_traj_rows.append({
                'data_name': data_name,
                'phone': phone,
                'parent': parent,
                'age': age,
                'imitation_distance': imit,
                'baseline_distance': base,
                'difference': diff,
                'ratio': ratio,
                'n_pairs': count
            })

reverse_traj_df = pd.DataFrame(reverse_traj_rows)
reverse_traj_csv = os.path.join(output_dir, 'reverse_imitation_trajectories.csv')
reverse_traj_df.to_csv(reverse_traj_csv, index=False)
print(f"? Saved reverse imitation trajectories (CSV) to: {reverse_traj_csv}")

# ========================== VISUALIZATION: CHILD->PARENT ADAPTATION =========================
print("\n" + "="*80)
print("VISUALIZING CHILD->PARENT ADAPTATION EFFECT")
print("="*80)

# Visualization 1: Absolute distance trajectories - MOTHER ADAPTATION
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
axes = axes.flatten() if n_phones > 1 else [axes]

for idx, phone in enumerate(top_phones):
    ax = axes[idx]
    data = reverse_phone_trajectories[phone]['mother']

    if len(data['ages']) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_title(f'/{phone}/ (n=0)', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'Age ({age_col.replace("age_", "")})', fontsize=10)
        ax.set_ylabel('t-SNE Distance', fontsize=10)
        ax.grid(True, alpha=0.3)
        continue

    ages = data['ages']
    total_count = sum(data['counts'])

    # Response = GREEN (solid line, circle)
    ax.plot(ages, data['imitation'],
            marker='o', linewidth=2.5, markersize=10,
            color='green', linestyle='-', alpha=0.8,
            label='Mother Response')

    # Baseline = BLUE (dashed line, square)
    ax.plot(ages, data['baseline'],
            marker='s', linewidth=2.5, markersize=10,
            color='blue', linestyle='--', alpha=0.8,
            label='Mother Baseline')

    ax.set_xlabel(f'Age ({age_col.replace("age_", "")})', fontsize=10, fontweight='bold')
    ax.set_ylabel('t-SNE Distance (from Child)', fontsize=10, fontweight='bold')
    ax.set_title(f'/{phone}/ (n={total_count})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(n_phones, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle(f'Child->Mother: Reverse Imitation Effect by Phone\n(Does Mother converge to Child? Green=Response, Blue=Baseline)\n{data_name}',
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'reverse_imitation_distance_mother_by_phone.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# Visualization 2: Absolute distance trajectories - FATHER ADAPTATION
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
axes = axes.flatten() if n_phones > 1 else [axes]

for idx, phone in enumerate(top_phones):
    ax = axes[idx]
    data = reverse_phone_trajectories[phone]['father']

    if len(data['ages']) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_title(f'/{phone}/ (n=0)', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'Age ({age_col.replace("age_", "")})', fontsize=10)
        ax.set_ylabel('t-SNE Distance', fontsize=10)
        ax.grid(True, alpha=0.3)
        continue

    ages = data['ages']
    total_count = sum(data['counts'])

    # Response = GREEN (solid line, triangle)
    ax.plot(ages, data['imitation'],
            marker='^', linewidth=2.5, markersize=10,
            color='green', linestyle='-', alpha=0.8,
            label='Father Response')

    # Baseline = BLUE (dashed line, diamond)
    ax.plot(ages, data['baseline'],
            marker='D', linewidth=2.5, markersize=10,
            color='blue', linestyle='--', alpha=0.8,
            label='Father Baseline')

    ax.set_xlabel(f'Age ({age_col.replace("age_", "")})', fontsize=10, fontweight='bold')
    ax.set_ylabel('t-SNE Distance (from Child)', fontsize=10, fontweight='bold')
    ax.set_title(f'/{phone}/ (n={total_count})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(n_phones, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle(f'Child->Father: Reverse Imitation Effect by Phone\n(Does Father converge to Child? Green=Response, Blue=Baseline)\n{data_name}',
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'reverse_imitation_distance_father_by_phone.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# Visualization 3: Adaptation effect (difference) - MOTHER
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
axes = axes.flatten() if n_phones > 1 else [axes]

for idx, phone in enumerate(top_phones):
    ax = axes[idx]
    data = reverse_phone_trajectories[phone]['mother']

    if len(data['ages']) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_title(f'/{phone}/ (n=0)', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'Age ({age_col.replace("age_", "")})', fontsize=10)
        ax.set_ylabel('Convergence Effect', fontsize=10)
        ax.grid(True, alpha=0.3)
        continue

    ages = data['ages']
    total_count = sum(data['counts'])

    ax.plot(ages, data['diff'],
            marker='o', linewidth=2.5, markersize=10,
            color='black', linestyle='-', alpha=0.7)

    # Color individual points
    for age, diff in zip(ages, data['diff']):
        color = 'green' if diff > 0 else 'red'
        ax.scatter(age, diff, s=100, color=color, alpha=0.7, zorder=3,
                  edgecolors='black', linewidths=1.5)

    # Shaded regions
    ax.fill_between(ages, 0, data['diff'],
                    where=np.array(data['diff']) > 0,
                    color='green', alpha=0.2, label='Convergence')
    ax.fill_between(ages, 0, data['diff'],
                    where=np.array(data['diff']) < 0,
                    color='red', alpha=0.2, label='Divergence')

    ax.axhline(y=0, color='black', linestyle=':', linewidth=1.5, alpha=0.5)

    ax.set_xlabel(f'Age ({age_col.replace("age_", "")})', fontsize=10, fontweight='bold')
    ax.set_ylabel('Mother Convergence Effect\n(Baseline - Response)', fontsize=10, fontweight='bold')
    ax.set_title(f'/{phone}/ (n={total_count})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(n_phones, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle(f'Child->Mother: Maternal Convergence Effect by Phone\n(Positive=Mother converges toward Child)\n{data_name}',
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'reverse_imitation_effect_mother_by_phone.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# Visualization 4: Adaptation effect (difference) - FATHER
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
axes = axes.flatten() if n_phones > 1 else [axes]

for idx, phone in enumerate(top_phones):
    ax = axes[idx]
    data = reverse_phone_trajectories[phone]['father']

    if len(data['ages']) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_title(f'/{phone}/ (n=0)', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'Age ({age_col.replace("age_", "")})', fontsize=10)
        ax.set_ylabel('Convergence Effect', fontsize=10)
        ax.grid(True, alpha=0.3)
        continue

    ages = data['ages']
    total_count = sum(data['counts'])

    ax.plot(ages, data['diff'],
            marker='^', linewidth=2.5, markersize=10,
            color='black', linestyle='-', alpha=0.7)

    # Color individual points
    for age, diff in zip(ages, data['diff']):
        color = 'green' if diff > 0 else 'red'
        ax.scatter(age, diff, s=100, color=color, alpha=0.7, marker='^',
                  zorder=3, edgecolors='black', linewidths=1.5)

    # Shaded regions
    ax.fill_between(ages, 0, data['diff'],
                    where=np.array(data['diff']) > 0,
                    color='green', alpha=0.2, label='Convergence')
    ax.fill_between(ages, 0, data['diff'],
                    where=np.array(data['diff']) < 0,
                    color='red', alpha=0.2, label='Divergence')

    ax.axhline(y=0, color='black', linestyle=':', linewidth=1.5, alpha=0.5)

    ax.set_xlabel(f'Age ({age_col.replace("age_", "")})', fontsize=10, fontweight='bold')
    ax.set_ylabel('Father Convergence Effect\n(Baseline - Response)', fontsize=10, fontweight='bold')
    ax.set_title(f'/{phone}/ (n={total_count})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

# Hide unused subplots
for idx in range(n_phones, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle(f'Child->Father: Paternal Convergence Effect by Phone\n(Positive=Father converges toward Child)\n{data_name}',
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'reverse_imitation_effect_father_by_phone.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# Print Summary Statistics
print("\n" + "="*80)
print("REVERSE IMITATION EFFECT SUMMARY (CHILD->PARENT)")
print("="*80)

for phone in top_phones:
    print(f"\n/{phone}/:")

    for parent in ['mother', 'father']:
        data = reverse_phone_trajectories[phone][parent]

        if len(data['ages']) == 0:
            continue

        print(f"  {parent.capitalize()} Adaptation:")

        avg_imitation = np.mean(data['imitation'])
        avg_baseline = np.mean(data['baseline'])
        avg_diff = np.mean(data['diff'])
        avg_ratio = np.mean(data['ratio'])

        print(f"    Average response distance: {avg_imitation:.2f}")
        print(f"    Average baseline distance: {avg_baseline:.2f}")
        print(f"    Average convergence effect: {avg_diff:.2f}")
        print(f"    Average ratio (response/baseline): {avg_ratio:.3f}")
        print(f"    Convergence to child: {'YES' if avg_ratio < 1.0 else 'NO'}")
        print(f"    Total pairs analyzed: {sum(data['counts'])}")

        # Count ages with convergence
        convergence_ages = sum(1 for r in data['ratio'] if r < 1.0)
        print(f"    {age_col.replace('age_', '').capitalize()}s showing convergence: {convergence_ages}/{len(data['ages'])}")

print("\n" + "="*80)

# ========================== STATISTICAL TESTING: REVERSE IMITATION =========================
print("\n" + "="*80)
print("STATISTICAL TESTING: PARENT ADAPTATION TO CHILD")
print("="*80)

def test_reverse_imitation_significance(reverse_all_results, top_phones):
    """
    Test if parent response distances are significantly different from baseline distances
    """

    significance_results = []

    for phone in top_phones:
        for parent in ['mother', 'father']:
            # Collect all paired distances across all ages
            response_dists = []
            baseline_dists = []

            for result in reverse_all_results:
                if result['phone'] == phone and result['parent'] == parent:
                    if len(result['imitation_distances']) > 0:
                        response_dists.extend(result['imitation_distances'])
                        baseline_dists.extend(result['baseline_distances'])

            if len(response_dists) < 3:
                continue

            # Convert to arrays
            response_arr = np.array(response_dists)
            baseline_arr = np.array(baseline_dists)

            # Paired t-test
            t_stat, t_pval = stats.ttest_rel(baseline_arr, response_arr)

            # Wilcoxon signed-rank test
            wilcoxon_stat, wilcoxon_pval = stats.wilcoxon(baseline_arr, response_arr)

            # Effect size: Cohen's d
            diff = baseline_arr - response_arr
            cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

            # Mean differences
            mean_response = np.mean(response_arr)
            mean_baseline = np.mean(baseline_arr)
            mean_diff = mean_baseline - mean_response

            # Percentage showing convergence
            convergence_count = np.sum(diff > 0)
            convergence_pct = 100 * convergence_count / len(diff)

            significance_results.append({
                'Phone': phone,
                'Parent': parent,
                'N': len(response_dists),
                'Mean_Response': mean_response,
                'Mean_Baseline': mean_baseline,
                'Mean_Diff': mean_diff,
                't_stat': t_stat,
                'p_value': t_pval,
                'p_value_wilcoxon': wilcoxon_pval,
                'Cohens_d': cohens_d,
                'Conv_%': convergence_pct
            })

    return pd.DataFrame(significance_results)

# Run statistical tests
reverse_sig_results = test_reverse_imitation_significance(reverse_all_results, top_phones)

# Create table
def create_reverse_significance_table(df, alpha=0.05):
    """
    Create a formatted table with significance highlighting for parent adaptation
    """

    fig, ax = plt.subplots(figsize=(16, len(df) * 0.4 + 1))
    ax.axis('tight')
    ax.axis('off')

    # Prepare data for table
    table_data = []

    # Header
    headers = ['Phone', 'Parent', 'N', 'Mean\nResponse', 'Mean\nBaseline', 'Mean\nDiff',
               't-stat', 'p-value\n(t-test)', 'p-value\n(Wilcoxon)', "Cohen's\nd", 'Conv.\n%']
    table_data.append(headers)

    # Data rows
    for _, row in df.iterrows():
        table_row = [
            f"/{row['Phone']}/",
            row['Parent'].capitalize(),
            f"{row['N']}",
            f"{row['Mean_Response']:.2f}",
            f"{row['Mean_Baseline']:.2f}",
            f"{row['Mean_Diff']:.2f}",
            f"{row['t_stat']:.2f}",
            f"{row['p_value']:.4f}",
            f"{row['p_value_wilcoxon']:.4f}",
            f"{row['Cohens_d']:.3f}",
            f"{row['Conv_%']:.1f}%"
        ]
        table_data.append(table_row)

    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.10, 0.08, 0.06, 0.09, 0.09, 0.09, 0.08, 0.10, 0.10, 0.08, 0.08])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(len(headers)):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white', fontsize=10)

    # Highlight significant rows
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        sig_t = row['p_value'] < alpha
        sig_w = row['p_value_wilcoxon'] < alpha
        positive_effect = row['Mean_Diff'] > 0

        if sig_t and sig_w and positive_effect:
            row_color = '#C6E0B4'
        elif sig_t and positive_effect:
            row_color = '#E2EFDA'
        elif positive_effect:
            row_color = '#F2F2F2'
        else:
            row_color = '#FCE4D6'

        for col in range(len(headers)):
            table[(idx, col)].set_facecolor(row_color)

        if sig_t:
            table[(idx, 7)].set_text_props(weight='bold')
        if sig_w:
            table[(idx, 8)].set_text_props(weight='bold')

    # Add legend
    legend_text = (
        "Highlighting Key:\n"
        "¢£ Dark Green: Both tests significant (p < 0.05) with positive convergence (parent adapts to child)\n"
        "¢£ Light Green: t-test significant with positive convergence\n"
        "¢£ Gray: Positive effect but not significant\n"
        "¢£ Orange: No convergence effect\n"
        "Bold p-values: Significant (p < 0.05)"
    )

    ax.text(0.5, -0.05, legend_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top', horizontalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(f'Statistical Significance of Parent Adaptation to Child\n{data_name}',
                fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(os.path.join(output_dir, 'reverse_imitation_significance_table.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

# Create the table
create_reverse_significance_table(reverse_sig_results, alpha=0.05)

# Print detailed results
print("\n" + "="*80)
print("SUMMARY OF STATISTICAL TESTS - PARENT ADAPTATION")
print("="*80)

for _, row in reverse_sig_results.iterrows():
    print(f"\n/{row['Phone']}/ - {row['Parent'].capitalize()} Adaptation:")
    print(f"  N pairs: {row['N']}")
    print(f"  Mean response distance: {row['Mean_Response']:.2f}")
    print(f"  Mean baseline distance: {row['Mean_Baseline']:.2f}")
    print(f"  Mean convergence effect: {row['Mean_Diff']:.2f}")
    print(f"  Cohen's d (effect size): {row['Cohens_d']:.3f}")
    print(f"  Convergence prevalence: {row['Conv_%']:.1f}%")
    print(f"  Paired t-test: t={row['t_stat']:.3f}, p={row['p_value']:.4f} {'***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else 'ns'}")
    print(f"  Wilcoxon test: p={row['p_value_wilcoxon']:.4f} {'***' if row['p_value_wilcoxon'] < 0.001 else '**' if row['p_value_wilcoxon'] < 0.01 else '*' if row['p_value_wilcoxon'] < 0.05 else 'ns'}")

# Summary statistics
print("\n" + "="*80)
print("OVERALL SUMMARY - PARENT ADAPTATION")
print("="*80)

print(f"\nTotal phone types tested: {len(reverse_sig_results)}")
print(f"Significant by t-test (p < 0.05): {reverse_sig_results['p_value'].lt(0.05).sum()} ({100*reverse_sig_results['p_value'].lt(0.05).mean():.1f}%)")
print(f"Significant by Wilcoxon (p < 0.05): {reverse_sig_results['p_value_wilcoxon'].lt(0.05).sum()} ({100*reverse_sig_results['p_value_wilcoxon'].lt(0.05).mean():.1f}%)")
print(f"Mean Cohen's d: {reverse_sig_results['Cohens_d'].mean():.3f}")
print(f"Mean convergence effect: {reverse_sig_results['Mean_Diff'].mean():.2f}")

# Save results
reverse_sig_results.to_csv(os.path.join(output_dir, 'reverse_imitation_significance_results.csv'), index=False)
print(f"\n? Saved results to: {os.path.join(output_dir, 'reverse_imitation_significance_results.csv')}")

print("\n" + "="*80)

# ========================== COMPARISON: CHILD VS PARENT ADAPTATION =========================
print("\n" + "="*80)
print("COMPARISON: CHILD IMITATION VS PARENT ADAPTATION")
print("="*80)

# Compare child imitation (forward) vs parent adaptation (reverse)
comparison_data = []

for phone in top_phones:
    for parent in ['mother', 'father']:
        # Get child imitation data
        child_imit = sig_results_df[
            (sig_results_df['Phone'] == phone) &
            (sig_results_df['Parent'] == parent)
        ]

        # Get parent adaptation data
        parent_adapt = reverse_sig_results[
            (reverse_sig_results['Phone'] == phone) &
            (reverse_sig_results['Parent'] == parent)
        ]

        if len(child_imit) > 0 and len(parent_adapt) > 0:
            comparison_data.append({
                'Phone': phone,
                'Parent': parent,
                'Child_Imitation_d': child_imit['Cohens_d'].values[0],
                'Parent_Adaptation_d': parent_adapt['Cohens_d'].values[0],
                'Child_Imitation_p': child_imit['p_value'].values[0],
                'Parent_Adaptation_p': parent_adapt['p_value'].values[0]
            })

comparison_df = pd.DataFrame(comparison_data)

# Visualization: Scatter plot comparing effect sizes
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for parent_idx, parent in enumerate(['mother', 'father']):
    ax = axes[parent_idx]
    parent_comp = comparison_df[comparison_df['Parent'] == parent]

    if len(parent_comp) == 0:
        continue

    # Color by significance
    colors = []
    for _, row in parent_comp.iterrows():
        child_sig = row['Child_Imitation_p'] < 0.05
        parent_sig = row['Parent_Adaptation_p'] < 0.05

        if child_sig and parent_sig:
            colors.append('green')
        elif child_sig:
            colors.append('blue')
        elif parent_sig:
            colors.append('red')
        else:
            colors.append('gray')

    ax.scatter(parent_comp['Child_Imitation_d'],
              parent_comp['Parent_Adaptation_d'],
              c=colors, s=150, alpha=0.7, edgecolors='black', linewidths=1.5)

    # Add phone labels
    for _, row in parent_comp.iterrows():
        ax.annotate(f"/{row['Phone']}/",
                   (row['Child_Imitation_d'], row['Parent_Adaptation_d']),
                   fontsize=9, ha='center', va='bottom')

    # Add reference lines
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.plot([-1, 2], [-1, 2], 'k:', alpha=0.3, label='Equal effect')

    ax.set_xlabel("Child Imitation Effect Size (Cohen's d)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Parent Adaptation Effect Size (Cohen's d)", fontsize=12, fontweight='bold')
    ax.set_title(f'{parent.capitalize()}: Child Imitation vs Parent Adaptation\n{data_name}',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Both significant'),
        Patch(facecolor='blue', edgecolor='black', label='Child only'),
        Patch(facecolor='red', edgecolor='black', label='Parent only'),
        Patch(facecolor='gray', edgecolor='black', label='Neither')
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc='upper left')

    # Add quadrant labels
    ax.text(0.95, 0.95, 'Mutual\nConvergence', transform=ax.transAxes,
           ha='right', va='top', fontsize=10, style='italic',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'child_vs_parent_adaptation_comparison.png'),
            dpi=150, bbox_inches='tight')
plt.close()

# Save comparison
comparison_df.to_csv(os.path.join(output_dir, 'child_parent_adaptation_comparison.csv'), index=False)
print(f"\n? Saved comparison to: {os.path.join(output_dir, 'child_parent_adaptation_comparison.csv')}")

print("\n" + "="*80)
print("COMPARISON ANALYSIS COMPLETE")
print("="*80)

# ========================== FINAL SUMMARY =========================
print("\n" + "="*80)
print("HUMAN NTT INTERACTION ANALYSIS - FINAL SUMMARY")
print("="*80)

print(f"\nAnalysis Configuration:")
print(f"  Data name: {data_name}")
print(f"  Age column: {age_col}")
print(f"  Age range: {min(age_values)} - {max(age_values)} {age_col.replace('age_', '')}")
print(f"  Total utterances: {len(utt_data)}")
print(f"  Total turn-taking pairs: {utt_data['has_response'].sum()}")

print(f"\nForward Imitation (Parent->Child):")
print(f"  Phones analyzed: {len(sig_results_df)}")
print(f"  Significant convergence: {sig_results_df['p_value'].lt(0.05).sum()}")
print(f"  Mean effect size (Cohen's d): {sig_results_df['Cohens_d'].mean():.3f}")

print(f"\nReverse Imitation (Child->Parent):")
print(f"  Phones analyzed: {len(reverse_sig_results)}")
print(f"  Significant convergence: {reverse_sig_results['p_value'].lt(0.05).sum()}")
print(f"  Mean effect size (Cohen's d): {reverse_sig_results['Cohens_d'].mean():.3f}")

print(f"\nAll results saved to: {output_dir}")

print("\n" + "="*80)
print("? ANALYSIS COMPLETE")
print("="*80)
# ```

# This complete script includes:

# 1. **Argument parsing** - All parameters configurable via command line
# 2. **Data loading** - Loads config, t-SNE, and speaker info
# 3. **Kana-to-phone conversion** - Using YAML mapping
# 4. **Label processing** - Multiple options (only_kana, all, etc.)
# 5. **Utterance-level aggregation** - Phone-level to utterance-level
# 6. **Response detection** - Turn-taking pair identification
# 7. **Trajectory analysis** - Vocal development visualization
# 8. **Response rate analysis** - Comprehensive statistics
# 9. **Forward imitation** - Parent->Child convergence
# 10. **Reverse imitation** - Child->Parent adaptation
# 11. **Statistical testing** - t-tests and Wilcoxon for both directions
# 12. **Comparison analysis** - Child vs Parent effects
# 13. **Complete output** - All plots and CSVs saved to output directory

# All outputs use the data_name in titles and are saved to the specified output_dir.

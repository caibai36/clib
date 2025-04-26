#!/usr/bin/env python3
"""
Data Division Tool for Speaker Identification

This script divides features into training, development, and test sets for speaker
identification tasks. It supports different feature files and speaker types.

Default features:
- exp/caller_identification/infant_vox_all.npy
- exp/caller_identification/parents_b3_0day_35days.npy

Default speakers:
- "infant"
- "adult"

Default split ratio:
- Training: 80%
- Development: 10%
- Test: 10%
"""

import os
import argparse
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='[ %(asctime)s | %(filename)s | %(levelname)s ] %(message)s',
    datefmt='%d/%m/%Y %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Divide data into training, development, and test sets.")

    parser.add_argument('--features', nargs='+', default=[
        'exp/caller_identification/infant_vox_all.npy',
        'exp/caller_identification/parents_b3_0day_35days.npy'
    ], help='List of feature files')

    parser.add_argument('--configs', nargs='+', default=[
        'exp/caller_identification/infant_vox_all_config.csv',
        'exp/caller_identification/parents_b3_0day_35days_config.csv'
    ], help='List of feature configuration files')

    parser.add_argument('--speakers', nargs='+', default=['infant', 'adult'],
                      help='Speaker labels corresponding to each feature file')

    parser.add_argument('--train-ratio', type=float, default=0.8,
                      help='Ratio of training data')

    parser.add_argument('--dev-ratio', type=float, default=0.1,
                      help='Ratio of development data')

    parser.add_argument('--test-ratio', type=float, default=0.1,
                      help='Ratio of test data')

    parser.add_argument('--output-dir', default='exp/caller_identification/data_division',
                      help='Output directory for saving the data division')

    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')

    return parser.parse_args()

def load_data(feature_path, config_path=None):
    """
    Load feature data and its configuration.

    Args:
        feature_path: Path to the feature file (.npy)
        config_path: Path to the configuration file (.csv)

    Returns:
        Tuple of (features, config)
    """
    logger.info(f"Loading features from {feature_path}")
    features = np.load(feature_path)

    config = None
    if config_path and os.path.exists(config_path):
        logger.info(f"Loading configuration from {config_path}")
        config = pd.read_csv(config_path)

        # Verify that features and config have matching lengths
        if len(features) != len(config):
            logger.warning(f"Mismatch between features ({len(features)}) and config ({len(config)}) lengths!")

    return features, config

def create_data_division(features_list, configs_list, speakers, train_ratio, dev_ratio, test_ratio, seed):
    """
    Create data division for training, development, and test sets.

    Args:
        features_list: List of feature arrays
        configs_list: List of configuration dataframes
        speakers: List of speaker labels
        train_ratio: Ratio for training set
        dev_ratio: Ratio for development set
        test_ratio: Ratio for test set
        seed: Random seed

    Returns:
        Dictionary containing training, development, and test data
    """
    assert len(features_list) == len(speakers), "Number of feature files must match number of speakers"

    # Initialize data containers
    train_features, dev_features, test_features = [], [], []
    train_configs, dev_configs, test_configs = [], [], []
    train_speakers, dev_speakers, test_speakers = [], [], []

    # Process each feature file and speaker type
    for i, (features, configs, speaker) in enumerate(zip(features_list, configs_list, speakers)):
        logger.info(f"Processing speaker '{speaker}' with {len(features)} samples")

        # First split: train vs. (dev+test)
        temp_ratio = dev_ratio + test_ratio
        X_train, X_temp, y_train, y_temp = train_test_split(
            features,
            np.arange(len(features)),
            test_size=temp_ratio,
            random_state=seed
        )

        # Second split: dev vs. test
        test_ratio_adjusted = test_ratio / temp_ratio
        X_dev, X_test, y_dev, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=test_ratio_adjusted,
            random_state=seed
        )

        # Add to containers
        train_features.append(X_train)
        dev_features.append(X_dev)
        test_features.append(X_test)

        train_speakers.extend([speaker] * len(X_train))
        dev_speakers.extend([speaker] * len(X_dev))
        test_speakers.extend([speaker] * len(X_test))

        # Handle configs if available
        if configs is not None:
            train_configs.append(configs.iloc[y_train].reset_index(drop=True))
            dev_configs.append(configs.iloc[y_dev].reset_index(drop=True))
            test_configs.append(configs.iloc[y_test].reset_index(drop=True))

            logger.info(f"  - Training: {len(X_train)} samples")
            logger.info(f"  - Development: {len(X_dev)} samples")
            logger.info(f"  - Test: {len(X_test)} samples")

    # Create data division dictionary
    data_division = {
        'train': {
            'features': train_features,
            'configs': train_configs if any(c is not None for c in train_configs) else None,
            'speakers': train_speakers
        },
        'dev': {
            'features': dev_features,
            'configs': dev_configs if any(c is not None for c in dev_configs) else None,
            'speakers': dev_speakers
        },
        'test': {
            'features': test_features,
            'configs': test_configs if any(c is not None for c in test_configs) else None,
            'speakers': test_speakers
        }
    }

    return data_division

def create_speaker_id_maps(speakers):
    """
    Create mappings between speaker names and numeric IDs.

    Args:
        speakers: List of unique speaker names

    Returns:
        Tuple of (speaker_to_id, id_to_speaker) dictionaries
    """
    unique_speakers = sorted(set(speakers))
    speaker_to_id = {speaker: i for i, speaker in enumerate(unique_speakers)}
    id_to_speaker = {i: speaker for i, speaker in enumerate(unique_speakers)}

    return speaker_to_id, id_to_speaker

def save_data_division(data_division, output_dir, speakers):
    """
    Save data division to files.

    Args:
        data_division: Dictionary containing training, development, and test data
        output_dir: Output directory
        speakers: List of speaker labels
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create speaker-to-id mappings
    unique_speakers = sorted(set(speakers))
    speaker_to_id, id_to_speaker = create_speaker_id_maps(unique_speakers)

    # Save speaker-to-id mapping to YAML files
    spk2id_path = os.path.join(output_dir, "spk2id.yaml")
    with open(spk2id_path, 'w') as f:
        yaml.dump(speaker_to_id, f, default_flow_style=False)
    logger.info(f"Saved speaker-to-id mapping to {spk2id_path}")

    id2spk_path = os.path.join(output_dir, "id2spk.yaml")
    with open(id2spk_path, 'w') as f:
        yaml.dump(id_to_speaker, f, default_flow_style=False)
    logger.info(f"Saved id-to-speaker mapping to {id2spk_path}")

    # Save data for each set
    for set_name in ['train', 'dev', 'test']:
        # Save features
        features_list = data_division[set_name]['features']
        all_features = np.vstack(features_list)
        feature_path = os.path.join(output_dir, f"{set_name}_features.npy")
        np.save(feature_path, all_features)

        # Convert speaker names to IDs
        speakers_list = data_division[set_name]['speakers']
        speaker_ids = [speaker_to_id[speaker] for speaker in speakers_list]

        # Save speaker IDs
        speakers_path = os.path.join(output_dir, f"{set_name}_speaker_ids.npy")
        np.save(speakers_path, np.array(speaker_ids))

        # Also save original speaker names for reference
        original_speakers_path = os.path.join(output_dir, f"{set_name}_speakers.npy")
        np.save(original_speakers_path, np.array(speakers_list))

        # Save configs if available
        configs_list = data_division[set_name]['configs']
        if configs_list and any(c is not None for c in configs_list):
            # Create a combined config with speaker columns
            combined_config = pd.concat(configs_list, ignore_index=True)
            combined_config['speaker'] = speakers_list
            combined_config['speaker_id'] = speaker_ids
            config_path = os.path.join(output_dir, f"{set_name}_config.csv")
            combined_config.to_csv(config_path, index=False)

        logger.info(f"Saved {set_name} set with {len(all_features)} samples to {output_dir}")

def main():
    """Main function to divide data for speaker identification."""
    args = parse_args()

    # Check if the ratios sum to 1
    total_ratio = args.train_ratio + args.dev_ratio + args.test_ratio
    if not np.isclose(total_ratio, 1.0):
        logger.warning(f"Sum of ratios ({total_ratio}) is not equal to 1.0!")

    # Load features and configs
    features_list = []
    configs_list = []

    for i, feature_path in enumerate(args.features):
        config_path = args.configs[i] if i < len(args.configs) else None
        features, configs = load_data(feature_path, config_path)
        features_list.append(features)
        configs_list.append(configs)

    # Create data division
    logger.info("Creating data division...")
    data_division = create_data_division(
        features_list,
        configs_list,
        args.speakers,
        args.train_ratio,
        args.dev_ratio,
        args.test_ratio,
        args.seed
    )

    # Save data division
    logger.info(f"Saving data division to {args.output_dir}")
    save_data_division(data_division, args.output_dir, args.speakers)

    logger.info("Data division completed successfully.")

if __name__ == "__main__":
    main()

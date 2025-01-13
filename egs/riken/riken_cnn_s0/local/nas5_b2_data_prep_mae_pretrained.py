#!/usr/bin/env python3
# coding: utf-8

"""
Feature Extraction and Dataset Preparation Script

This script processes a CSV file containing audio segment information to:
1. Extract spectrogram features from audio segments
2. Split data into train/dev/test sets based on age_days
3. Save features and labels in numpy format
4. Save configuration files matching input CSV format

The script uses age_days for dataset splitting and supports relative indexing
(e.g., -1 for last day, -2 for second to last day, etc.)
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import torchaudio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[ %(asctime)s | %(filename)s | %(levelname)s ] %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S"
)
logger = logging.getLogger(__name__)

def find_audio_path(root_path, audioid):
    """
    Find the full path to an audio file within a directory structure.

    Args:
        root_path (str): Root directory to start the search
        audioid (str): Audio ID to search for

    Returns:
        str or None: Full path to the audio file if found, None if not found
        If multiple matches are found, returns the first match with a warning

    Example:
        >>> find_audio_path("/data/audio", "240310_008_ch1")
        "/data/audio/folder/240310_008_ch1.wav"
    """
    matching_files = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            if audioid in filename and filename.endswith('.wav'):
                matching_files.append(os.path.join(dirpath, filename))

    if len(matching_files) == 0:
        return None
    elif len(matching_files) > 1:
        logger.warning(f"Multiple matches found for audioid {audioid}, using first match")
        for path in matching_files:
            logger.warning(f"  {path}")
        return matching_files[0]
    else:
        return matching_files[0]

def extract_audio_features(audio_path, begin_sec, end_sec, sample_rate=48000):
    """
    Extract spectrogram features from a specific segment of an audio file.

    Args:
        audio_path (str): Path to the audio file
        begin_sec (float): Start time in seconds
        end_sec (float): End time in seconds
        sample_rate (int): Target sample rate (default: 48000)

    Returns:
        np.ndarray: Spectrogram features of shape (257, 256) in float16 format

    The function:
    1. Loads the audio file
    2. Extracts the specified segment
    3. Resamples if necessary
    4. Computes the spectrogram
    5. Converts to dB scale
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load audio file
    data, rate = torchaudio.load(audio_path)
    data = data.to(device)

    # Convert time to samples
    start_index = int(rate * begin_sec)
    end_index = int(rate * end_sec)
    signal_piece = data[:, start_index:end_index]

    # Resample if necessary
    if rate != sample_rate:
        resampler = torchaudio.transforms.Resample(rate, sample_rate).to(device)
        signal_piece = resampler(signal_piece)

    # Create spectrogram
    nfft = 512
    window_size = end_sec - begin_sec
    hop_length = int((window_size*48000 - nfft) // (256 - 1))
    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=nfft,
        hop_length=hop_length,
        power=2,
        center=False
    ).to(device)

    spec = spectrogram_transform(signal_piece)
    spec_db = torchaudio.transforms.AmplitudeToDB().to(device)(spec)
    result = spec_db[0].detach().cpu().numpy()

    return result.astype(np.float16)

def process_dataset(df, audio_path_mapping):
    """
    Process a dataset DataFrame to extract features and prepare labels.

    Args:
        df (pd.DataFrame): DataFrame containing audio segment information
        audio_path_mapping (dict): Mapping from audio IDs to file paths

    Returns:
        tuple: (features, labels, config_df) where:
            - features: np.ndarray of spectrograms
            - labels: np.ndarray of corresponding labels
            - config_df: pd.DataFrame containing metadata for each feature

    The function processes each row in the DataFrame, extracting features
    and maintaining configuration information in the same format as input.
    """
    features = []
    labels = []
    config_info = []

    for idx, row in df.iterrows():
        # Get audio path from mapping
        audio_path = audio_path_mapping.get(row['audioid'])
        if audio_path is None:
            logger.warning(f"No audio file found for audioid: {row['audioid']}")
            continue

        try:
            # Extract features for this segment
            feature = extract_audio_features(
                audio_path,
                row['ext_cut_begin_sec'],
                row['ext_cut_end_sec']
            )

            # Verify feature shape
            if feature.shape != (257, 256):
                logger.warning(f"Invalid feature shape {feature.shape} for index {idx}")
                continue

            features.append(feature)
            labels.append(row['label'])
            config_info.append(row)

            # Log progress
            if len(features) % 100 == 0:
                logger.info(f"Processed {len(features)} samples")

        except Exception as e:
            logger.error(f"Error processing index {idx}: {str(e)}")
            continue

    return (np.array(features, dtype=np.float16),
            np.array(labels),
            pd.DataFrame(config_info))

def main():
    """
    Main function to run the feature extraction and dataset preparation process.

    Handles command line arguments, coordinates the overall process of:
    1. Loading the CSV file
    2. Creating audio path mapping
    3. Splitting the dataset
    4. Processing each split
    5. Saving features, labels, and configuration files
    """
    parser = argparse.ArgumentParser(description="""
    Extract features and prepare train/dev/test sets from CSV file.
    Uses age_days for splitting datasets, with support for relative indexing.
    Default split: last 10-2 days for training, day -2 for dev, last day for test.
    """)

    # Add command line arguments
    parser.add_argument("--csv_path", type=str,
                        default="data/nas5_b2_f1/b2_f1_cnn.csv",
                        help="Path to input CSV file")

    parser.add_argument("--audio_root", type=str,
                        default="/data01/share/bin-wu/data/marmoset/vocalization/riken_long/nas5/b2_1305F_759M_3162F",
                        help="Root directory containing audio files. Note that audioids in the csv should be unique in audio filenames as a substring under the root directory.")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for features and configuration")

    parser.add_argument("--train_start", type=int, default=-8,
                        help="Training set start day (relative to end)")

    parser.add_argument("--train_end", type=int, default=-2,
                        help="Training set end day (relative to end)")

    parser.add_argument("--dev_start", type=int, default=-2,
                        help="Dev set start day (relative to end)")

    parser.add_argument("--dev_end", type=int, default=-1,
                        help="Dev set end day (relative to end)")

    parser.add_argument("--test_start", type=int, default=-1,
                        help="Test set start day (relative to end)")

    parser.add_argument("--test_end", type=int, default=None,
                        help="Test set end day (relative to end)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Read CSV file
    df = pd.read_csv(args.csv_path)
    logger.info(f"Loaded CSV file with {len(df)} entries")

    # Create audio path mapping
    unique_audioids = df['audioid'].unique()
    audio_path_mapping = {}
    for audioid in unique_audioids:
        path = find_audio_path(args.audio_root, audioid)
        if path is None:
            logger.warning(f"No audio file found for audioid: {audioid}")
        else:
            audio_path_mapping[audioid] = path

    # Get unique days and sort them
    unique_days = sorted(df.age_days.unique())

    # Prepare dataset splits
    train_days = unique_days[args.train_start:args.train_end]
    dev_days = unique_days[args.dev_start:args.dev_end]
    test_days = unique_days[args.test_start:args.test_end]

    # Split dataframe by days
    train_df = df[df.age_days.isin(train_days)]
    dev_df = df[df.age_days.isin(dev_days)]
    test_df = df[df.age_days.isin(test_days)]

    # Process each dataset
    logger.info("Processing training set...")
    train_features, train_labels, train_config = process_dataset(train_df, audio_path_mapping)

    logger.info("Processing development set...")
    dev_features, dev_labels, dev_config = process_dataset(dev_df, audio_path_mapping)

    logger.info("Processing test set...")
    test_features, test_labels, test_config = process_dataset(test_df, audio_path_mapping)

    # Save features and labels
    np.save(os.path.join(args.output_dir, 'train_input.npy'), train_features)
    np.save(os.path.join(args.output_dir, 'train_target.npy'), train_labels)
    np.save(os.path.join(args.output_dir, 'dev_input.npy'), dev_features)
    np.save(os.path.join(args.output_dir, 'dev_target.npy'), dev_labels)
    np.save(os.path.join(args.output_dir, 'test_input.npy'), test_features)
    np.save(os.path.join(args.output_dir, 'test_target.npy'), test_labels)

    # Save configuration CSVs
    train_config.to_csv(os.path.join(args.output_dir, 'train_config.csv'), index=False)
    dev_config.to_csv(os.path.join(args.output_dir, 'dev_config.csv'), index=False)
    test_config.to_csv(os.path.join(args.output_dir, 'test_config.csv'), index=False)

    # Log summary information
    logger.info(f"Training set: {len(train_features)} samples, days {train_days}")
    logger.info(f"Development set: {len(dev_features)} samples, days {dev_days}")
    logger.info(f"Test set: {len(test_features)} samples, days {test_days}")
    logger.info('Data preparation completed successfully.')

if __name__ == "__main__":
    main()

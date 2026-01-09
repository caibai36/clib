#!/usr/bin/env python3
# coding: utf-8

"""
Feature Extraction and Dataset Preparation Script for NTT Infant Data

This script processes a CSV file containing audio segment information to:
1. Extract linear spectrogram features from 16kHz audio segments instead of mel spectrogram from 48kHz
2. Split data into train/dev/test sets based on age_months OR process entire dataset with --whole flag
3. Save features and labels in numpy format
4. Save configuration files matching input CSV format

The script uses age_months for dataset splitting and supports relative indexing
(e.g., -1 for last month, -2 for second to last month, etc.)
Extracts 257 x 256 linear spectrogram segments at 16kHz sample rate.
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import torchaudio
import torch.nn.functional as F

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
        >>> find_audio_path("/data/audio", "sa001_1")
        "/data/audio/sa/wav/sa001_1.wav"
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

def extract_linear_spectrogram_segment(audio_path, begin_sec, end_sec, window_size=0.5, nfft=512, sample_rate=16000):
    """
    Extract linear spectrogram features from a specific segment of an audio file.
    Following the approach from riken_mae_vit_audio2seg_highres_feat_batch_padding_lin16k_yaml.py

    Args:
        audio_path (str): Path to the audio file
        begin_sec (float): Start time in seconds
        end_sec (float): End time in seconds
        window_size (float): Window size in seconds (default: 0.5)
        nfft (int): Number of FFT points (default: 512)
        sample_rate (int): Target sample rate (default: 16000)

    Returns:
        np.ndarray: Linear spectrogram features of shape (257, 256) in float16 format
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load audio file
    data, rate = torchaudio.load(audio_path)
    data = data.to(device)

    # Resample to 16kHz if necessary
    if rate != sample_rate:
        resampler = torchaudio.transforms.Resample(rate, sample_rate).to(device)
        data = resampler(data)

    # Convert time to samples and extract segment
    start_index = int(sample_rate * begin_sec)
    end_index = int(sample_rate * end_sec)
    signal_piece = data[:, start_index:end_index]

    # Calculate hop_length for 256 time frames following the reference
    hop_length = int((window_size * sample_rate - nfft) // (256 - 1))

    # Create linear spectrogram transform
    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=nfft,
        hop_length=hop_length,
        power=2,
        center=False
    ).to(device)

    # Create amplitude to dB transform
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(device)

    # Compute spectrogram and convert to dB
    spec = spectrogram_transform(signal_piece)
    spec_db = amplitude_to_db(spec)
    result = spec_db[0].detach().cpu().numpy()

    # Crop to ensure exactly 256 time frames
    if result.shape[1] > 256:
        result = result[:, :256]

    return result.astype(np.float16)

def extract_linear_spectrograms_batch(audio_paths, begin_secs, end_secs, batch_size=32, window_size=0.5, nfft=512, sample_rate=16000, cache_size=100):
    """
    Extract linear spectrogram features from audio segments in batches with audio caching.
    Following the batch processing approach from riken_mae_vit_audio2seg_highres_feat_batch_padding_lin16k_yaml.py

    Args:
        audio_paths (list): List of paths to audio files
        begin_secs (list): List of start times in seconds for each audio file
        end_secs (list): List of end times in seconds for each audio file
        batch_size (int): Number of samples to process at once (default: 32)
        window_size (float): Window size in seconds (default: 0.5)
        nfft (int): Number of FFT points (default: 512)
        sample_rate (int): Target sample rate for 16kHz processing (default: 16000)
        cache_size (int): Maximum number of audio files to keep in memory (default: 100)

    Returns:
        list: List of linear spectrogram features, each of shape (257, 256) in float16 format.
    """
    # Set up GPU/CPU device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        logger.info(f"Using {torch.cuda.get_device_name(device)} for processing")
    else:
        logger.info("Using CPU for processing")

    # Initialize LRU cache for storing loaded audio tensors
    from collections import OrderedDict
    audio_cache = OrderedDict()

    # Calculate hop_length for 256 time frames following the reference implementation
    hop_length = int((window_size * sample_rate - nfft) // (256 - 1))

    # Initialize transforms
    spectrogram_transform = torchaudio.transforms.Spectrogram(
        n_fft=nfft,
        hop_length=hop_length,
        power=2,
        center=False
    ).to(device)
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(device)

    features = []
    logger.info(f"Starting batch processing with batch size {batch_size}")

    # Process audio files in batches
    for i in range(0, len(audio_paths), batch_size):
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(audio_paths) + batch_size - 1)//batch_size}")

        # Get current batch
        batch_paths = audio_paths[i:i + batch_size]
        batch_begins = begin_secs[i:i + batch_size]
        batch_ends = end_secs[i:i + batch_size]

        batch_signals = []

        # Load and process audio for each sample in batch
        for path, begin, end in zip(batch_paths, batch_begins, batch_ends):
            # Try to retrieve audio from cache first
            if path in audio_cache:
                data = audio_cache[path]
                # Move to end of cache (most recently used)
                audio_cache.move_to_end(path)
            else:
                # Load audio and resample to 16kHz if needed
                data, rate = torchaudio.load(path)
                if rate != sample_rate:
                    resampler = torchaudio.transforms.Resample(rate, sample_rate).to(device)
                    data = resampler(data)

                # Add to cache
                audio_cache[path] = data

                # Remove oldest cache entry if cache is full
                if len(audio_cache) > cache_size:
                    audio_cache.popitem(last=False)

            # Extract the requested segment from audio
            start_index = int(sample_rate * begin)
            end_index = int(sample_rate * end)
            signal_piece = data[:, start_index:end_index]

            # Ensure signal has expected length for window_size
            expected_length = int(sample_rate * window_size)
            if signal_piece.shape[1] < expected_length:
                # Pad if too short
                pad_length = expected_length - signal_piece.shape[1]
                signal_piece = F.pad(signal_piece, (0, pad_length), mode='constant', value=0)
            elif signal_piece.shape[1] > expected_length:
                # Crop if too long
                signal_piece = signal_piece[:, :expected_length]

            batch_signals.append(signal_piece)

        if not batch_signals:
            logger.info(f"No valid segments in batch {i//batch_size + 1}. Skipping.")
            continue

        # Stack all signals in batch and move to device
        batch_tensor = torch.cat(batch_signals, dim=0).to(device)

        # Compute linear spectrograms for the batch
        logger.info(f"Computing linear spectrograms for batch {i//batch_size + 1}")
        specs = spectrogram_transform(batch_tensor)
        specs_db = amplitude_to_db(specs)

        # Process each spectrogram in the batch
        batch_features = specs_db.detach().cpu().numpy()
        for spec in batch_features:
            # Crop to ensure exactly 256 time frames
            if spec.shape[1] > 256:
                spec = spec[:, :256]

            # Validate shape
            if spec.shape != (257, 256):
                logger.warning(f"Invalid spectrogram shape: {spec.shape}, expected (257, 256)")
                continue

            features.append(spec.astype(np.float16))

        logger.info(f"Completed batch {i//batch_size + 1}, processed {len(features)} samples total")

    logger.info(f"Feature extraction completed for all {len(features)} samples")
    return features

def process_dataset(df, audio_path_mapping, batch_size=32):
    """
    Process a dataset DataFrame to extract features and prepare labels using batch processing.

    Args:
        df (pd.DataFrame): DataFrame containing audio segment information including:
            - audioid: unique identifier for audio files
            - ext_cut_begin_sec: start time of segment
            - ext_cut_end_sec: end time of segment
            - label: target label for the segment
        audio_path_mapping (dict): Mapping from audio IDs to file paths
        batch_size (int): Number of samples to process simultaneously (default: 32)

    Returns:
        tuple: (features, labels, config_df) where:
            - features: np.ndarray of linear spectrograms (N, 257, 256) in float16
            - labels: np.ndarray of corresponding labels
            - config_df: pd.DataFrame containing metadata for valid samples
    """
    # Reset the DataFrame index to ensure continuous indexing
    df = df.reset_index(drop=True)

    audio_paths = []
    begin_secs = []
    end_secs = []
    valid_indices = []
    labels = []

    for idx, row in df.iterrows():
        audio_path = audio_path_mapping.get(row['audioid'])
        if audio_path is None:
            logger.warning(f"No audio file found for audioid: {row['audioid']}")
            continue

        # Round time values to handle floating point errors
        begin_sec = round(row['ext_cut_begin_sec'], 6)
        end_sec = round(row['ext_cut_end_sec'], 6)

        audio_paths.append(audio_path)
        begin_secs.append(begin_sec)
        end_secs.append(end_sec)
        valid_indices.append(idx)
        labels.append(row['label'])

    logger.info(f"Preparing to process {len(audio_paths)} samples")

    # Extract features using batch processing
    features = extract_linear_spectrograms_batch(
        audio_paths,
        begin_secs,
        end_secs,
        batch_size=batch_size
    )

    # Validate features and prepare output
    valid_features = []
    valid_labels = []
    valid_rows = []

    for feature, label, idx in zip(features, labels, valid_indices):
        if feature.shape == (257, 256):
            valid_features.append(feature)
            valid_labels.append(label)
            valid_rows.append(df.loc[idx])
        else:
            logger.warning(f"Invalid feature shape {feature.shape} for index {idx}, expected (257, 256)")

    logger.info(f"Successfully processed {len(valid_features)} valid samples")

    return (np.array(valid_features, dtype=np.float16),
            np.array(valid_labels),
            pd.DataFrame(valid_rows))

def main():
    """
    Main function to run the feature extraction and dataset preparation process.

    Handles command line arguments, coordinates the overall process of:
    1. Loading the CSV file
    2. Creating audio path mapping
    3. Splitting the dataset by age_months OR processing entire dataset with --whole flag
    4. Processing each split (or entire dataset)
    5. Saving features, labels, and configuration files
    """
    parser = argparse.ArgumentParser(description="""
    Extract linear spectrogram features and prepare train/dev/test sets from CSV file for NTT infant data.
    Uses age_months for splitting datasets, with support for relative indexing.
    Default split: last 10-2 months for training, month -2 for dev, last month for test.
    Processes 16kHz audio to extract 257x256 linear spectrograms.
    Use --whole to process entire dataset without splitting.
    """)

    # Add command line arguments
    parser.add_argument("--csv_path", type=str,
                        default="data/ntt_infant_phone/metas/meta_sa.csv",
                        help="Path to input CSV file")

    parser.add_argument("--audio_root", type=str,
                        default="/data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/processed/ntt_infant_data",
                        help="Root directory containing audio files. Note that audioids in the csv should be unique in audio filenames as a substring under the root directory.")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for features and configuration")

    parser.add_argument("--whole", action='store_true',
                        help="Process entire dataset without splitting into train/dev/test")

    parser.add_argument("--whole_begin", type=int, default=None,
                        help="Start month for whole dataset processing (relative to end, default: None = use all)")

    parser.add_argument("--whole_end", type=int, default=None,
                        help="End month for whole dataset processing (relative to end, default: None = use all)")

    parser.add_argument("--train_start", type=int, default=-8,
                        help="Training set start month (relative to end)")

    parser.add_argument("--train_end", type=int, default=-2,
                        help="Training set end month (relative to end)")

    parser.add_argument("--dev_start", type=int, default=-2,
                        help="Dev set start month (relative to end)")

    parser.add_argument("--dev_end", type=int, default=-1,
                        help="Dev set end month (relative to end)")

    parser.add_argument("--test_start", type=int, default=-1,
                        help="Test set start month (relative to end)")

    parser.add_argument("--test_end", type=int, default=None,
                        help="Test set end month (relative to end)")

    # Add batch size argument
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for feature extraction")

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

    if args.whole:
        # Process entire dataset without splitting (with optional age filtering)
        logger.info("Processing entire dataset (--whole mode)...")

        # Apply age filtering if whole_begin or whole_end is specified
        if args.whole_begin is not None or args.whole_end is not None:
            unique_months = sorted(df.age_months.unique())
            whole_months = unique_months[args.whole_begin:args.whole_end]
            df = df[df.age_months.isin(whole_months)]
            logger.info(f"Filtered to months: {whole_months}")

        features, labels, config_df = process_dataset(
            df, audio_path_mapping, batch_size=args.batch_size)

        # Save features and configuration
        np.save(os.path.join(args.output_dir, 'spec_raw.npy'), features)
        config_df.to_csv(os.path.join(args.output_dir, 'info.csv'), index=False)

        # Log summary information
        logger.info(f"Total samples processed: {len(features)}")
        logger.info(f"Output files: spec_raw.npy, info.csv")
        logger.info('Data preparation completed successfully.')

    else:
        # Original split-based processing
        # Get unique months and sort them
        unique_months = sorted(df.age_months.unique())

        # Prepare dataset splits by months
        train_months = unique_months[args.train_start:args.train_end]
        dev_months = unique_months[args.dev_start:args.dev_end]
        test_months = unique_months[args.test_start:args.test_end]

        # Split dataframe by months
        train_df = df[df.age_months.isin(train_months)]
        dev_df = df[df.age_months.isin(dev_months)]
        test_df = df[df.age_months.isin(test_months)]

        # Process each dataset split
        logger.info("Processing training set...")
        train_features, train_labels, train_config = process_dataset(
            train_df, audio_path_mapping, batch_size=args.batch_size)

        logger.info("Processing development set...")
        dev_features, dev_labels, dev_config = process_dataset(
            dev_df, audio_path_mapping, batch_size=args.batch_size)

        logger.info("Processing test set...")
        test_features, test_labels, test_config = process_dataset(
            test_df, audio_path_mapping, batch_size=args.batch_size)

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
        logger.info(f"Training set: {len(train_features)} samples, months {train_months}")
        logger.info(f"Development set: {len(dev_features)} samples, months {dev_months}")
        logger.info(f"Test set: {len(test_features)} samples, months {test_months}")
        logger.info('Data preparation completed successfully.')

if __name__ == "__main__":
    main()

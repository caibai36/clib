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
import GPUtil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[ %(asctime)s | %(filename)s | %(levelname)s ] %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S"
)
logger = logging.getLogger(__name__)

def set_device(gpu):
    """
    Determine the device (GPU or CPU) based on the specified GPU argument.
    """
    if gpu == 'auto':
        # Get the available GPU with the least memory usage
        available_gpus = GPUtil.getAvailable(order='memory')
        if available_gpus:
            device = torch.device(f"cuda:{available_gpus[0]}")
        else:
            device = torch.device("cpu")
    else:
        # Use the specified GPU device if available, otherwise use CPU
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    return device

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

def extract_audio_features(audio_path, begin_sec, end_sec, sample_rate=48000, device=None):
    """
    Extract spectrogram features from a specific segment of an audio file.

    Args:
        audio_path (str): Path to the audio file
        begin_sec (float): Start time in seconds
        end_sec (float): End time in seconds
        sample_rate (int): Target sample rate (default: 48000)
        device (torch.device): Device to use for computation

    Returns:
        np.ndarray: Spectrogram features of shape (257, 256) in float16 format

    The function:
    1. Loads the audio file
    2. Extracts the specified segment
    3. Resamples if necessary
    4. Computes the spectrogram
    5. Converts to dB scale
    """
    if device is None:
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

def extract_audio_features_batch(audio_paths, begin_secs, end_secs, batch_size=32, sample_rate=48000, cache_size=100, device=None):
    """
    Extract spectrogram features from audio segments in batches with audio caching.

    This function processes multiple audio files simultaneously to improve performance.
    Python's list slicing naturally handles the last incomplete batch, ensuring all
    samples are processed correctly regardless of total sample count.
    All signals should have the same length due to constant window size.

    Uses LRU caching to avoid repeatedly loading the same audio files from disk.
    Previously loaded audio tensors are stored in memory and reused when needed,
    with oldest entries being removed when cache reaches capacity.

    Args:
        audio_paths (list): List of paths to audio files
        begin_secs (list): List of start times in seconds for each audio file
        end_secs (list): List of end times in seconds for each audio file
        batch_size (int): Number of samples to process at once (default: 32)
        sample_rate (int): Target sample rate for all audio (default: 48000)
        cache_size (int): Maximum number of audio files to keep in memory (default: 100)
        device (torch.device): Device to use for computation

    Returns:
        tuple: (features, valid_indices) where:
            - features: List of spectrogram features, each of shape (257, 256) in float16 format
            - valid_indices: List of indices that were successfully processed
    """
    # Set up GPU/CPU device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        logger.info(f"Using {torch.cuda.get_device_name(device)} for processing")
    else:
        logger.info("Using CPU for processing")

    # Initialize LRU cache for storing loaded audio tensors
    from collections import OrderedDict
    audio_cache = OrderedDict()

    features = []
    valid_indices = []
    logger.info(f"Starting batch processing with batch size {batch_size}")

    # Process audio files in batches
    for i in range(0, len(audio_paths), batch_size):
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(audio_paths) + batch_size - 1)//batch_size}")

        # Get current batch (list slicing handles incomplete last batch automatically)
        batch_paths = audio_paths[i:i + batch_size]
        batch_begins = begin_secs[i:i + batch_size]
        batch_ends = end_secs[i:i + batch_size]

        batch_signals = []
        batch_valid_indices = []

        # Load audio and verify signal lengths
        expected_length = None
        for local_idx, (path, begin, end) in enumerate(zip(batch_paths, batch_begins, batch_ends)):
            global_idx = i + local_idx

            try:
                # Try to retrieve audio from cache first
                if path in audio_cache:
                    data = audio_cache[path]
                    # Move to end of cache (most recently used)
                    audio_cache.move_to_end(path)
                else:
                    # Load audio and resample if needed
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
                start_index = round(sample_rate * begin)
                end_index = round(sample_rate * end)

                # Validate segment bounds
                audio_length = data.shape[1]
                if start_index >= audio_length:
                    logger.error(f"Start index {start_index} >= audio length {audio_length} for {path}")
                    logger.error(f"  begin_sec: {begin}, end_sec: {end}, audio_duration: {audio_length/sample_rate:.3f}s")
                    continue

                if end_index > audio_length:
                    logger.warning(f"End index {end_index} > audio length {audio_length} for {path}, clipping to audio end")
                    end_index = audio_length

                if start_index >= end_index:
                    logger.error(f"Invalid segment: start_index {start_index} >= end_index {end_index} for {path}")
                    logger.error(f"  begin_sec: {begin}, end_sec: {end}")
                    continue

                signal_piece = data[:, start_index:end_index]

                # Check if signal is empty
                if signal_piece.shape[1] == 0:
                    logger.error(f"Empty signal extracted from {path}")
                    logger.error(f"  begin_sec: {begin}, end_sec: {end}, start_index: {start_index}, end_index: {end_index}")
                    continue

                # Verify all signals have same length
                if expected_length is None:
                    expected_length = signal_piece.shape[1]
                else:
                    if signal_piece.shape[1] != expected_length:
                        logger.error(f"Signal length mismatch for {path}")
                        logger.error(f"  Expected: {expected_length}, got: {signal_piece.shape[1]}")
                        logger.error(f"  begin_sec: {begin}, end_sec: {end}, duration: {end-begin:.6f}s")
                        continue

                batch_signals.append(signal_piece)
                batch_valid_indices.append(global_idx)

            except Exception as e:
                logger.error(f"Error processing {path}: {str(e)}")
                continue

        # Skip batch if no valid signals
        if len(batch_signals) == 0:
            logger.warning(f"No valid signals in batch {i//batch_size + 1}, skipping")
            continue

        try:
            # Stack all signals in batch and move to GPU/CPU
            batch_tensor = torch.stack(batch_signals).to(device)

            # Create spectrogram
            logger.info(f"Computing spectrograms for {len(batch_signals)} valid samples")
            nfft = 512
            window_size = batch_ends[0] - batch_begins[0]  # Use first valid sample for window size
            hop_length = int((window_size*48000 - nfft) // (256 - 1))
            spectrogram_transform = torchaudio.transforms.Spectrogram(
                n_fft=nfft,
                hop_length=hop_length,
                power=2,
                center=False
            ).to(device)

            # Compute spectrograms and convert to dB scale
            specs = spectrogram_transform(batch_tensor)
            specs_db = torchaudio.transforms.AmplitudeToDB().to(device)(specs)

            # Store results
            batch_features = specs_db.detach().cpu().numpy()
            features.extend([feat[0].astype(np.float16) for feat in batch_features])
            valid_indices.extend(batch_valid_indices)

            logger.info(f"Completed batch {i//batch_size + 1}, processed {len(features)} samples total")

        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            continue

    logger.info(f"Feature extraction completed for {len(features)} samples out of {len(audio_paths)} total")
    return features, valid_indices

def process_dataset(df, audio_path_mapping, batch_size=32, device=None):
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
        device (torch.device): Device to use for computation

    Returns:
        tuple: (features, labels, config_df) where:
            - features: np.ndarray of spectrograms (N, 257, 256) in float16
            - labels: np.ndarray of corresponding labels
            - config_df: pd.DataFrame containing metadata for valid samples
    Note:
        Rounds time values to 6 decimal places to handle floating point rounding errors in CSV.
    """
    # Reset the DataFrame index to ensure continuous indexing
    df = df.reset_index(drop=True)

    audio_paths = []
    begin_secs = []
    end_secs = []
    valid_df_indices = []
    labels = []

    for idx, row in df.iterrows():
        audio_path = audio_path_mapping.get(row['audioid'])
        if audio_path is None:
            logger.warning(f"No audio file found for audioid: {row['audioid']}")
            continue

        # Round time values to handle floating point errors
        begin_sec = round(row['ext_cut_begin_sec'], 6)
        end_sec = round(row['ext_cut_end_sec'], 6)

        # Validate time bounds
        if begin_sec >= end_sec:
            logger.warning(f"Invalid time bounds: begin_sec {begin_sec} >= end_sec {end_sec} for row {idx}")
            continue

        audio_paths.append(audio_path)
        begin_secs.append(begin_sec)
        end_secs.append(end_sec)
        valid_df_indices.append(idx)
        labels.append(row['label'])

    logger.info(f"Preparing to process {len(audio_paths)} samples")

    # Extract features using batch processing
    features, processed_indices = extract_audio_features_batch(
        audio_paths,
        begin_secs,
        end_secs,
        batch_size=batch_size,
        device=device
    )

    # Match processed features with their corresponding labels and DataFrame rows
    valid_features = []
    valid_labels = []
    valid_rows = []

    for feature, processed_idx in zip(features, processed_indices):
        if feature.shape == (257, 256):
            valid_features.append(feature)
            valid_labels.append(labels[processed_idx])
            valid_rows.append(df.loc[valid_df_indices[processed_idx]])
        else:
            logger.warning(f"Invalid feature shape {feature.shape} for processed index {processed_idx}, expected (257, 256)")

    logger.info(f"Successfully processed {len(valid_features)} valid samples out of {len(df)} total samples")

    return (np.array(valid_features, dtype=np.float16),
            np.array(valid_labels),
            pd.DataFrame(valid_rows))

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

    # Add batch size argument
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for feature extraction")

    # Add GPU argument
    parser.add_argument('--gpu', type=str, default='auto',
                        help="GPU selection: number for specific GPU or 'auto' for least used")

    args = parser.parse_args()

    # Set device
    device = set_device(args.gpu)
    logger.info(f"Using device: {device}")

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

    # Process datasets
    logger.info("Processing training set...")
    train_features, train_labels, train_config = process_dataset(
        train_df, audio_path_mapping, batch_size=args.batch_size, device=device)

    logger.info("Processing development set...")
    dev_features, dev_labels, dev_config = process_dataset(
        dev_df, audio_path_mapping, batch_size=args.batch_size, device=device)

    logger.info("Processing test set...")
    test_features, test_labels, test_config = process_dataset(
        test_df, audio_path_mapping, batch_size=args.batch_size, device=device)

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

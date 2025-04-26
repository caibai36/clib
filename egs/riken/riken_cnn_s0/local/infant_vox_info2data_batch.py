#!/usr/bin/env python3
# coding: utf-8

"""
Feature Extraction Script for Infant Marmoset Vocalization Data

This script processes a CSV file containing audio segment information to:
1. Extract spectrogram features from audio segments
2. Save features and labels in numpy format
3. Save configuration file matching input CSV format
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
def extract_audio_features_batch(audio_paths, begin_secs, end_secs, batch_size=32, sample_rate=48000, cache_size=100):
    """
    Extract spectrogram features from audio segments in batches with audio caching.

    Args:
        audio_paths (list): List of paths to audio files
        begin_secs (list): List of start times in seconds for each audio file
        end_secs (list): List of end times in seconds for each audio file
        batch_size (int): Number of samples to process at once (default: 32)
        sample_rate (int): Target sample rate for all audio (default: 48000)
        cache_size (int): Maximum number of audio files to keep in memory (default: 100)

    Returns:
        list: List of spectrogram features, each of shape (257, 256) in float16 format
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

    features = []
    logger.info(f"Starting batch processing with batch size {batch_size}")

    # Process audio files in batches
    for i in range(0, len(audio_paths), batch_size):
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(audio_paths) + batch_size - 1)//batch_size}")

        # Get current batch (list slicing handles incomplete last batch automatically)
        batch_paths = audio_paths[i:i + batch_size]
        batch_begins = begin_secs[i:i + batch_size]
        batch_ends = end_secs[i:i + batch_size]

        batch_signals = []
        valid_indices = []  # Keep track of valid signal indices for this batch

        # Load audio and verify signal lengths
        expected_length = None
        for j, (path, begin, end) in enumerate(zip(batch_paths, batch_begins, batch_ends)):
            # Try to retrieve audio from cache first
            if path in audio_cache:
                data = audio_cache[path]
                # Move to end of cache (most recently used)
                audio_cache.move_to_end(path)
            else:
                try:
                    # Load audio and resample if needed
                    data, rate = torchaudio.load(path)
                    
                    # Move data to appropriate device before resampling
                    data = data.to(device)
                    
                    if rate != sample_rate:
                        resampler = torchaudio.transforms.Resample(rate, sample_rate).to(device)
                        data = resampler(data)

                    # Add to cache
                    audio_cache[path] = data

                    # Remove oldest cache entry if cache is full
                    if len(audio_cache) > cache_size:
                        audio_cache.popitem(last=False)
                except Exception as e:
                    logger.warning(f"Error loading audio file {path}: {str(e)}")
                    continue

            # Extract the requested segment from audio
            start_index = round(sample_rate * begin)
            end_index = round(sample_rate * end)
            
            # Skip segments that would result in zero length
            if start_index >= end_index or start_index >= data.shape[1] or end_index <= 0:
                logger.warning(f"Invalid segment time [{begin}-{end}] for file {path} (shape: {data.shape})")
                continue
                
            # Adjust indices to be within audio bounds
            start_index = max(0, start_index)
            end_index = min(data.shape[1], end_index)
            
            signal_piece = data[:, start_index:end_index]
            
            # Skip zero-length segments
            if signal_piece.shape[1] == 0:
                logger.warning(f"Zero-length segment for file {path} at [{begin}-{end}]")
                continue

            # Set expected length from first valid segment if not already set
            if expected_length is None:
                expected_length = signal_piece.shape[1]
                
            # Skip segments with different lengths
            if signal_piece.shape[1] != expected_length:
                logger.warning(f"Signal length mismatch: expected {expected_length}, got {signal_piece.shape[1]} for file {path}")
                continue

            batch_signals.append(signal_piece)
            valid_indices.append(j)  # Track the index in the original batch

        # Skip if no valid signals in this batch
        if not batch_signals:
            logger.warning(f"No valid audio segments in batch {i//batch_size + 1}, skipping")
            # Add placeholders for skipped samples
            features.extend([None] * len(batch_paths))
            continue

        # Stack all signals in batch (already on the device)
        batch_tensor = torch.stack(batch_signals)

        # Create spectrogram
        logger.info(f"Computing spectrograms for {len(batch_signals)} valid segments")
        nfft = 512
        window_size = batch_ends[valid_indices[0]] - batch_begins[valid_indices[0]]
        hop_length = int((window_size*sample_rate - nfft) // (256 - 1))
        
        # Ensure hop_length is positive
        if hop_length <= 0:
            logger.warning(f"Invalid hop_length {hop_length}, using default value 1")
            hop_length = 1
            
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
        
        # Create a list of features for this batch, with None for invalid segments
        batch_results = [None] * len(batch_paths)
        for valid_idx, feat_idx in enumerate(valid_indices):
            batch_results[feat_idx] = batch_features[valid_idx][0].astype(np.float16)
            
        # Add the results to the overall features list
        features.extend(batch_results)

        logger.info(f"Completed batch {i//batch_size + 1}, processed {sum(1 for f in features if f is not None)} valid samples total")

    logger.info(f"Feature extraction completed with {sum(1 for f in features if f is not None)} valid samples")
    return features

def process_dataset(df, audio_path_mapping, batch_size=32):
    """
    Process a dataset DataFrame to extract features and prepare labels.

    Args:
        df (pd.DataFrame): DataFrame containing audio segment information
        audio_path_mapping (dict): Mapping from audio IDs to file paths
        batch_size (int): Number of samples to process simultaneously (default: 32)

    Returns:
        tuple: (features, labels, config_df) where:
            - features: np.ndarray of spectrograms (N, 257, 256) in float16
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

        # Use the appropriate column names from infant_vox.csv
        begin_sec = round(row['ext_cut_begin_sec'], 6)
        end_sec = round(row['ext_cut_end_sec'], 6)

        audio_paths.append(audio_path)
        begin_secs.append(begin_sec)
        end_secs.append(end_sec)
        valid_indices.append(idx)
        labels.append(row['caller'])  # Using 'caller' column as label

    logger.info(f"Preparing to process {len(audio_paths)} samples")

    # Extract features using batch processing
    features = extract_audio_features_batch(
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
        if feature is not None and feature.shape == (257, 256):
            valid_features.append(feature)
            valid_labels.append(label)
            valid_rows.append(df.loc[idx])  # Use loc instead of iloc
        elif feature is not None:
            logger.warning(f"Invalid feature shape {feature.shape} for index {idx}, expected (257, 256)")

    logger.info(f"Successfully processed {len(valid_features)} valid samples")

    return (np.array(valid_features, dtype=np.float16),
            np.array(valid_labels),
            pd.DataFrame(valid_rows))

def main():
    """
    Main function to run the feature extraction and dataset preparation process.
    """
    parser = argparse.ArgumentParser(description="""
    Extract features from infant marmoset vocalization data.
    Processes all data in the CSV file without splitting into train/dev/test sets.
    """)

    # Add command line arguments
    parser.add_argument("--csv_path", type=str,
                        default="data/infant_vox/infant_vox.csv",
                        help="Path to input CSV file")

    parser.add_argument("--audio_root", type=str,
                        default="/data/share/bin-wu/data/marmoset/vocalization/marmoset_vox/original/InfantMarmosetsVox/data",
                        help="Root directory containing audio files")

    parser.add_argument("--output_feat", type=str,
                        default="exp/caller_identification/infant_vox_all.npy",
                        help="Output path for feature array")

    parser.add_argument("--output_config", type=str,
                        default="exp/caller_identification/infant_vox_all_config.csv",
                        help="Output path for configuration CSV")

    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for feature extraction")

    args = parser.parse_args()

    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(args.output_feat), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_config), exist_ok=True)

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

    # Process the entire dataset
    logger.info("Processing dataset...")
    features, labels, config_df = process_dataset(
        df, audio_path_mapping, batch_size=args.batch_size)

    # Save features only
    np.save(args.output_feat, features)
    logger.info(f"Saved features to {args.output_feat}")

    # Save configuration CSV
    config_df.to_csv(args.output_config, index=False)
    logger.info(f"Saved configuration to {args.output_config}")

    # Log summary information
    logger.info(f"Processed dataset: {len(features)} samples")
    logger.info('Data preparation completed successfully.')

if __name__ == "__main__":
    main()

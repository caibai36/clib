#!/usr/bin/env python3
"""
Extract MAE features from marmoset dataset CSV files.

This script:
1. Reads CSV file with segment information
2. Extracts spectrogram features (spec_raw.npy)
3. Extracts MAE features using pretrained model (mae_raw.npy)
4. Saves info.csv matching the feature arrays

Usage with wav.scp:
    python local/extract_mae_feature_from_csv.py \
        --csv_path data/exp1/exp1.csv \
        --wav_scp data/local/Experiment_1/wav.scp \
        --output_dir exp/sandbox/exp1 \
        --mae_model conf/model/mae_pretrained_model_epoch_400_base_48days.pt

Usage with wav_dir:
    python local/extract_mae_feature_from_csv.py \
        --csv_path data/b3_762F_763M_3201M/b3_762F_763M_3201M.csv \
        --wav_dir /path/to/wav/ \
        --output_dir exp/sandbox/b3_762F_763M_3201M \
        --mae_model conf/model/mae_pretrained_model_epoch_400_base_48days.pt

Implemented by bin-wu 20251216

Note:
Main difference from riken_mae_pretrained_feature_extraction.py
 Adding 1e-8: x = (x - x.min()) / (x.max() - x.min() + 1e-8)
 Using np.float16: np.save(mae_output_path, mae_features.astype(np.float16))
 Assume audio feature exactration same
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
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict

from transformers import ViTMAEModel, ViTMAEConfig
from torchvision import transforms

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[ %(asctime)s | %(filename)s | %(levelname)s ] %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Reuse functions from the original script
def set_seed(seed):
    """
    Set random seeds for reproducibility across NumPy and PyTorch.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_device(gpu):
    """Determine the device (GPU or CPU) based on the specified GPU argument."""
    if gpu == 'auto':
        available_gpus = GPUtil.getAvailable(order='memory')
        if available_gpus:
            device = torch.device(f"cuda:{available_gpus[0]}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    return device

def load_wav_scp(wav_scp_path):
    """
    Load wav.scp file and create mapping from audioid to file path.

    Args:
        wav_scp_path: Path to wav.scp file

    Returns:
        dict: Mapping from audioid to audio file path
    """
    audio_path_mapping = {}
    with open(wav_scp_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                audioid = parts[0]
                audio_path = ' '.join(parts[1:])  # Handle paths with spaces
                audio_path_mapping[audioid] = audio_path
    return audio_path_mapping

# def create_audio_path_mapping_from_dir(wav_dir):
#     """
#     Create audio path mapping from wav directory.

#     Args:
#         wav_dir: Directory containing WAV files

#     Returns:
#         dict: Mapping from audioid to audio file path
#     """
#     audio_path_mapping = {}
#     if not os.path.exists(wav_dir):
#         raise FileNotFoundError(f"WAV directory not found: {wav_dir}")

#     for filename in os.listdir(wav_dir):
#         if filename.endswith('.wav'):
#             audioid = os.path.splitext(filename)[0]
#             audio_path = os.path.join(wav_dir, filename)
#             audio_path_mapping[audioid] = audio_path

#     return audio_path_mapping

def create_audio_path_mapping_from_dir(wav_dir):
    """
    Create audio path mapping from wav directory, including subdirectories.

    Args:
        wav_dir: Directory containing WAV files (searches recursively)

    Returns:
        dict: Mapping from audioid to audio file path
    """
    audio_path_mapping = {}

    if not os.path.exists(wav_dir):
        raise FileNotFoundError(f"WAV directory not found: {wav_dir}")

    # Walk through all subdirectories
    for root, dirs, files in os.walk(wav_dir):
        for filename in files:
            if filename.endswith('.wav'):
                audioid = os.path.splitext(filename)[0]
                audio_path = os.path.join(root, filename)

                # Handle duplicate audioids by keeping the first occurrence
                # or use a different strategy based on your needs
                if audioid not in audio_path_mapping:
                    audio_path_mapping[audioid] = audio_path
                else:
                    print(f"Warning: Duplicate audioid '{audioid}' found. Keeping: {audio_path_mapping[audioid]}")

    return audio_path_mapping

def extract_spectrogram_batch(audio_paths, begin_secs, end_secs, batch_size=32,
                               sample_rate=48000, cache_size=100, device=None):
    """
    Extract spectrogram features from audio segments in batches with audio caching.

    Args:
        audio_paths: List of paths to audio files
        begin_secs: List of start times in seconds
        end_secs: List of end times in seconds
        batch_size: Number of samples to process at once
        sample_rate: Target sample rate
        cache_size: Maximum number of audio files to keep in memory
        device: Device to use for computation

    Returns:
        tuple: (features, valid_indices)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Using device: {device}")

    # Initialize LRU cache
    audio_cache = OrderedDict()

    features = []
    valid_indices = []

    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i:i + batch_size]
        batch_begins = begin_secs[i:i + batch_size]
        batch_ends = end_secs[i:i + batch_size]

        batch_signals = []
        batch_valid_indices = []

        expected_length = None
        for local_idx, (path, begin, end) in enumerate(zip(batch_paths, batch_begins, batch_ends)):
            global_idx = i + local_idx

            try:
                # Check cache first
                if path in audio_cache:
                    data = audio_cache[path]
                    audio_cache.move_to_end(path)
                else:
                    data, rate = torchaudio.load(path)
                    if rate != sample_rate:
                        resampler = torchaudio.transforms.Resample(rate, sample_rate).to(device)
                        data = resampler(data)
                    audio_cache[path] = data

                    if len(audio_cache) > cache_size:
                        audio_cache.popitem(last=False)

                # Extract segment
                start_index = round(sample_rate * begin)
                end_index = round(sample_rate * end)

                if start_index >= data.shape[1] or start_index >= end_index:
                    logger.warning(f"Invalid segment for {path}: start={start_index}, end={end_index}")
                    continue

                if end_index > data.shape[1]:
                    end_index = data.shape[1]

                signal_piece = data[:, start_index:end_index]

                if signal_piece.shape[1] == 0:
                    logger.warning(f"Empty signal for {path}")
                    continue

                # Verify length consistency
                if expected_length is None:
                    expected_length = signal_piece.shape[1]
                elif signal_piece.shape[1] != expected_length:
                    logger.warning(f"Length mismatch for {path}: expected {expected_length}, got {signal_piece.shape[1]}")
                    continue

                batch_signals.append(signal_piece)
                batch_valid_indices.append(global_idx)

            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                continue

        if len(batch_signals) == 0:
            continue

        try:
            # Stack and compute spectrograms
            batch_tensor = torch.stack(batch_signals).to(device)

            nfft = 512
            window_size = batch_ends[0] - batch_begins[0]
            hop_length = int((window_size * sample_rate - nfft) // (256 - 1))

            spectrogram_transform = torchaudio.transforms.Spectrogram(
                n_fft=nfft,
                hop_length=hop_length,
                power=2,
                center=False
            ).to(device)

            specs = spectrogram_transform(batch_tensor)
            specs_db = torchaudio.transforms.AmplitudeToDB().to(device)(specs)

            batch_features = specs_db.detach().cpu().numpy()
            features.extend([feat[0].astype(np.float16) for feat in batch_features])
            valid_indices.extend(batch_valid_indices)

        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            continue

    logger.info(f"Extracted {len(features)} spectrograms")
    return features, valid_indices

def load_mae_model(model_path, device):
    """
    Load pretrained MAE model for feature extraction.

    Args:
        model_path: Path to pretrained model checkpoint
        device: Device to load model on

    Returns:
        model: Loaded MAE model in eval mode
    """
    logger.info(f"Loading MAE model from {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu')

    # Create model
    config = ViTMAEConfig.from_pretrained('facebook/vit-mae-base')
    model = ViTMAEModel.from_pretrained('facebook/vit-mae-base', config=config)

    # Extract encoder weights from checkpoint
    if 'model' in checkpoint:
        encoder_state_dict = {}
        for key, value in checkpoint['model'].items():
            if key.startswith('vit.'):
                encoder_state_dict[key[4:]] = value
        model.load_state_dict(encoder_state_dict, strict=False)
        logger.info("Loaded encoder weights from checkpoint")
    else:
        model.load_state_dict(checkpoint)
        logger.info("Loaded weights directly from checkpoint")

    model.to(device)
    model.eval()

    return model

def preprocess_spectrogram(spec):
    """
    Preprocess spectrogram for MAE model input.

    Args:
        spec: Spectrogram array of shape (257, 256)

    Returns:
        tensor: Preprocessed tensor of shape (3, 224, 224)
    """
    # Normalize to [0, 1]
    spec = spec.astype(np.float32)
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-8)

    # Convert to tensor and add channel dimension
    spec_tensor = torch.from_numpy(spec).unsqueeze(0)

    # Convert to 3 channels
    spec_tensor = spec_tensor.repeat(3, 1, 1)

    # Resize to 224x224
    resize = transforms.Resize((224, 224))
    spec_tensor = resize(spec_tensor)

    # Apply ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    spec_tensor = normalize(spec_tensor)

    return spec_tensor

def extract_mae_features(spectrograms, mae_model, batch_size=64, device=None):
    """
    Extract MAE features from spectrograms.

    Args:
        spectrograms: List of spectrogram arrays
        mae_model: Pretrained MAE model
        batch_size: Batch size for processing
        device: Device to use

    Returns:
        np.ndarray: MAE features array
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mae_features = []

    for i in tqdm(range(0, len(spectrograms), batch_size), desc="Extracting MAE features"):
        batch_specs = spectrograms[i:i + batch_size]

        # Preprocess batch
        batch_tensors = [preprocess_spectrogram(spec) for spec in batch_specs]
        batch_tensor = torch.stack(batch_tensors).to(device)

        # Extract features
        with torch.no_grad():
            outputs = mae_model(batch_tensor, output_hidden_states=True)
            cls_tokens = outputs.last_hidden_state[:, 0, :]  # Get [CLS] token
            mae_features.append(cls_tokens.cpu().numpy())

    mae_features = np.concatenate(mae_features, axis=0)
    logger.info(f"Extracted MAE features shape: {mae_features.shape}")

    return mae_features

def process_csv(csv_path, wav_scp_path, wav_dir, output_dir, mae_model_path,
                batch_size=32, gpu='auto'):
    """
    Process CSV file to extract both spectrogram and MAE features.

    Args:
        csv_path: Path to input CSV file
        wav_scp_path: Path to wav.scp file (optional if wav_dir is provided)
        wav_dir: Directory containing WAV files (optional if wav_scp_path is provided)
        output_dir: Output directory
        mae_model_path: Path to pretrained MAE model
        batch_size: Batch size for processing
        gpu: GPU selection
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = set_device(gpu)
    logger.info(f"Using device: {device}")

    # Load CSV
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {csv_path}")

    # Load audio path mapping
    if wav_scp_path:
        audio_path_mapping = load_wav_scp(wav_scp_path)
        logger.info(f"Loaded {len(audio_path_mapping)} audio paths from wav.scp")
    elif wav_dir:
        audio_path_mapping = create_audio_path_mapping_from_dir(wav_dir)
        logger.info(f"Found {len(audio_path_mapping)} audio files in {wav_dir}")
    else:
        raise ValueError("Either --wav_scp or --wav_dir must be provided")

    # Prepare data
    audio_paths = []
    begin_secs = []
    end_secs = []
    valid_indices = []

    for idx, row in df.iterrows():
        audioid = row['audioid']
        if audioid not in audio_path_mapping:
            logger.warning(f"Audio file not found for {audioid}")
            continue

        audio_paths.append(audio_path_mapping[audioid])
        begin_secs.append(round(row['ext_cut_begin_sec'], 6))
        end_secs.append(round(row['ext_cut_end_sec'], 6))
        valid_indices.append(idx)

    logger.info(f"Prepared {len(audio_paths)} samples for processing")

    # Extract spectrograms
    logger.info("=" * 60)
    logger.info("EXTRACTING SPECTROGRAMS")
    logger.info("=" * 60)

    spectrograms, spec_valid_indices = extract_spectrogram_batch(
        audio_paths,
        begin_secs,
        end_secs,
        batch_size=batch_size,
        device=device
    )

    # Filter valid samples
    valid_specs = []
    final_indices = []
    for spec, idx in zip(spectrograms, spec_valid_indices):
        if spec.shape == (257, 256):
            valid_specs.append(spec)
            final_indices.append(valid_indices[idx])
        else:
            logger.warning(f"Invalid spectrogram shape: {spec.shape}")

    logger.info(f"Valid spectrograms: {len(valid_specs)}")

    # Save spectrograms
    spec_array = np.array(valid_specs, dtype=np.float16)
    spec_output_path = os.path.join(output_dir, 'spec_raw.npy')
    np.save(spec_output_path, spec_array)
    logger.info(f"Saved spectrograms to {spec_output_path}, shape: {spec_array.shape}")

    # Extract MAE features
    logger.info("=" * 60)
    logger.info("EXTRACTING MAE FEATURES")
    logger.info("=" * 60)

    mae_model = load_mae_model(mae_model_path, device)
    mae_features = extract_mae_features(valid_specs, mae_model, batch_size=64, device=device)

    # Save MAE features
    mae_output_path = os.path.join(output_dir, 'mae_raw.npy')
    np.save(mae_output_path, mae_features.astype(np.float16))
    logger.info(f"Saved MAE features to {mae_output_path}, shape: {mae_features.shape}")

    # Save info CSV
    info_df = df.iloc[final_indices].reset_index(drop=True)
    info_output_path = os.path.join(output_dir, 'info.csv')
    info_df.to_csv(info_output_path, index=False)
    logger.info(f"Saved info CSV to {info_output_path}, {len(info_df)} rows")

    # Print summary
    logger.info("=" * 60)
    logger.info("PROCESSING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total input samples: {len(df)}")
    logger.info(f"Successfully processed: {len(info_df)}")
    logger.info(f"Spectrogram shape: {spec_array.shape}")
    logger.info(f"MAE feature shape: {mae_features.shape}")
    logger.info(f"Output directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Extract MAE features from marmoset dataset CSV')

    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to input CSV file (e.g., data/exp1/exp1.csv)')
    parser.add_argument('--wav_scp', type=str, default=None,
                        help='Path to wav.scp file: each line audioid path_to_audio. (e.g., data/local/Experiment_1/wav.scp)')
    parser.add_argument('--wav_dir', type=str, default=None,
                        help='Directory containing WAV files: $wav_dir/${audioid}.wav. (alternative to wav_scp)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory (e.g., exp/sandbox/exp1)')
    parser.add_argument('--mae_model', type=str,
                        default='conf/model/mae_pretrained_model_epoch_400_base_48days.pt',
                        help='Path to pretrained MAE model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for spectrogram extraction')
    parser.add_argument('--gpu', type=str, default='auto',
                        help="GPU selection: number or 'auto'")
    parser.add_argument('--seed', type=int, default=2020,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Validate arguments
    if not args.wav_scp and not args.wav_dir:
        parser.error("Either --wav_scp or --wav_dir must be provided")

    logger.info("=" * 60)
    logger.info("MAE FEATURE EXTRACTION")
    logger.info("=" * 60)
    logger.info(f"CSV path: {args.csv_path}")
    if args.wav_scp:
        logger.info(f"WAV scp: {args.wav_scp}")
    if args.wav_dir:
        logger.info(f"WAV dir: {args.wav_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"MAE model: {args.mae_model}")

    process_csv(
        args.csv_path,
        args.wav_scp,
        args.wav_dir,
        args.output_dir,
        args.mae_model,
        batch_size=args.batch_size,
        gpu=args.gpu
    )

    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()

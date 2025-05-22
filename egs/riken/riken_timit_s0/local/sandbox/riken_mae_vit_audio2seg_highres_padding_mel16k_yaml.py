# Implemented by bin-wu at 12:25 on 26 April 2024
# Updated to use Vision Transformer (ViT) with MAE architecture at 14:30 on 13 May 2025
# Adapted for 16kHz mel spectrograms at 16:45 on 15 May 2025
# Added batch processing for feature extraction at 15:00 on 20 May 2025
# Updated to handle data division YAML and multiple files at 16:00 on 20 May 2025

import os
import sys

import math
import random
import argparse

import GPUtil
from omegaconf import OmegaConf
import codecs

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader

# Use ViTMAE ViT encoder
from transformers import ViTMAEForPreTraining
from torchvision import transforms

import logging
# Configure the logging system
logging.basicConfig(
    level=logging.INFO,
    format="[ %(asctime)s | %(filename)s | %(levelname)s ] %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S"
)

# Create a logger object
logger = logging.getLogger(__name__)

def set_seed(seed):
    """
    Set the seed for reproducibility across NumPy and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def create_spec_data_batch(wav_file, segment_duration=2.5, sample_rate=16000, nfft=1024, win_size=0.5, keep_last_incomplete_segment=False, use_padding=False, feat_batch_size=32):
    """
    Generates mel spectrogram data from an input WAV file with optional padding, using batch processing.

    Args:
        wav_file (str): Path to the input WAV file.
        segment_duration (float): Duration of each audio segment in seconds. Default is 2.5 seconds.
        sample_rate (int): Sampling rate to which the audio should be resampled. Default is 16,000 Hz.
        nfft (int): Number of data points used in each block for the FFT. Default is 1024 for better low-frequency resolution.
        win_size(float): The window size in seconds. Default 0.5.
        keep_last_incomplete_segment (bool): keep the last incomplete segment or not. Default False.
        use_padding (bool): Whether to add padding to the audio (default: False).
        feat_batch_size (int): Number of signal pieces to process in a batch during feature extraction. Default is 32.

    Returns:
        np.ndarray: Mel spectrogram data as a NumPy array in 16-bit float format
        float: Original audio duration in seconds
        bool: Whether padding was used
    """
    logger.info(f"Processing wav audio file: {wav_file}...")
    waveform, rate = torchaudio.load(wav_file)
    original_duration = waveform.shape[1] / rate
    waveform = waveform.to(device)

    if use_padding:
        # Calculate padding size (half of segment_duration)
        pad_size = int(rate * segment_duration / 2)
        # Add padding to both ends
        waveform = F.pad(waveform, (pad_size, pad_size), mode='reflect')
        logger.info(f"Added padding of {pad_size} samples ({pad_size/rate:.3f}s) to each side of the audio")

    num_segments = int(len(waveform[0]) / (rate * segment_duration))

    # Initialize the mel spectrogram transform
    hop_length = int((win_size*sample_rate - nfft) // (256 - 1)) # Calculate hop_length using consistent approach
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=nfft,
        hop_length=hop_length,
        n_mels=257,
        power=2,
        center=False
    ).to(device)
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(device)

    # Prepare resampler if needed
    if rate != sample_rate:
        resampler = torchaudio.transforms.Resample(rate, sample_rate).to(device)

    spec_data = []
    num_segs = num_segments if keep_last_incomplete_segment else num_segments - 1

    for batch_start in range(0, num_segs, feat_batch_size):
        batch_end = min(batch_start + feat_batch_size, num_segs)
        batch_signals = []

        for i in range(batch_start, batch_end):
            # Extract the audio segment
            start = int(rate * segment_duration * i)
            end = int(rate * segment_duration * (i + 1))
            signal_piece = waveform[:, start:end]

            # Resample the audio segment if necessary
            if rate != sample_rate:
                signal_piece = resampler(signal_piece)

            # Skip if the signal piece is shorter than expected
            if signal_piece.shape[1] < int(sample_rate * segment_duration):
                logger.info(f"Skipping segment {i}: insufficient length (expected {int(sample_rate * segment_duration)}, got {signal_piece.shape[1]})")
                continue

            batch_signals.append(signal_piece)

        if not batch_signals:
            logger.info(f"No valid segments in batch {batch_start} to {batch_end}. Skipping.")
            continue

        # Stack the batch of signals
        batch_tensor = torch.cat(batch_signals, dim=0)

        # Compute the mel spectrogram of the audio segments in batch
        mel_spec = mel_spectrogram_transform(batch_tensor)
        mel_spec_db = amplitude_to_db(mel_spec)

        # Process each mel spectrogram in the batch
        for spectrum in mel_spec_db:
            spectrum = spectrum.detach().cpu().numpy()
            # Crop to ensure expected time frames
            if spectrum.shape[1] > 1299:
                spectrum = spectrum[:, :1299]

            if spectrum.shape != (257, 1299):  # (frequency, time)
                logger.info(f"Invalid array shape: {spectrum.shape}")
                continue
            spec_data.append(spectrum.astype(np.float16))

        # Print progress information
        logger.info(f"Processed {batch_end}/{num_segments} segments")

    # Return the mel spectrogram data as a numpy array
    return np.array(spec_data, dtype=np.float16), original_duration, use_padding

class MAEViTEncoder(nn.Module):
    """
    Vision Transformer (ViT) encoder using MAE's architecture without masking mechanism for single-channel mel spectrogram input.

    This model:
    - Uses the encoder part of MAE architecture without any masking
    - Handles single-channel to 3-channel conversion for mel spectrograms
    - Supports both training from scratch and loading pretrained weights
    - Provides flexible pooling options and latent representation extraction
    - Uses ImageNet normalization for pretrained model compatibility

    Args:
        image_size (int): Input image size (assumes square images). Default: 224 (MAE default)
        patch_size (int): Size of patches. Default: 16 (MAE default)
        num_classes (int): Number of output classes
        dim (int): Hidden dimension size. Default: 768 (MAE default)
        depth (int): Number of transformer layers. Default: 12 (MAE default)
        heads (int): Number of attention heads. Default: 12 (MAE default)
        mlp_dim (int): Dimension of MLP layer. Default: 3072 (MAE default)
        pool (str): Pooling type ('cls' or 'mean'). Default: 'cls'
        channels (int): Number of input channels. Default: 1 (for spectrograms)
        dim_head (int): Dimension of each attention head. Default: 64
        dropout (float): Dropout rate. Default: 0.0
        emb_dropout (float): Embedding dropout rate. Default: 0.0
        return_attention (bool): Whether to return attention weights. Default: False
        return_logits (bool): Whether to return logits instead of probabilities. Default: False
        return_latent (bool): Whether to return latent representations. Default: False
        pretrained (bool): Whether to load pretrained MAE weights. Default: False

    Input shape:
        - Single channel: (batch_size, 1, height, width) or (batch_size, height, width)
        - Will be converted to: (batch_size, 3, 224, 224)

    Output shape:
        Based on configuration:
        - If return_latent: (batch_size, dim)
        - If return_logits: (batch_size, num_classes) before softmax
        - Otherwise: (batch_size, num_classes) after softmax
    """
    def __init__(self, *, image_size=224, patch_size=16, num_classes, dim=768,
                 depth=12, heads=12, mlp_dim=3072, pool='cls', channels=1,
                 dim_head=64, dropout=0., emb_dropout=0., return_attention=False,
                 return_logits=False, return_latent=False, mae_pretrained='base',
                 mae_train_layers=-1):
        super().__init__()

        # Initialize ViT model
        if mae_pretrained:
            if mae_pretrained == 'base':
                # Get the ViT encoder from pretrained MAE base model
                mae = ViTMAEForPreTraining.from_pretrained(
                    'facebook/vit-mae-base',
                    mask_ratio=0.0  # Disable masking
                )
                self.vit = mae.vit
            else:
                # Load from checkpoint saved by riken_mae_pretrained.py
                checkpoint = torch.load(mae_pretrained, map_location='cpu')
                # Create a new MAE model first
                mae = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base', mask_ratio=0.0)
                # Load the saved state dict
                mae.load_state_dict(checkpoint['model'])
                self.vit = mae.vit

            # Freeze layers based on mae_train_layers parameter
            self.freeze_base_layers(mae_train_layers)
        else:
            # Initialize fresh MAE with masking disabled
            from transformers import ViTMAEConfig, ViTMAEModel

            config = ViTMAEConfig(
                image_size=image_size,
                patch_size=patch_size,
                hidden_size=dim,
                num_hidden_layers=depth,
                num_attention_heads=heads,
                intermediate_size=mlp_dim,
                hidden_dropout_prob=dropout,
                attention_probs_dropout_prob=dropout,
                num_channels=3,  # Always use 3 channels after input processing
                mask_ratio=0.0  # Disable masking
            )
            self.vit = ViTMAEModel(config)

        # Classification head for supervised learning
        self.mlp_head = nn.Linear(dim, num_classes)

        # Model configuration
        self.pool = pool
        self.return_attention = return_attention
        self.return_logits = return_logits
        self.return_latent = return_latent

        # Input processing configuration
        self.patch_size = patch_size
        self.image_size = image_size

        # Input transformation pipeline
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )

    def freeze_base_layers(self, mae_train_layers):
        """
        Freeze specific layers based on mae_train_layers parameter when using pretrained model.
        All parameters are trainable by default, only freezing layers if specified.

        Args:
            mae_train_layers (int): Number of last layers to keep trainable.
                                    If -1, all layers remain trainable.
        """
        if mae_train_layers > 0:
            # Calculate which layers to freeze
            num_layers = len(self.vit.encoder.layer)
            layers_to_freeze = num_layers - mae_train_layers

            # Freeze only the specified number of early layers
            for i in range(layers_to_freeze):
                for param in self.vit.encoder.layer[i].parameters():
                    param.requires_grad = False

    def process_input(self, x):
        """
        Process single-channel mel spectrogram input to match MAE's expected input format.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, height, width)
                            or (batch_size, height, width)

        Returns:
            torch.Tensor: Processed tensor of shape (batch_size, 3, 224, 224)

        Steps:
            1. Add channel dimension if needed
            2. Resize to required dimensions
            3. Normalize to [0, 1]
            4. Convert to 3 channels
            5. Apply ImageNet normalization
        """
        # Add channel dimension if needed
        if len(x.shape) == 3:
            x = x.unsqueeze(1)

        # Resize if necessary
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = transforms.Resize((self.image_size, self.image_size))(x)

        # Normalize to [0, 1]
        x = (x - x.min()) / (x.max() - x.min())

        # Convert to 3 channels
        x = x.repeat(1, 3, 1, 1)

        # Apply ImageNet normalization
        x = self.normalize(x)

        return x

    def forward(self, img, y=None, mode='train'):
        """
        Forward pass of the model.

        Args:
            img (torch.Tensor): Input spectrogram
            y (torch.Tensor, optional): Ground truth labels for supervised training
            mode (str, optional): The mode of operation ('train' or 'eval'). Default: 'train'

        Returns:
            Different return formats based on configuration:
            - If return_latent:
                tuple: (latent_features, attention_weights)
            - If y is provided:
                tuple: (output, loss, accuracy, attention_weights)
            - Otherwise:
                tuple: (output, attention_weights)

            Where:
            - latent_features: tensor of shape (batch_size, dim)
            - output: tensor of shape (batch_size, num_classes)
            - attention_weights: list of attention matrices if return_attention=True
            - loss: scalar tensor if y is provided
            - accuracy: list of per-sample accuracy if y is provided
        """
        # Process input to match expected format
        img = self.process_input(img)

        # Forward pass through ViT encoder (no masking)
        outputs = self.vit(
            img,
            output_attentions=self.return_attention,
            return_dict=True
        )

        # Get latent features based on pooling strategy
        if self.pool == 'cls':
            latent = outputs.last_hidden_state[:, 0]  # Use CLS token
        else:
            latent = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

        # Get attention weights if requested
        attention_weights = outputs.attentions if self.return_attention else None

        # Return latent features if requested
        if self.return_latent:
            return latent, attention_weights

        # Classification head
        logits = self.mlp_head(latent)
        output = logits if self.return_logits else F.softmax(logits, dim=1)

        # Handle supervised training case
        if y is not None:
            y = y.type(torch.LongTensor).to(y.device)
            loss = F.cross_entropy(logits, y)
            predicted = logits.argmax(dim=1)
            accuracy = (predicted == y).float().tolist()
            return output, loss, accuracy, attention_weights

        return output, attention_weights

def pred_input_function(xs, i, window_size=256, step_size=26):
    """
    Creates a minibatch of 50 500ms-spectrograms from a 2500ms-spectrogram using a sliding window
    with the window size of 500ms and the window shift of 50ms.

    This function takes a sequence of spectral segments (xs) and an index (i) representing
    the current position in the sequence. It extracts 50 elements, each of 500ms spectrogram with
    a shape of (257, 256), for the current position of a 2500ms-spectrogram with a shape of (257, 1299).

    The extraction process is done using a sliding window approach with a window size of 256 and a
    step size of 26 (the step size or window shift of 26 pixels is from floor(1299/50)).
    The first num_complete_elements can be extracted entirely from the current 2500ms-segment, while
    the remaining elements require concatenation with the next segment.

    Args:
        xs (array-like): Input spectral segments, of shape (*, 257, 1299),
                         where * is number of batches (number of 2500ms-segments)
        i (int): The index of the current spectral segment.
        window_size (int): The size of each element (default: 256).
        step_size (int): The step size for sliding the window (default: 26).

    Returns:
        numpy.ndarray: A minibatch of input features, of shape (50, 257, 256).

    Note: The window shift of 50ms is used for generating the segment files with a resolution of 50ms
    for the test set.
    """
    num_elements = 50
    num_complete_elements = math.floor((xs.shape[2] - window_size) / step_size) + 1

    elements = []

    # Extract complete elements from the current spectral segment
    for j in range(num_complete_elements):
        start = j * step_size
        element = xs[i, :, start:start+window_size] # Take i-th batch with the whole frequency bins and a window time.
        elements.append(element)

    # Extract remaining elements that cross the segment boundary
    for k in range(num_elements - num_complete_elements):
        start = (num_complete_elements + k) * step_size
        init_element = xs[i, :, start:]
        remaining = window_size - init_element.shape[1]
        element = np.concatenate([init_element, xs[i+1, :, :remaining]], axis=1)
        elements.append(element)

    return np.array(elements, dtype=np.float32)

def predict(model, pred_data, model_path, avg_pred_win, batch_size=None):
    """
    Function for predicting with the trained network for the testing set.

    Args:
        model (MAEViTEncoder): The trained model used for prediction.
        pred_data (str or numpy.array): Path to the file containing spectrogram arrays for prediction or specgram array itself.
        model_path (str): Path to the saved model to be used for prediction.
        avg_pred_win (int): Size of the averaging window for predictions.
        batch_size (int): Minibatch size for prediction.

    Returns:
        numpy.ndarray: The predicted probabilities.
    """
    logger.info(f"Loading model from {model_path}")

    # Load the input data from path or directly use it
    if isinstance(pred_data, str):
        with open(pred_data, 'rb') as f:
            predict_x = np.load(f)
    elif isinstance(pred_data, np.ndarray):
        predict_x = pred_data
    else:
        raise ValueError("Invalid input type for pred_data. Expected string or numpy array.")

    # Restore the saved model for prediction
    model.load_state_dict(torch.load(model_path, map_location=device)["model"])
    logger.info(f'Model restored from {model_path}')

    # Perform predictions on the input data
    preds_list = []
    predictions = []
    logger.info('Predicting')
    model.eval()

    length = predict_x.shape[0]
    num_batches = length
    for i in range(num_batches-1):
        # Get the input batch using the pred_input_function
        inputs = pred_input_function(predict_x, i)
        # Run the model to get the predictions for the current batch
        preds, _ = model(torch.from_numpy(inputs).to(device))
        # Append the predictions to the preds_list
        preds_list.extend(preds.detach().cpu().numpy())
        if (i + 1) % 25 == 0:
            logger.info(f"Processed {i + 1}/{num_batches} batches")

    # Average predictions across consecutive windows
    for i in range(len(preds_list) - (avg_pred_win - 1)):
        # Calculate the mean predictions for the current window
        mean_preds = np.mean(preds_list[i:i + avg_pred_win], axis=0)
        # Append the mean predictions to the final predictions list
        predictions.append(mean_preds)

    logger.info(f"Predictions shape: {np.shape(predictions)}")
    logger.info(f"Used the model: {model_path}")

    return np.array(predictions)

def merge_segments(input_segments):
    """
    Merges adjoining segments with the same labels from a given segment file and outputs the results.

    Args:
        input_segments: A list of unmerged segments, where each segment is a tuple (start_time, end_time, label)

    Returns:
        list: A list of merged segments, where each segment is a tuple (start_time, end_time, label).
    """
    segments = input_segments
    segments.sort(key=lambda x: x[0])

    # Iterate over the segments and merge adjoining segments with the same label
    merged_segments = []
    current_start, current_end, current_label = segments[0]

    for start_time, end_time, label in segments[1:]:
        # Check if the current segment can be merged with the previous segment
        if start_time == current_end and label == current_label:
            # Update the end time of the merged segment
            current_end = end_time
        else:
            # Add the previous merged segment to the list of merged segments
            merged_segments.append((round(current_start, 6), round(current_end, 6), current_label))
            # Update the current segment variables
            current_start, current_end, current_label = start_time, end_time, label

    # Add the last merged segment to the list of merged segments
    merged_segments.append((round(current_start, 6), round(current_end, 6), current_label))

    return merged_segments

def predict_segments(pred_prob, label2id):
    """
    Predict labels from the given prediction probabilities and return the segments

    Args:
        pred_prob (np.array): Numpy array that contain prediction probabilities.
        label2id (dict): dict from label to id

    Returns:
        raw_segments, merged_segments
        list: A list of raw segments, where each segment is a tuple (start_time, end_time, label).
        list: A list of merged segments, where each segment is a tuple (start_time, end_time, label).
    """
    id2label = {id:label for label, id in label2id.items()}
    assert len(label2id) == len(id2label), f"label and id in label2id file should be one-to-one correspondence"

    raw_segments = []
    for i, pred in enumerate(pred_prob):
        # Get the index of the maximum prediction
        pred_label_idx = np.argmax(pred)

        current = i
        window_size = 0.5
        window_shift = 0.05 # shift of batch elements of 2500ms long segment, not exactly 0.05
        middle_part = 0.05
        start_window = current * window_shift
        end_window = start_window + window_size
        middle_start = start_window + (window_size - middle_part) / 2
        middle_end = start_window + (window_size + middle_part) / 2

        if pred_label_idx != label2id['noise']:  # Check if the prediction is not 'noise'
            start_time = middle_start + 0.1 # fixed factor of 0.1 might considering 1) shift of batch elements of 2500ms long segment that is not exactly 50ms; 2) prob avg window effects
            end_time = middle_end + 0.1
            label = id2label[pred_label_idx]
            raw_segments.append((round(start_time, 6), round(end_time, 6), label))

    if not raw_segments:
        print("\nWarning: all predictions are noises\n")
        return [], []

    merged_segments = merge_segments(raw_segments)
    return raw_segments, merged_segments

def extract_spectrogram_segments_with_padding_batch(audio_file, window_size=0.5, window_shift=0.15, nfft=1024, sample_rate=16000, feat_batch_size=32, use_padding=False):
    """
    Extract mel spectrogram segments from an audio file using a sliding window with optional padding, processing in batches.

    Args:
        audio_file (str): Path to the audio file (.wav).
        window_size (float): The size of the sliding window in seconds (default: 0.5).
        window_shift (float): The shift of the sliding window in seconds (default: 0.15).
        nfft (int): The number of FFT points (default: 1024 for better low-frequency resolution).
        sample_rate (int): The desired sample rate of the audio (default: 16000).
        feat_batch_size (int): Number of signal pieces to process in a batch during feature extraction (default: 32).
        use_padding (bool): Whether to add padding to the audio (default: False).

    Returns:
        np.ndarray: Mel spectrogram segments of shape (num_segments, 257, 256).
        float: The original audio duration in seconds.
        bool: Whether padding was used.
    """
    data, rate = torchaudio.load(audio_file)
    data = data.to(device)

    # Get original audio duration for later time adjustments
    original_duration = data.shape[1] / rate

    # Resample the audio to 16kHz if necessary
    if rate != sample_rate:
        resampler = torchaudio.transforms.Resample(rate, sample_rate).to(device)
        data = resampler(data)

    if use_padding:
        # Calculate the padding needed in samples
        pad_samples = int(window_size * sample_rate / 2)
        # Add padding to both sides of the audio using reflection padding
        padded_data = F.pad(data, (pad_samples, pad_samples), mode='reflect')
        logger.info(f"Added padding of {pad_samples} samples ({pad_samples/sample_rate:.3f}s) to each side of the audio")
    else:
        padded_data = data

    # Calculate the number of segments for the audio
    padded_duration = padded_data.shape[1] / sample_rate
    num_segments = int((padded_duration - window_size) / window_shift) + 1

    # Initialize the mel spectrogram transform
    hop_length = int((window_size * sample_rate - nfft) // (256 - 1))  # Calculate hop_length for 256 frames
    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=nfft,
        hop_length=hop_length,
        n_mels=257,
        power=2,
        center=False
    ).to(device)
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(device)

    input_spec = []
    window_samples = int(window_size * sample_rate)

    for batch_start in range(0, num_segments, feat_batch_size):
        batch_end = min(batch_start + feat_batch_size, num_segments)
        batch_signals = []

        for i in range(batch_start, batch_end):
            # Take a window_size piece out of the padded data with a step size of window_shift
            start_index = int(sample_rate * i * window_shift)
            end_index = start_index + window_samples
            signal_piece = padded_data[:, start_index:end_index]

            # Skip if the signal piece is shorter than expected
            if signal_piece.shape[1] < window_samples:
                logger.info(f"Skipping segment {i}: insufficient length (expected {window_samples}, got {signal_piece.shape[1]})")
                continue

            batch_signals.append(signal_piece)

        if not batch_signals:
            logger.info(f"No valid segments in batch {batch_start} to {batch_end}. Skipping.")
            continue

        # Stack the batch of signals
        batch_tensor = torch.cat(batch_signals, dim=0)

        # Create the mel spectrograms and scale them logarithmically
        mel_specs = mel_spectrogram_transform(batch_tensor)
        mel_specs_db = amplitude_to_db(mel_specs)

        # Process each mel spectrogram in the batch
        for mel_spec in mel_specs_db:
            result = mel_spec.detach().cpu().numpy()

            # Crop to ensure exactly 256 time frames
            if result.shape[1] > 256:
                result = result[:, :256]

            # Skip this piece if the shape is not as expected
            if np.shape(result) != (257, 256):  # (frequency, time)
                logger.info(f"Invalid shape: {np.shape(result)}")
                continue
            input_spec.append(np.array(result, dtype=np.float16))

        logger.info(f"Processed {batch_end}/{num_segments} segments")

    input_spec = np.array(input_spec, dtype=np.float16)
    logger.info(f'Number of spectrogram segments: {len(input_spec)}')
    return input_spec, original_duration, use_padding

class OneStreamDataset(Dataset):
    """
    A PyTorch Dataset class for the one-stream data.

    Args:
        xs (numpy.ndarray): Input data, of shape (data_length, 257, 256).
        train (bool): Whether the dataset is used for training or evaluation/development.
        apply_random_shift (bool): Whether to apply data argumentation of random shifts. Default is True.
                                    Each sample in the batch applies a new random shift within 5 pixels.
    """

    def __init__(self, xs, train=True, apply_random_shift=True):
        self.xs = xs
        self.train = train
        self.apply_random_shift = apply_random_shift

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        x = self.xs[idx]

        if self.train and self.apply_random_shift:
            ver_shift = random.randint(-5, 5)
            hor_shift = random.randint(-5, 5)
            x = np.roll(x, (ver_shift, hor_shift), axis=(0, 1))

        return x.astype(np.float32)

def create_dataloader(input, batch_size, train=True):
    """
    Create a data loader for the given input file.

    Args:
        input (str): Path to the input file.
        batch_size (int): Batch size for the data loader.
        train (bool): Whether to create a data loader for training or evaluation.

    Returns:
        torch.utils.data.DataLoader: The created data loader.
    """
    # Load data from the input file
    if isinstance(input, str):
        with open(input, 'rb') as f:
            xs = np.load(f)
    elif isinstance(input, np.ndarray):
        xs = input
    else:
        raise ValueError("Invalid input type. Expected string or numpy array.")

    # Create OneStreamDataset instance
    dataset = OneStreamDataset(xs, train=train, apply_random_shift=train)

    # Create DataLoader with the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=train)

    return data_loader

def predict_from_dataloader(model, dataloader, model_path, avg_pred_win, batch_size=None):
    """
    Function for predicting with the trained network using a dataloader.

    Args:
        model (MAEViTEncoder): The trained model used for prediction.
        dataloader (torch.utils.data.DataLoader): The dataloader created by create_dataloader.
        model_path (str): Path to the saved model to be used for prediction.
        avg_pred_win (int): Size of the averaging window for predictions.
        batch_size (int): Minibatch size for prediction.

    Returns:
        numpy.ndarray: The predicted probabilities.
    """
    logger.info(f"Loading model from {model_path}")

    # Restore the saved model for prediction
    model.load_state_dict(torch.load(model_path, map_location=device)["model"])
    logger.info(f'Model restored from {model_path}')

    # Perform predictions on the input data
    preds_list = []
    predictions = []
    logger.info('Predicting')
    model.eval()
    num_batches = len(dataloader)

    with torch.no_grad():
        for i, inputs in enumerate(dataloader):
            inputs = inputs.to(device)
            preds, _ = model(inputs)
            preds_list.extend(preds.detach().cpu().numpy())

            if (i + 1) % 25 == 0:
                logger.info(f"Processed {i + 1}/{num_batches} batches")

    # Average predictions across consecutive windows
    for i in range(len(preds_list) - (avg_pred_win - 1)):
        # Calculate the mean predictions for the current window
        mean_preds = np.mean(preds_list[i:i + avg_pred_win], axis=0)

        # Append the mean predictions to the final predictions list
        predictions.append(mean_preds)

    logger.info(f"Predictions shape: {np.shape(predictions)}")
    logger.info(f"Used the model: {model_path}")

    return np.array(predictions)

def predict_segments_v1(pred_prob, label2id, original_duration=None, window_size=0.5, window_shift=0.05, middle_part=0.05, fixed_factor=0.1, use_padding=False):
    """
    Predict labels from the given prediction probabilities and return the segments.

    Args:
        pred_prob (np.array): Numpy array that contains prediction probabilities.
        label2id (dict): dict from label to id
        original_duration (float): Original duration of the audio file in seconds (needed if use_padding=True).
        window_size (float): The size of the sliding window in seconds (default: 0.5).
        window_shift (float): The shift of the sliding window in seconds (default: 0.05).
        middle_part (float): The proportion of the middle part of the window in seconds for label assignment (default: 0.05).
        fixed_factor (float): A fixed factor added to the start and end times of segments (default: 0.1).
        use_padding (bool): Whether padding was used in preprocessing (default: False).

    Returns:
        raw_segments, merged_segments
        list: A list of raw segments, where each segment is a tuple (start_time, end_time, label).
        list: A list of merged segments, where each segment is a tuple (start_time, end_time, label).
    """
    id2label = {id: label for label, id in label2id.items()}
    assert len(label2id) == len(id2label), f"label and id in label2id file should be one-to-one correspondence"

    # Calculate the time offset due to padding (half of window size)
    time_offset = window_size / 2 if use_padding else 0

    raw_segments = []
    for i, pred in enumerate(pred_prob):
        # Get the index of the maximum prediction
        pred_label_idx = np.argmax(pred)

        current = i
        # Adjust the start time by subtracting the padding offset if padding was used
        start_window = (current * window_shift) - time_offset
        end_window = start_window + window_size
        middle_start = start_window + (window_size - middle_part) / 2
        middle_end = start_window + (window_size + middle_part) / 2

        if pred_label_idx != label2id['noise']:  # Check if the prediction is not 'noise'
            start_time = middle_start + fixed_factor
            end_time = middle_end + fixed_factor

            # If padding was used, ensure times are within the original audio duration
            if use_padding and original_duration is not None:
                start_time = max(0, start_time)
                end_time = min(original_duration, end_time)

                # Only add segments that are within the original audio duration
                if start_time >= original_duration or end_time <= 0 or start_time >= end_time:
                    continue

            label = id2label[pred_label_idx]
            raw_segments.append((round(start_time, 6), round(end_time, 6), label))

    if not raw_segments:
        print("\nWarning: all predictions are noises\n")
        return [], []

    merged_segments = merge_segments(raw_segments)
    return raw_segments, merged_segments

def process_files_from_yaml(info_json, eval_data_division_yaml, eval_dir, ref_key_in_info_json, model, eval_model, label2id, args):
    """
    Process all test files from data division YAML and generate predictions and references.

    Args:
        info_json (dict): Info JSON containing file paths
        eval_data_division_yaml (str): Path to data division YAML file
        eval_dir (str): Output evaluation directory
        ref_key_in_info_json (str): Key in info.json for reference files
        model: The model for prediction
        eval_model (str): Path to the saved model
        label2id (dict): Label to ID mapping
        args: Arguments containing model parameters
    """
    # Load data division YAML
    data_division = OmegaConf.load(eval_data_division_yaml)

    if 'test' not in data_division:
        logger.error(f"No 'test' section found in {eval_data_division_yaml}")
        return

    test_ids = data_division['test']
    logger.info(f"Found {len(test_ids)} test files: {test_ids}")

    # Create output directories
    hypo_dir = os.path.join(eval_dir, 'hyp')
    ref_dir = os.path.join(eval_dir, 'ref')
    os.makedirs(hypo_dir, exist_ok=True)
    os.makedirs(ref_dir, exist_ok=True)

    for test_id in test_ids:
        if test_id not in info_json:
            logger.warning(f"Test ID {test_id} not found in info.json, skipping...")
            continue

        wav_file = info_json[test_id]['wav']
        logger.info(f"Processing {test_id}: {wav_file}")

        # Generate hypothesis (prediction) file
        hypo_file = os.path.join(hypo_dir, f"{test_id}.txt")

        try:
            # Predict the segments using MAE ViT model
            if args.fast_pred:
                logger.info("Using the fast prediction with the 50ms prediction resolution by creating a dataloader from the 2500ms segments...")
                spec, original_duration, use_padding = create_spec_data_batch(
                    wav_file,
                    keep_last_incomplete_segment=args.complete,
                    use_padding=args.padding,
                    feat_batch_size=args.feat_batch_size
                )

                pred_prob = predict(model, spec, model_path=eval_model, avg_pred_win=args.avg_pred_win)

                _, merged_segments = predict_segments_v1(
                    pred_prob,
                    label2id,
                    original_duration=original_duration,
                    window_size=0.5,
                    window_shift=0.05,
                    middle_part=0.05,
                    fixed_factor=0.1,
                    use_padding=use_padding
                )
            else:
                spec, original_duration, use_padding = extract_spectrogram_segments_with_padding_batch(
                    wav_file,
                    window_size=0.5,
                    window_shift=args.pred_resolution,
                    use_padding=args.padding,
                    feat_batch_size=args.feat_batch_size
                )

                test_loader = create_dataloader(spec, args.batch_size, train=False)
                pred_prob = predict_from_dataloader(model, test_loader, model_path=eval_model, avg_pred_win=args.avg_pred_win)

                # Determine fixed_factor based on input argument
                if args.fixed_factor == -1:
                    # Auto-calculate
                    fixed_factor = 0.1 * (args.pred_resolution/0.05)
                else:
                    # Use the specified value
                    fixed_factor = args.fixed_factor

                _, merged_segments = predict_segments_v1(
                    pred_prob,
                    label2id,
                    original_duration=original_duration,
                    window_size=0.5,
                    window_shift=args.pred_resolution,
                    middle_part=args.pred_resolution,
                    fixed_factor=fixed_factor,
                    use_padding=use_padding
                )

            # Save hypothesis file
            with codecs.open(hypo_file, 'w', 'utf-8') as f:
                for start_time, end_time, label in merged_segments:
                    f.write(f"{start_time}\t{end_time}\t{label}\n")

            logger.info(f"Saved hypothesis file: {hypo_file}")

            # Copy reference file if ref_key is provided
            if ref_key_in_info_json and ref_key_in_info_json in info_json[test_id]:
                ref_source = info_json[test_id][ref_key_in_info_json]
                ref_file = os.path.join(ref_dir, f"{test_id}.txt")

                if os.path.exists(ref_source):
                    # Copy reference file
                    with open(ref_source, 'r') as src, codecs.open(ref_file, 'w', 'utf-8') as dst:
                        dst.write(src.read())
                    logger.info(f"Copied reference file: {ref_file}")
                else:
                    logger.warning(f"Reference file not found: {ref_source}")

        except Exception as e:
            logger.error(f"Error processing {test_id}: {str(e)}")
            continue

# Update default paths for MAE ViT model with mel 16k support
default_model = "exp/sys_mae_vit/timit/division_timit_winmid0.05size0.5shift0.05_noisekeep1_mel16k/mae_vit-run0/bs256baselr0.0001wd0.3warmup5total25avgpredwin5_dim768depth12heads12mlpdim3072patch16dropout0.1maetrain6maepretrainedbase/train/model.ckpt"
default_dict = "conf/dict/timit_label2id.yaml"
default_resolution=0.01
default_data_division="conf/data/division_timit.yaml"
default_info_json="data/timit/info.json"

parser = argparse.ArgumentParser(description="Convert from audio files to segment files using MAE ViT model with 16kHz mel spectrograms with data division YAML support.")

# Main arguments
parser.add_argument("--eval_model", type=str, default=default_model, help="Model path for prediction or evaluation")
parser.add_argument("--eval_dir", type=str, default="", help="Output evaluation directory")
parser.add_argument("--eval_data_division_yaml", type=str, default=default_data_division, help="YAML file containing data division with test IDs")
parser.add_argument("--info_json", type=str, default=default_info_json, help="JSON file mapping IDs to audio and label file paths")
parser.add_argument("--ref_key_in_info_json", type=str, default="seg", help="Key in info.json for reference files")

# Data arguments
parser.add_argument("--batch_size", type=int, default=25, help="Batch size for the dataloader")
parser.add_argument("--feat_batch_size", type=int, default=20480, help="Feat batch size for extracting features")

# Dictionary arguments
parser.add_argument("--label2id_yaml", type=str, default=default_dict, help="The YAML file that contains the label-to-labelID mapping.")

# Evaluation arguments
parser.add_argument("--avg_pred_win", type=int, default=5, help="Collect predicted probabilities by averaging across x consecutive predictions")

# Model arguments for MAE ViT
parser.add_argument("--image_size", type=int, default=224, help="Image size for the MAE model")
parser.add_argument("--patch_size", type=int, default=16, help="Patch size for the MAE model")
parser.add_argument("--dim", type=int, default=768, help="Embedding dimension")
parser.add_argument("--depth", type=int, default=12, help="Number of transformer layers")
parser.add_argument("--heads", type=int, default=12, help="Number of attention heads")
parser.add_argument("--mlp_dim", type=int, default=3072, help="Dimension of the MLP layer")
parser.add_argument("--pool", type=str, default='cls', choices=['cls', 'mean'], help="Pooling type")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
parser.add_argument("--return_attention", action='store_true', help="Return attention weights")
parser.add_argument("--return_logits", action='store_true', help="Return logits instead of probabilities")
parser.add_argument('--mae_pretrained', type=str, default='base', help='Path to pretrained MAE model, "base" for facebook/vit-mae-base, or "" for no pretraining')
parser.add_argument('--mae_train_layers', type=int, default=-1, help='Number of last layers to train in MAE when using pretrained model. -1 means train all layers')

# Other arguments
parser.add_argument('--gpu', type=str, default="auto", help="e.g., '--gpu 2' for using device of gpu 'cuda:2'; '--gpu auto' for gpu with the least gpu memory; '--gpu cpu' for cpu.")
parser.add_argument("--pred_resolution", type=float, default=default_resolution, help="The resolution of the prediction. Default 0.01.")
parser.add_argument("--complete", action="store_true", help="Keep the last potentially incomplete segment")
parser.add_argument("--fast_pred", action="store_true", help="Fast prediction using 50ms prediction resolution by creating a dataloader from 2500ms segments")
parser.add_argument("--fixed_factor", type=float, default=0, help="Fixed factor added to segment times. Default 0.")
parser.add_argument("--padding", action="store_true", help="Add padding to the audio to enable prediction at the start and end")

args = parser.parse_args()
args.padding = True  # always using padding

# Check required arguments
if not args.eval_dir:
    logger.error("--eval_dir is required")
    sys.exit(1)

# Load configuration files
label2id = OmegaConf.load(args.label2id_yaml)
info_json = OmegaConf.load(args.info_json)

# Set seed and device
set_seed(2020)
device = set_device(args.gpu)
logger.info(f"Device: {device}")

# Create the MAE ViT model
model = MAEViTEncoder(
    image_size=args.image_size,
    patch_size=args.patch_size,
    num_classes=len(label2id),
    dim=args.dim,
    depth=args.depth,
    heads=args.heads,
    mlp_dim=args.mlp_dim,
    pool=args.pool,
    dropout=args.dropout,
    return_attention=args.return_attention,
    return_logits=args.return_logits,
    mae_pretrained=args.mae_pretrained,
    mae_train_layers=args.mae_train_layers
).to(device)

# Process all files from YAML
process_files_from_yaml(
    info_json=info_json,
    eval_data_division_yaml=args.eval_data_division_yaml,
    eval_dir=args.eval_dir,
    ref_key_in_info_json=args.ref_key_in_info_json,
    model=model,
    eval_model=args.eval_model,
    label2id=label2id,
    args=args
)

logger.info(f"Processing complete. Results saved in {args.eval_dir}")

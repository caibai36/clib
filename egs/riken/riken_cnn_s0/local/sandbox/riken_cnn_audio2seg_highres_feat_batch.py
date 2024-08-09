# Implemented by bin-wu at 12:25 on 26 April 2024

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

def create_spec_data_batch(wav_file, segment_duration=2.5, sample_rate=48000, nfft=512, win_size=0.5, keep_last_incomplete_segment=False, feat_batch_size=32):
    """
    Generates spectrogram data from an input WAV file and saves it to a specified location.
    This function reads an audio file, segments it into chunks of specified duration,
    resamples if necessary, computes the spectrogram for each segment in batches, and then saves
    the collection of spectrograms to a file.
    The function returns the spectrogram data as a NumPy array in 16-bit float format

    Args:
    wav_file (str): Path to the input WAV file.
    segment_duration (float): Duration of each audio segment to be analyzed, in seconds. Default is 2.5 seconds.
    sample_rate (int): Sampling rate to which the audio should be resampled. Default is 48,000 Hz.
    nfft (int): Number of data points used in each block for the FFT. Default is 512.
    win_size(float): The window size of an element in a batch, where each long segment will form a batch in evaluation. Default 0.5.
    keep_last_incomplete_segment (bool): keep the last incomplete segment or not (experimental). Default False.
    feat_batch_size (int): Number of signal pieces to process in a batch during feature extraction. Default is 32.

    Note:
    The evaluation dataloader currently only supports that 1) each batch element with the window size of 500ms and 2) the segment_duration of 2500ms
    """
    logger.info(f"Processing wav audio file: {wav_file}...")
    waveform, rate = torchaudio.load(wav_file)
    waveform = waveform.to(device)
    num_segments = int(len(waveform[0]) / (rate * segment_duration))
    spec_data = []
    num_segs = num_segments if keep_last_incomplete_segment else num_segments - 1

    # Initialize the spectrogram transform
    hop_length = int((win_size*48000 - nfft) // (256 - 1)) # Calculate hop_length for 256 frames, hop_length 92
    spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=nfft, hop_length=hop_length, power=2, center=False).to(device)
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(device)

    # Prepare resampler if needed
    if rate != sample_rate:
        resampler = torchaudio.transforms.Resample(rate, sample_rate).to(device)

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

        # Compute the spectrogram of the audio segments in batch
        spec = spectrogram_transform(batch_tensor)
        spec_db = amplitude_to_db(spec)

        # Process each spectrogram in the batch
        for spectrum in spec_db:
            spectrum = spectrum.detach().cpu().numpy()
            if spectrum.shape != (257, 1299):  # (frequency, time)
                logger.info(f"Invalid array shape: {spectrum.shape}")
                continue
            spec_data.append(spectrum.astype(np.float16))

        # Print progress information
        logger.info(f"Processed {batch_end}/{num_segments} segments")

    # Return the spectrogram data as a numpy array
    return np.array(spec_data, dtype=np.float16)

def create_spec_data(wav_file, segment_duration=2.5, sample_rate=48000, nfft=512, win_size=0.5, keep_last_incomplete_segment=False):
    """
    Generates spectrogram data from an input WAV file and saves it to a specified location.

    This function reads an audio file, segments it into chunks of specified duration,
    resamples if necessary, computes the spectrogram for each segment, and then saves
    the collection of spectrograms to a file.
    The function returns the spectrogram data as a NumPy array in 16-bit float format

    Args:
    wav_file (str): Path to the input WAV file.
    segment_duration (float): Duration of each audio segment to be analyzed, in seconds. Default is 2.5 seconds.
    sample_rate (int): Sampling rate to which the audio should be resampled. Default is 48,000 Hz.
    nfft (int): Number of data points used in each block for the FFT. Default is 512.
    win_size(float): The window size of an element in a batch, where each long segment will form a batch in evaluation. Default 0.5.
    keep_last_incomplete_segment (bool): keep the last incomplete segment or not (experimental). Default False.

    Note:
    The evaluation dataloader currently only supports that 1) each batch element with the window size of 500ms and 2) the segment_duration of 2500ms
    """
    logger.info(f"Processing wav audio file: {wav_file}...")
    waveform, rate = torchaudio.load(wav_file)
    waveform = waveform.to(device)
    num_segments = int(len(waveform[0]) / (rate * segment_duration))

    spec_data = []
    num_segs = num_segments if keep_last_incomplete_segment else num_segments - 1
    for i in range(num_segs): # Remove the last segment that might be incomplete
        # Extract the audio segment
        start = int(rate * segment_duration * i)
        end = int(rate * segment_duration * (i + 1))
        signal_piece = waveform[:, start:end]

        # Resample the audio segment if necessary
        if rate != sample_rate:
            resampler = torchaudio.transforms.Resample(rate, sample_rate).to(device)
            signal_piece = resampler(signal_piece)

        # Compute the spectrogram of the audio segment
        # Got spectrogram images of size 257x256 (freq, time) for 500ms segments from the 2500ms long segment as a batch
        # win_size = 0.5 # 500ms
        hop_length = int((win_size*48000 - nfft) // (256 - 1)) # Calculate hop_length for 256 frames, hop_length 92
        spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=nfft, hop_length=hop_length, power=2, center=False).to(device)
        spec = spectrogram_transform(signal_piece) # Compute the spectrogram
        spec_db = torchaudio.transforms.AmplitudeToDB().to(device)(spec) # Convert power spectrogram to dB scale (for visualization purposes)
        spectrum = spec_db[0].detach().cpu().numpy() # The first dimension is the batch size

        if spectrum.shape != (257, 1299): # (frequency, time)
            logger.info(f"Invalid array shape: {spectrum.shape}")
            continue
        spec_data.append(spectrum.astype(np.float16))


        # Print progress information
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{num_segments} segments")

    # Return the spectrogram data as a numpy array
    return np.array(spec_data, dtype=np.float16)

class OneStreamCNNModel(nn.Module):
    """
    A one-stream convolutional neural network model.

    Args:
    dropout_rate (float, optional): The dropout rate. Defaults to 0.5.
    num_classes (int): The number of categories for the classification
    """

    def __init__(self, num_classes, dropout_rate=0.5):
        super(OneStreamCNNModel, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        # Convolutional stream
        self.conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(8 * 8 * 64, 1024)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc2 = nn.Linear(1024, self.num_classes)

    def forward(self, x, y=None, mode='train'):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor for the one stream, of shape (batch_size, 257, 256).
            y (torch.Tensor, optional): Ground truth labels for output, of shape (batch_size,). Defaults to None.
            mode (str, optional): The mode of operation ('train' or 'eval'). Defaults to 'train'.

        Returns:
            If y is provided:
                tuple: A tuple containing:
                    - probs (torch.Tensor): Probabilities for the output, of shape (batch_size, num_classes).
                    - loss (torch.Tensor): The computed loss.
                    - accuracy (list): A list of accuracies for each sample in the batch.
            If y is not provided:
                    - probs (torch.Tensor): Probabilities for the output, of shape (batch_size, num_classes).
        """
        # Pass input x through the convolutional stream
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.conv(x)
        x = x.view(x.size(0), -1)

        # Pass the features through the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Obtain the logits for the output
        logits = self.fc2(x)

        # Compute the probabilities using softmax
        probs = F.softmax(logits, dim=1)

        # If ground truth labels are provided, compute loss and accuracy
        if y is not None:
            y = y.type(torch.LongTensor).to(y.device)
            loss = F.cross_entropy(logits, y)
            classes = logits.argmax(dim=1)
            accuracy = (classes == y).float()
            return probs, loss, accuracy.tolist()

        # If ground truth labels are not provided, return only the probabilities
        return probs

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

    A window shift of 26 pixels is around 50ms.
    26 * frame_shift = 26 * (92/48000*1000) = 49.833333ms

    # Got a spectrogram image of size 257x256 (freq, time) for a 500ms segment
    # A) To make num_freqs close to 256
    # nfft = 512 => num_freqs = 512/2 + 1 = 257 (1 for the origin, div by 2 for symmetricity of FFT)
    # B) To make num_frames close to 256
    # size = nfft = 512 (samples)
    # shift = (500/1000*48000 - 512) / (256-1) = 92.1 ~ 92,
    # where -1 in "(256 - 1)" means that having placed the last frame, calculate the distance between other adjoining frames
    #
    # Frame size = 512/48000*1000 = 10.7ms
    # Frame shift = 92/48000*1000 = 1.916ms
    # Frame size 10.7ms and shift 1.916ms with 82% ((10.7-1.916)/10.7) overlap
    # (500-10.7)/1.926 + 1 = 255.0498

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
        model (OneStreamCNNModel): The trained model used for prediction.
        pred_data (str or numpy.array): Path to the file containing spectrogram arrays for prediction or specgram array itself.
        model_path (str): Path to the saved model to be used for prediction.
        avg_pred_win (int): Size of the averaging window for predictions.
        batch_size (int): Minibatch size for prediction.

    Returns:
        numpy.ndarray: The predicted probabilities.
    """
    logger.info(args)

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
        preds = model(torch.from_numpy(inputs).to(device))
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
    logger.info(f"Used the model: {eval_model}")

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
    assert len(label2id) == len(id2label), f"label and id in label2id file '{args.label2id_yaml}' should be one-to-one correspondence"

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

def extract_spectrogram_segments_batch(audio_file, window_size=0.5, window_shift=0.15, nfft=512, sample_rate=48000, feat_batch_size=32):
    """
    Extract spectrogram segments from an audio file using a sliding window, processing in batches.
    Args:
        audio_file (str): Path to the audio file (.wav).
        window_size (float): The size of the sliding window in seconds (default: 0.5).
        window_shift (float): The shift of the sliding window in seconds (default: 0.15).
        nfft (int): The number of FFT points (default: 512).
        sample_rate (int): The desired sample rate of the audio (default: 48000).
        feat_batch_size (int): Number of signal pieces to process in a batch during feature extraction (default: 32).
    Returns:
        np.ndarray: Spectrogram segments of shape (num_segments, 257, 256).
    """
    data, rate = torchaudio.load(audio_file)
    data = data.to(device)
    
    # Resample the audio if the file has a rate that is not 48kHz
    if rate != sample_rate:
        resampler = torchaudio.transforms.Resample(rate, sample_rate).to(device)
        data = resampler(data)
    
    # Calculate the number of segments based on the audio duration and window parameters
    audio_duration = data.shape[1] / sample_rate
    num_segments = int((audio_duration - window_size) / window_shift) + 1
    
    # Initialize the spectrogram transform
    hop_length = int((window_size * sample_rate - nfft) // (256 - 1))  # Calculate hop_length for 256 frames
    spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=nfft, hop_length=hop_length, power=2, center=False).to(device)
    amplitude_to_db = torchaudio.transforms.AmplitudeToDB().to(device)
    
    input_spec = []
    window_samples = int(window_size * sample_rate)
    
    for batch_start in range(0, num_segments, feat_batch_size):
        batch_end = min(batch_start + feat_batch_size, num_segments)
        batch_signals = []
        
        for i in range(batch_start, batch_end):
            # Take a window_size piece out of the data with a step size of window_shift
            start_index = int(sample_rate * i * window_shift)
            end_index = start_index + window_samples
            signal_piece = data[:, start_index:end_index]
            
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
        
        # Create the spectrograms and scale them logarithmically
        specs = spectrogram_transform(batch_tensor)
        specs_db = amplitude_to_db(specs)
        
        # Process each spectrogram in the batch
        for spec in specs_db:
            result = spec.detach().cpu().numpy()
            # Skip this piece if the shape is not as expected
            if np.shape(result) != (257, 256):  # (frequency, time)
                logger.info(f"Invalid shape: {np.shape(result)}")
                continue
            input_spec.append(np.array(result, dtype=np.float16))
        
        logger.info(f"Processed {batch_end}/{num_segments} segments")
    
    input_spec = np.array(input_spec, dtype=np.float16)
    logger.info(f'Number of spectrogram segments: {len(input_spec)}')
    return input_spec

# Dataloader for different prediction resolutions
def extract_spectrogram_segments(audio_file, window_size=0.5, window_shift=0.15, nfft=512, sample_rate=48000):
    """
    Extract spectrogram segments from an audio file using a sliding window.

    Args:
        audio_file (str): Path to the audio file (.wav).
        window_size (float): The size of the sliding window in seconds (default: 0.5).
        window_shift (float): The shift of the sliding window in seconds (default: 0.15).
        nfft (int): The number of FFT points (default: 512).
        sample_rate (int): The desired sample rate of the audio (default: 48000).

    Returns:
        np.ndarray: Spectrogram segments of shape (num_segments, 257, 256).

    Example:
        audio_file = "/path/to/audio.wav"
        input_spec = extract_spectrogram_segments(audio_file)
        print(f"{input_spec.shape=}")  # input_spec.shape=(num_segments, 257, 256)
    """
    data, rate = torchaudio.load(audio_file)
    data = data.to(device)

    # Resample the audio if the file has a rate that is not 48kHz
    if rate != sample_rate:
        resampler = torchaudio.transforms.Resample(rate, sample_rate).to(device)
        data = resampler(data)

    # Calculate the number of segments based on the audio duration and window parameters
    audio_duration = data.shape[1] / sample_rate
    num_segments = int((audio_duration - window_size) / window_shift) + 1

    # Initialize the spectrogram transform
    hop_length = int((window_size * sample_rate - nfft) // (256 - 1))  # Calculate hop_length for 256 frames
    spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=nfft, hop_length=hop_length, power=2, center=False).to(device)

    input_spec = []
    for i in range(num_segments):
        # Take a window_size piece out of the data with a step size of window_shift
        start_index = int(sample_rate * i * window_shift)
        end_index = int(sample_rate * (i * window_shift + window_size))
        signal_piece = data[:, start_index:end_index]

        # Create the spectrogram and scale it logarithmically
        spec = spectrogram_transform(signal_piece)
        spec_db = torchaudio.transforms.AmplitudeToDB().to(device)(spec)
        result = spec_db[0].detach().cpu().numpy()

        # Skip this piece if the shape is not as expected
        if np.shape(result) != (257, 256): # (frequency, time)
            logger.info(f"Invalid shape: {np.shape(result)}")
            continue

        input_spec.append(np.array(result, dtype=np.float16))

        if i % 1000 == 0:
            logger.info(f"Processing the {i}th segment...")

    input_spec = np.array(input_spec, dtype=np.float16)
    logger.info(f'Number of spectrogram segments: {len(input_spec)}')

    return input_spec

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
        model (OneStreamCNNModel): The trained model used for prediction.
        dataloader (torch.utils.data.DataLoader): The dataloader created by create_dataloader.
        model_path (str): Path to the saved model to be used for prediction.
        avg_pred_win (int): Size of the averaging window for predictions.
        batch_size (int): Minibatch size for prediction.

    Returns:
        numpy.ndarray: The predicted probabilities.
    """
    logger.info(args)

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
            preds = model(inputs)
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

def predict_segments_v1(pred_prob, label2id, window_size=0.5, window_shift=0.05, middle_part=0.05, fixed_factor=0.1):
    """
    Predict labels from the given prediction probabilities and return the segments

    Args:
        pred_prob (np.array): Numpy array that contains prediction probabilities.
        label2id (dict): dict from label to id
        window_size (float): The size of the sliding window in seconds (default: 0.5).
        window_shift (float): The shift of the sliding window in seconds (default: 0.05).
        middle_part (float): The proportion of the middle part of the window in seconds for label assignment (default: 0.05).
        fixed_factor (float): A fixed factor added to the start and end times of segments (default: 0.1).

    Returns:
        raw_segments, merged_segments
        list: A list of raw segments, where each segment is a tuple (start_time, end_time, label).
        list: A list of merged segments, where each segment is a tuple (start_time, end_time, label).
    """
    id2label = {id: label for label, id in label2id.items()}
    assert len(label2id) == len(id2label), f"label and id in label2id file '{args.label2id_yaml}' should be one-to-one correspondence"

    raw_segments = []
    for i, pred in enumerate(pred_prob):
        # Get the index of the maximum prediction
        pred_label_idx = np.argmax(pred)

        current = i
        start_window = current * window_shift
        end_window = start_window + window_size
        middle_start = start_window + (window_size - middle_part) / 2
        middle_end = start_window + (window_size + middle_part) / 2

        if pred_label_idx != label2id['noise']:  # Check if the prediction is not 'noise'
            start_time = middle_start + fixed_factor  # fixed factor might consider 1) shift of batch elements of 2500ms long segment that is not exactly 50ms; 2) prob avg window effects
            end_time = middle_end + fixed_factor
            label = id2label[pred_label_idx]
            raw_segments.append((round(start_time, 6), round(end_time, 6), label))

    if not raw_segments:
        print("\nWarning: all predictions are noises\n")
        return [], []

    merged_segments = merge_segments(raw_segments)
    return raw_segments, merged_segments

default_model = "/home/bin-wu/workspace/projects/clib/egs/riken/riken_cnn_s0/exp/sys/riken2024/division_jay_half_winmid0.05size0.5shift0.05_noisekeep5/cnn-run0/bs2048lr0.0003lrdecay1avgpredwin5/train/model.ckpt"
default_wav_file = "/home/bin-wu/share/data/riken/sample/1100F_0124_2017_0s_10s_ch1.wav"
# default_wav_file = "/home/bin-wu/share/data/riken/riken2024/annotation_examples/wav/akiko_variant.wav"
# default_wav_file = "/home/bin-wu/share/data/riken/riken2024/annotation_examples/wav/jay_balanced.wav"
# default_wav_file = "/home/bin-wu/share/data/riken/riken2024/annotation_examples/wav/nakanishi_balanced.wav"
# default_wav_file = "/home/bin-wu/share/data/riken/riken2024/jay_family/wav/230807_001_ch1.wav"
# default_wav_file = "/data/share/bin-wu/data/marmoset/vocalization/marmoset_mit/data/pair1/pair1_animal1_together.wav"

parser = argparse.ArgumentParser(description="Convert from a audio to its segment file using riken cnn model (the one-stream implementation of a CNN from [Oikarinen, 2019].  Note that end segments might be discarded in prediction when creating spectra input or averaging the predicted windows. Consider --complete option if end segments are important. Note that the high resolution prediction with high --pred_res would be slow becauseits dataloader is not optimized.")
# Main arguments
parser.add_argument("--eval_model", type=str, default=default_model, help="Model path for prediction or evaluation")
parser.add_argument("--wav_file", type=str, default=default_wav_file, help="Path of test wav")
parser.add_argument("--out_dir", type=str, default="./exp/out", help="Output directory to store predicted segment file if the out_seg_file is not specified.")
parser.add_argument("--out_seg_file", type=str, default=None, help="Output path of the predicted segment file.")
# Data arguments
parser.add_argument("--batch_size", type=int, default=25, help="Batch size for the dataloader")
parser.add_argument("--feat_batch_size", type=int, default=25, help="Feat batch size for the extracting features")
# Dictionary arguments
parser.add_argument("--label2id_yaml", type=str, default="conf/dict/label2id.yaml", help="The YAML file that contains the label-to-labelID mapping.")
# Evaluation arguments
parser.add_argument("--avg_pred_win", type=int, default=5, help="Collect predicted probabilities by averaging across x consecutive predictions")
# Other arguments
parser.add_argument("--pred_resolution", type=float, default=0.01, help="The resolution of the prediction. The final segments are merged from predicted labels of x-seconds middle parts of a sliding window with the window shift x-seconds. The resolution would be safe when less than 0.05 seconds. Default 0.01")
parser.add_argument("--complete", action="store_true", help="May have bugs, keep the last potentially incomplete segment (used by fast prediction)")
parser.add_argument("--fast_pred", action="store_true", help="Fast prediction using 50ms prediction resolution by creating a dataloader from 2500ms segments")

args = parser.parse_args()

# General parameters
eval_model = args.eval_model
wav_file = args.wav_file
out_dir = args.out_dir
label2id = OmegaConf.load(args.label2id_yaml)
batch_size = args.batch_size

# Prediction parameters
avg_pred_win = args.avg_pred_win

# Set seed and device
set_seed(2020)
device = set_device("auto")
logger.info(f"Device: {device}")

# Predict the segments
model = OneStreamCNNModel(num_classes=len(label2id)).to(device)

if (args.fast_pred):
    logger.info("Using the fast prediction with the 50ms prediction resolution by creating a dataloader from the 2500ms segments...")
    # spec = create_spec_data(wav_file, keep_last_incomplete_segment=args.complete)
    spec = create_spec_data_batch(wav_file, keep_last_incomplete_segment=args.complete, feat_batch_size=args.feat_batch_size)
    pred_prob = predict(model, spec, model_path=eval_model, avg_pred_win=avg_pred_win)
    _, merged_segments = predict_segments(pred_prob, label2id)
else:
    # spec = extract_spectrogram_segments(wav_file, window_size=0.5, window_shift=args.pred_resolution)
    spec = extract_spectrogram_segments_batch(wav_file, window_size=0.5, window_shift=args.pred_resolution, feat_batch_size=args.feat_batch_size)
    test_loader = create_dataloader(spec, batch_size, train=False)
    pred_prob = predict_from_dataloader(model, test_loader, model_path=eval_model, avg_pred_win=avg_pred_win)
    fixed_factor = 0.1 * (args.pred_resolution/0.05) # reduce the fixed factor when resolution higher that 0.1 is default for 0.05 resolution
    _, merged_segments = predict_segments_v1(pred_prob, label2id, window_size=0.5, window_shift=args.pred_resolution, middle_part=args.pred_resolution, fixed_factor=fixed_factor)

# Save the segments
if args.out_seg_file:
    os.makedirs(os.path.dirname(args.out_seg_file), exist_ok=True)
else:
    os.makedirs(out_dir, exist_ok=True)

file_name, _ = os.path.splitext(os.path.basename(wav_file))
if args.fast_pred:
    merged_seg_file = args.out_seg_file or os.path.join(out_dir, f"cnn_pred_{file_name}.txt") # When out_file not specified, use out_dir
else:
    merged_seg_file = args.out_seg_file or os.path.join(out_dir, f"cnn_pred_resol{args.pred_resolution}_{file_name}.txt") # When out_file not specified, use out_dir

# Open the output file in write mode with UTF-8 encoding for Audacity compatibility
with codecs.open(merged_seg_file, 'w', 'utf-8') as f_merged:
    for start_time, end_time, label in merged_segments:
        f_merged.write(f"{start_time}\t{end_time}\t{label}\n")

logger.info(f"Segment file saved at: {os.path.abspath(merged_seg_file)}")

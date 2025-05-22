# Implemented by bin-wu at 20:37 on 16 April 2024
import os
import sys
import random
import re

import logging
# Configure the logging system
logging.basicConfig(
    level=logging.INFO,
    format="[ %(asctime)s | %(filename)s | %(levelname)s ] %(message)s",
    datefmt="%d/%m/%Y %H:%M:%S"
)

# Create a logger object
logger = logging.getLogger(__name__)

import argparse
import json

import GPUtil
from omegaconf import OmegaConf

import numpy as np
import torchaudio
import torch

def init_logger(file_name="", stream="stdout"):
    """ Initialize a logger to terminal and file at the same time. """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[ %(asctime)s | %(filename)s | %(levelname)s ] %(message)s", "%d/%m/%Y %H:%M:%S")

    logger.handlers = [] # Clear existing stream and file handlers
    if stream == "stdout":
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if file_name:
        file_handler = logging.FileHandler(file_name, 'w') # overwrite the log file if exists
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

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

device = set_device("auto")
print(f"Device: {device}")

def process_audio_segment(audio_file, segment_file, label2id, middle_part=0.15, window_size=0.5, window_shift=0.15, noise_preserve_steps=5):
    """
    Processes a single audio segment and its corresponding label file.
    Creates a mel spectrogram of the audio and generates target labels.

    Split audio into chunks using a sliding window. Assign a target label
    to the chunk when the middle part of the chunk overlaps with annotated segments.
    Extract 257x256 mel spectrogram for each chunk as input.
    Note that we only keep every noise_preserve_steps-th all-noise-no-label chunk.

    Args:
        audio_file (str): Path to the audio file (.wav).
        segment_file (str): Path to the segment label file (.txt).
        label2id (dict): Mapping of label names to their corresponding IDs.
        middle_part (float): The proportion of the middle part of the window in seconds for label assignment (default: 0.15).
        window_size (float): The size of the sliding window in seconds for label assigment (default: 0.5).
        window_shift (float): The shift of the sliding window in seconds for label assigment (default: 0.15).
        noise_preserve_steps (int): The number of steps to skip between preserved all-noise-no-label chunks (default: 5).

    Returns:
        tuple: A tuple containing the input mel spectrogram and target labels.
            - input_spec (np.ndarray): Input mel spectrogram of shape (num_chunks, 257, 256).
            - target_labels (np.ndarray): Target labels of shape (num_chunks).

    Example:
        # import matplotlib as mpl
        # mpl.use('Agg') # Necessary when running on a server (not needed on a jupyter notebook)

        import os
        import numpy as np
        from scipy.io import wavfile
        from scipy import signal
        import matplotlib.pyplot as plt

        label2id = {'cha': 0, 'chi': 1, 'ek': 2, 'ph': 3, 'ts': 4, 'tr': 5, 'trph': 6, 'tw': 7, 'noise': 8,
                    'tw2': 9, 'trph2': 10, 'tr2': 11, 'ts2': 12, 'ph2': 13, 'ek2': 14, 'chi2': 15, 'cha2': 16}
        audio_file = "/home/bin-wu/share/data/riken/sample/1100F_0124_2017_0s_10s_ch1.wav"
        segment_file = "/home/bin-wu/share/data/riken/sample/1100F_0124_2017_0s_10s_ch1.txt"
        input_spec, target = process_audio_segment(audio_file, segment_file, label2id)
        print(f"{input_spec.shape=}, {target.shape=}") # input_spec.shape=(52, 257, 256), target.shape=(52)
    """
    logger.info(f"Processing segment label file: {segment_file}...")
    lines = []
    first = float('inf')  # The chunk index of the first label

    with open(segment_file, 'r') as labels:
        # segments = [line.strip().split('\t') for line in labels if line.strip().split('\t')[0] != "\\"]  # Some audacity files contain lines of "\ min_freq max_freq"
        segments = [re.split(r"\s+", line.strip()) for line in labels if re.split(r"\s+", line.strip())[0] != "\\"]  # Some audacity files contain lines of "\ min_freq max_freq"

    current = 0
    # The middle part of the window
    start_window = current * window_shift
    end_window = start_window + window_size
    middle_start = start_window + (window_size - middle_part) / 2
    middle_end = start_window + (window_size + middle_part) / 2

    for seg in segments:
        if len(seg) != 3:
            logger.warning(f"Warning: ignoring the segment '{seg}', right format should be 'begin_sec, end_sec, label'")
        else:
            start_t, end_t, label = seg

        # Slide a window with size window_size and shift window_shift.
        # When the middle part of the current window overlaps with the label segment,
        # assign the label to the window; otherwise, assign "noise".
        start_t, end_t = float(start_t), float(end_t)

        # Move the window until it reaches the current segment or overlaps with it
        while max(start_t, middle_start) > min(end_t, middle_end) and start_t > middle_end:
            lines.append("noise")
            current += 1
            start_window = current * window_shift
            end_window = start_window + window_size
            middle_start = start_window + (window_size - middle_part) / 2
            middle_end = start_window + (window_size + middle_part) / 2

        # Assign the label to the window while it overlaps with the segment
        while max(start_t, middle_start) <= min(end_t, middle_end):  # Overlap
            if current < first:
                first = current  # Record the first window overlaps with the label segment
            lines.append(label.rstrip().lower())  # Remove the empty space or the remaining /r from window's newline (/r/n) after the label.
            current += 1
            start_window = current * window_shift
            end_window = start_window + window_size
            middle_start = start_window + (window_size - middle_part) / 2
            middle_end = start_window + (window_size + middle_part) / 2

    length = len(lines)
    target = []
    for line_label in lines:
        target.append(label2id.get(line_label, label2id['noise'])) # Treat the label that is not in label2id dict as 'noise'

    nfft = 1024  # Using 1024 for better low-frequency resolution
    data, rate = torchaudio.load(audio_file)
    data = data.to(device)

    input_spec = []
    target_labels = []
    logger.info(f"Processing wav audio file: {audio_file}...")
    logger.info(f"Creating wav chunks from the chunk index of the first label: {first} to the chunk index of the last label: {length}...")

    # Using 16kHz sample rate instead of 48kHz
    sample_rate = 16000

    for i in range(first, length):
        # Create the mel spectrogram if the current piece is not labeled as noise
        # Or pick every noise_preserve_steps-th piece that is labeled as noise
        if target[i] != label2id['noise'] or i % noise_preserve_steps == 0:
            # Take a window_size piece out of the data with a step size of window_shift
            start_index = int(rate * i * window_shift)
            end_index = int(rate * (i * window_shift + window_size))
            signal_piece = data[:, start_index:end_index]

            # Resample the signal piece to 16kHz
            if rate != sample_rate:
                resampler = torchaudio.transforms.Resample(rate, sample_rate).to(device)
                signal_piece = resampler(signal_piece)

            # Create the mel spectrogram and scale it logarithmically
            # Got a mel spectrogram image of size 257x256 (freq, time) for a 500ms segment
            # A) To make num_freqs close to 256
            # nfft = 1024 => num_freqs = 1024/2 + 1 = 513 (1 for the origin, div by 2 for symmetricity of FFT)
            # But we use n_mels=257 to match the original dimension
            # B) To make num_frames close to 256
            # size = nfft = 1024 (samples)
            # shift = (500/1000*16000 - 1024) / (256-1) = ~29.8
            # where -1 in "(256 - 1)" means that having placed the last frame, calculate the distance between other adjoining frames
            #
            # Frame size = 1024/16000*1000 = 64ms
            # Frame shift = 29.8/16000*1000 = 1.86ms
            # Frame size 64ms and shift 1.86ms with 97% ((64-1.86)/64) overlap
            # (500-64)/1.86 + 1 = ~235
            # Initialize the MelSpectrogram transform
            hop_length = int((window_size*sample_rate - nfft) // (256 - 1)) # Calculate hop_length for 256 frames

            mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=nfft,
                hop_length=hop_length,
                n_mels=257,
                power=2,
                center=False
            ).to(device)

            mel_spec = mel_spectrogram_transform(signal_piece)
            mel_spec_db = torchaudio.transforms.AmplitudeToDB().to(device)(mel_spec)
            result = mel_spec_db[0].detach().cpu().numpy()

            # Crop to ensure exactly 256 time frames # 16k would be (257, 259)
            if result.shape[1] > 256:
                result = result[:, :256]

            # Skip this piece if the shape is not as expected
            if np.shape(result) != (257, 256): # (frequency, time)
                logger.info(f"Invalid shape: {np.shape(result)}")
                continue

            input_spec.append(np.array(result, dtype=np.float16))
            target_labels.append(target[i])

            if i % 1000 == 0:
                logger.info(f"Processing the {i}th chunk...")

    input_spec = np.array(input_spec, dtype=np.float16)
    target_labels = np.array(target_labels, dtype=np.float16)
    logger.info(f'Length of dataset: {len(input_spec)}')

    return input_spec, target_labels

def process_audio_segments(audio_files, segment_files, label2id, middle_part=0.15, window_size=0.5, window_shift=0.15, noise_preserve_steps=5):
    """
    Processes multiple audio segments and their corresponding label files.
    Calls the process_audio_segment function for each pair of audio and segment files,
    and concatenates the results.

    Args:
        audio_files (list): List of paths to the audio files (.wav).
        segment_files (list): List of paths to the segment label files (.txt).
        label2id (dict): Mapping of label names to their corresponding IDs.
        middle_part (float): The proportion of the middle part of the window in seconds for label assignment (default: 0.15).
        window_size (float): The size of the sliding window in seconds for label assigment (default: 0.5).
        window_shift (float): The shift of the sliding window in seconds for label assigment (default: 0.15).
        noise_preserve_steps (int): The number of steps to skip between preserved all-noise-no-label chunks (default: 5).

    Returns:
        tuple: A tuple containing the concatenated input spectrograms and target labels.
            - input_spec (np.ndarray): Concatenated input spectrograms of shape (total_chunks, 257, 256).
            - target_labels (np.ndarray): Concatenated target labels of shape (total_chunks).

    Example:
        >>> label2id = {'cha': 0, 'chi': 1, 'ek': 2, ..., 'cha2': 16}
        >>> audio_files = ["/path/to/audio1.wav", "/path/to/audio2.wav"]
        >>> segment_files = ["/path/to/segment1.txt", "/path/to/segment2.txt"]
        >>> input_spec, target_labels = process_audio_segments(audio_files, segment_files, label2id)
        >>> input_spec.shape
        (104, 257, 256)
        >>> target_labels.shape
        (104)
    """
    input_spec_list = []
    target_labels_list = []
    for audio_file, segment_file in zip(audio_files, segment_files):
        input_spec, target_labels = process_audio_segment(audio_file, segment_file, label2id,
                                                          middle_part=middle_part, window_size=window_size,
                                                          window_shift=window_shift, noise_preserve_steps=noise_preserve_steps)
        input_spec_list.append(input_spec)
        target_labels_list.append(target_labels)

    return np.concatenate(input_spec_list, axis=0), np.concatenate(target_labels_list, axis=0)

# # For mel ((257, 1444) => (257, 1299) for 16k)
# def create_spec_data(wav_file, segment_duration=2.5, sample_rate=16000, nfft=1024, win_size=0.5):
#     """
#     Generates mel spectrogram data from an input WAV file and saves it to a specified location.

#     This function reads an audio file, segments it into chunks of specified duration,
#     resamples to 16kHz if necessary, computes the mel spectrogram for each segment, and then saves
#     the collection of mel spectrograms to a file.
#     The function returns the mel spectrogram data as a NumPy array in 16-bit float format

#     Args:
#     wav_file (str): Path to the input WAV file.
#     segment_duration (float): Duration of each audio segment to be analyzed, in seconds. Default is 2.5 seconds.
#     sample_rate (int): Sampling rate to which the audio should be resampled. Default is 16,000 Hz.
#     nfft (int): Number of data points used in each block for the FFT. Default is 1024 for better low-frequency resolution.
#     win_size(float): The window size of an element in a batch, where each long segment will form a batch in evaluation. Default 0.5.

#     Note:
#     The evaluation dataloader currently only supports that 1) each batch element with the window size of 500ms and 2) the segment_duration of 2500ms
#     """
#     logger.info(f"Processing wav audio file: {wav_file}...")
#     waveform, rate = torchaudio.load(wav_file)
#     waveform = waveform.to(device)
#     num_segments = int(len(waveform[0]) / (rate * segment_duration))

#     spec_data = []
#     for i in range(num_segments - 1): # Remove the last segment that might be incomplete
#         # Extract the audio segment
#         start = int(rate * segment_duration * i)
#         end = int(rate * segment_duration * (i + 1))
#         signal_piece = waveform[:, start:end]

#         # Resample the audio segment to 16kHz
#         if rate != sample_rate:
#             resampler = torchaudio.transforms.Resample(rate, sample_rate).to(device)
#             signal_piece = resampler(signal_piece)

#         # Compute the mel spectrogram of the audio segment
#         # Got mel spectrogram images of size 257x256 (freq, time) for 500ms segments from the 2500ms long segment as a batch
#         # A) To make num_freqs close to 257
#         # nfft = 1024 => num_freqs = 1024/2 + 1 = 513, but we use n_mels=257 to match original dimensions
#         # B) To calculate appropriate hop_length for 2500ms segments with a win_size of 500ms
#         # Using same formula as in the 500ms segments for consistency
#         hop_length = int((win_size*sample_rate - nfft) // (256 - 1)) # Calculate hop_length using consistent approach

#         # Create the mel spectrogram
#         mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
#             sample_rate=sample_rate,
#             n_fft=nfft,
#             hop_length=hop_length,
#             n_mels=257,
#             power=2,
#             center=False
#         ).to(device)

#         mel_spec = mel_spectrogram_transform(signal_piece)
#         mel_spec_db = torchaudio.transforms.AmplitudeToDB().to(device)(mel_spec)
#         spectrum = mel_spec_db[0].detach().cpu().numpy()

#         # Crop to ensure expected time frames ((257, 1444) => (257, 1299) for 16k)
#         if spectrum.shape[1] > 1299:
#             spectrum = spectrum[:, :1299]
#         if spectrum.shape != (257, 1299): # (frequency, time)
#             logger.info(f"Invalid array shape: {spectrum.shape}")
#             continue

#         spec_data.append(spectrum.astype(np.float16))

#         # Print progress information
#         if (i + 1) % 100 == 0:
#             logger.info(f"Processed {i + 1}/{num_segments} segments")

#     # Return the mel spectrogram data as a numpy array
#     return np.array(spec_data, dtype=np.float16)

parser = argparse.ArgumentParser(description="Create datasets of audios and labels stored in numpy array format. Split audio into chunks using a sliding window. Assign a target label to the chunk when the middle part of the chunk overlaps with annotated segments. Extract 257x256 mel specgram for each 500ms-chunk as input. Note that we keep every nth all-noise-no-label chunk. The program also generates 2500ms long segments for each wav file of the test set. Using the long segments for the test set will save the data processing time.", epilog="Note that we extract the 500ms-segments ($output_dir/{train,dev,test}_{input,target}.npy) between the first and the last labels in the segment files. We extract all 2500ms-segments (test_input_${id}.npy) including beginning and ending silences for the prediction of the test set. Each 2500ms-segment will make a batch of 500ms chunks with a sliding window in the evaluation dataloader.")

parser.add_argument("--info_json", type=str, default="data/mit_sample0/info.json", help="JSON file that maps utterance ID to key-value pairs. Example format: {wav1_id: {wav: wav1_path, seg: seg_label1_path}, wav2_id: {wav: wav2_path, seg: seg_label2_path}}. Each line of the Audacity segment label file should be in the format 'begin_time_sec<tab>end_time_sec<tab>label'.")
parser.add_argument("--data_div_yaml", type=str, default="conf/data/division_sample.yaml", help="The data division YAML file containing lists of wave IDs for training, development, and test sets. Example format: train: [id1, id2], dev: [id3], test: [id4]")
parser.add_argument("--label2id_yaml", type=str, default="conf/dict/label2id_marmoset.yaml", help="The YAML file that contains the label-to-labelID mapping.")
parser.add_argument("--out_dir", type=str, default="exp/data/division_sample_winmid0.05size0.5shift0.05_noisekeep5", help="Output directory to store the spectra and labels for datasets.")

parser.add_argument("--middle_part", type=float, default=0.05, help="The proportion of the middle part of the window in seconds for label assigment (default: 0.05).")
parser.add_argument("--window_size", type=float, default=0.5, help="The size of the sliding window in seconds for label assigment (default: 0.5). Ok to modify the middle_part and window_shift. Be careful to modify window_size due to consistency to 2500ms chunks.")
parser.add_argument("--window_shift", type=float, default=0.05, help="The shift of the sliding window in seconds for label assigment (default: 0.05).")
parser.add_argument("--noise_preserve_steps", type=int, default=5, help="The number of steps to skip between preserved all-noise-no-label chunks. (1 means keeping all noise segments) (default:5).")

args = parser.parse_args()

save_dir = args.out_dir
if not os.path.exists(save_dir): os.makedirs(save_dir, exist_ok=True)
logger = init_logger(os.path.join(save_dir, "report.log"))

logger.info(args)

info = OmegaConf.load(args.info_json)
data = OmegaConf.load(args.data_div_yaml)
label2id = OmegaConf.load(args.label2id_yaml)

train_wav_files = [info[uttid]['wav'] for uttid in data['train']]
train_seg_files = [info[uttid]['seg'] for uttid in data['train']]
dev_wav_files = [info[uttid]['wav'] for uttid in data['dev']]
dev_seg_files = [info[uttid]['seg'] for uttid in data['dev']]
test_wav_files = [info[uttid]['wav'] for uttid in data['test']]
test_seg_files = [info[uttid]['seg'] for uttid in data['test']]

# Input spectra and target labels
train_input, train_target = process_audio_segments(train_wav_files,
                                                   train_seg_files,
                                                   label2id,
                                                   middle_part=args.middle_part,
                                                   window_size=args.window_size,
                                                   window_shift=args.window_shift,
                                                   noise_preserve_steps=args.noise_preserve_steps)
dev_input, dev_target = process_audio_segments(dev_wav_files,
                                               dev_seg_files,
                                               label2id,
                                               middle_part=args.middle_part,
                                               window_size=args.window_size,
                                               window_shift=args.window_shift,
                                               noise_preserve_steps=args.noise_preserve_steps)
test_input, test_target = process_audio_segments(test_wav_files,
                                                 test_seg_files,
                                                 label2id,
                                                 middle_part=args.middle_part,
                                                 window_size=args.window_size,
                                                 window_shift=args.window_shift,
                                                 noise_preserve_steps=args.noise_preserve_steps)

data_sets = [
    ('train_input.npy', train_input),
    ('train_target.npy', train_target),
    ('dev_input.npy', dev_input),
    ('dev_target.npy', dev_target),
    ('test_input.npy', test_input),
    ('test_target.npy', test_target)
 ]

# Save pectra and labels
for filename, filedata in data_sets:
    np.save(os.path.join(save_dir, filename), filedata)

# # Save spectra of 2500ms long segments for test audios
# id2data = {}
# for uttid in data['test']:
#     wav_file = info[uttid]['wav']
#     spec_data = create_spec_data(wav_file)
#     id2data[uttid] = spec_data
#     np.save(os.path.join(save_dir, f"test_input_{uttid}.npy"), spec_data)

# Print out the shapes
for filename, filedata in data_sets:
    logger.info(f"{os.path.join(save_dir, filename)} with the shape of {filedata.shape}")
# for uttid in data['test']:
#     logger.info(f"{os.path.join(save_dir, f"test_input_{uttid}.npy")} with the shape of {id2data[uttid].shape}")

logger.info('Data saved successfully.')

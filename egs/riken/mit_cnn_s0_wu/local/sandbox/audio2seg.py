"""Implemented by bin-wu at 8:00 on 2024/11/14"""
import argparse
import os

import math
import random
import argparse

import GPUtil
import codecs

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

import torch
import torch.nn as nn
import torch.nn.functional as F

truth_values=['cha','chi','ek','ph','ts','tr','trph','tw','noise']

def set_seed(seed):
    """
    Set the seed for reproducibility across NumPy and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

def create_spec_data(wav_file, save_loc=None, segment_duration=2.5, sample_rate=48000, nfft=512, noverlap=420):
    # Read the WAV file
    rate, data = wavfile.read(wav_file)
    num_segments = int(len(data) / (rate * segment_duration))

    spec_data = []
    for i in range(num_segments - 1): # Remove the last segment that might be incomplete
        # Extract the audio segment
        start = int(rate * segment_duration * i)
        end = int(rate * segment_duration * (i + 1))
        signal_piece = data[start:end]

        # Resample the audio segment if necessary
        if rate != sample_rate:
            signal_piece = signal.resample(signal_piece, int(len(signal_piece) * sample_rate / rate))

        # Compute the spectrogram of the audio segment
        spectrum, freqs, time, image = plt.specgram(signal_piece, NFFT=nfft, Fs=sample_rate,
                                                    window=np.hamming(nfft), noverlap=noverlap,
                                                    scale='linear', detrend='none')
        spectrum = 10 * np.log(np.abs(spectrum + 1e-6))
        if spectrum.shape != (257, 1299):
            print(f"Invalid array shape: {spectrum.shape}")
            continue
        spec_data.append(spectrum.astype(np.float16))

        plt.clf()

        # Print progress information
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{num_segments} segments")

    # Convert the spectrogram data to a numpy array
    spec_data = np.array(spec_data, dtype=np.float16)

    return spec_data

class TwoStreamCNNModel(nn.Module):
    """
    A two-stream convolutional neural network model.

    Args:
        dropout_rate (float, optional): The dropout rate. Defaults to 0.5.
    """

    def __init__(self, dropout_rate=0.5):
        super(TwoStreamCNNModel, self).__init__()
        self.dropout_rate = dropout_rate

        # First convolutional stream
        self.conv1 = nn.Sequential(
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

        # Second convolutional stream
        self.conv2 = nn.Sequential(
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
        self.fc1 = nn.Linear(8 * 8 * 64 * 2, 1024)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc2 = nn.Linear(1024, 9)
        self.fc3 = nn.Linear(1024, 9)

    def forward(self, x1, x2, y1=None, y2=None, mode='train'):
        """
        Forward pass of the model.

        Args:
            x1 (torch.Tensor): Input tensor for the first stream, of shape (batch_size, 257, 256).
            x2 (torch.Tensor): Input tensor for the second stream, of shape (batch_size, 257, 256).
            y1 (torch.Tensor, optional): Ground truth labels for the first output, of shape (batch_size,). Defaults to None.
            y2 (torch.Tensor, optional): Ground truth labels for the second output, of shape (batch_size,). Defaults to None.
            mode (str, optional): The mode of operation ('train' or 'eval'). Defaults to 'train'.

        Returns:
            If y1 and y2 are provided:
                tuple: A tuple containing:
                    - probs1 (torch.Tensor): Probabilities for the first output, of shape (batch_size, 9).
                    - probs2 (torch.Tensor): Probabilities for the second output, of shape (batch_size, 9).
                    - loss (torch.Tensor): The computed loss.
                    - accuracy (list): A list of accuracies for each sample in the batch.
            If y1 and y2 are not provided:
                tuple: A tuple containing:
                    - probs1 (torch.Tensor): Probabilities for the first output, of shape (batch_size, 9).
                    - probs2 (torch.Tensor): Probabilities for the second output, of shape (batch_size, 9).
        """
        # Pass input x1 through the first convolutional stream
        x1 = x1.unsqueeze(1)  # Add channel dimension
        x1 = self.conv1(x1)
        x1 = x1.view(x1.size(0), -1)

        # Pass input x2 through the second convolutional stream
        x2 = x2.unsqueeze(1)  # Add channel dimension
        x2 = self.conv2(x2)
        x2 = x2.view(x2.size(0), -1)

        # Concatenate the outputs of the two convolutional streams
        x = torch.cat((x1, x2), dim=1)

        # Pass the concatenated features through the fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x) if mode == 'train' else x

        # Obtain the logits for the two outputs
        logits1 = self.fc2(x)
        logits2 = self.fc3(x)

        # Compute the probabilities using softmax
        probs1 = F.softmax(logits1, dim=1)
        probs2 = F.softmax(logits2, dim=1)

        # If ground truth labels are provided, compute loss and accuracy
        if y1 is not None and y2 is not None:
            loss1 = F.cross_entropy(logits1, y1)
            loss2 = F.cross_entropy(logits2, y2)
            loss = loss1 + loss2

            classes1 = logits1.argmax(dim=1)
            classes2 = logits2.argmax(dim=1)
            accuracy = ((classes1 == y1) & (classes2 == y2)).float()

            return probs1, probs2, loss, accuracy.tolist()

        # If ground truth labels are not provided, return only the probabilities
        return probs1, probs2
    
def pred_input_function(xs1, xs2, i, window_size=256, step_size=26):
    """
    Creates a minibatch of 50 500ms-spectrograms from an 2500ms-spectrogram using a sliding window
    with the window size of 500ms and the window shift of 50ms.

    This function takes two sequences of spectral segments (xs1 and xs2) and an index (i) representing
    the current position in the sequences. It extracts 50 elements, each of 500ms spectrogram with
    a shape of (257, 256), for the current position of a 2500ms-spectrogram with a shape of (257, 1299).

    The extraction process is done using a sliding window approach with a window size of 256 and a
    step size of 26 (the step size or window shift of 26 pixel is from (floor(1299/50)).
    The first num_complete_elements can be extracted entirely from the current 2500ms-segment, while
    the remaining elements require concatenation with the next segment.


    Args:
        xs1 (array-like): Input spectral segments for the first stream, of shape (*, 257, 1299).
        xs2 (array-like): Input spectral segments for the second stream, of shape (*, 257, 1299).
        i (int): The index of the current spectral segment.
        window_size (int): The size of each element (default: 256).
        step_size (int): The step size for sliding the window (default: 26).

    Returns:
        dict: A dictionary containing the i-th minibatch of input features.
              Keys: 'x', 'x2'. Values: NumPy arrays of shape (50, 257, 256) for each.
              The batch_size is fixed as 50.

    Note: The window shift of 50ms is used for generating the segment files with resolution of 50ms
    for the test set.
    """
    num_elements = 50
    num_complete_elements = math.floor((xs1.shape[2] - window_size) / step_size) + 1

    new_features = {}
    for key, xs in zip(['x', 'x2'], [xs1, xs2]):
        elements = []

        # Extract complete elements from the current spectral segment
        for j in range(num_complete_elements):
            start = j * step_size
            element = xs[i, :, start:start+window_size]
            elements.append(element)

        # Extract remaining elements that cross the segment boundary
        for k in range(num_elements - num_complete_elements):
            start = (num_complete_elements + k) * step_size
            init_element = xs[i, :, start:]
            remaining = window_size - init_element.shape[1]
            element = np.concatenate([init_element, xs[i+1, :, :remaining]], axis=1)
            elements.append(element)

        new_features[key] = np.array(elements, dtype=np.float32)

    return new_features

def predict(model, pred_data1, pred_data2, model_path, device, avg_pred_win=5):
    """
    Function for predicting with the trained network for the testing set.

    Args:
        model (TwoStreamCNNModel): The trained model used for prediction.
        pred_data1 (numpy.ndarray): First file containing spectrogram arrays for prediction.
        pred_data2 (numpy.ndarray or None): Second input spectrogram arrays from the same session.
                                  If None, an array of zeros will be used as the second input.
        model_path (str): Path to the saved model to be used for prediction.
        device (torch.device): The device to be used for computation.
        avg_pred_win (int): Size of the averaging window for predictions (default: 5).

    Returns:
        tuple: A tuple containing the predicted probabilities for the first and second inputs.
               - predictions (list): Predicted probabilities for the first input.
               - predictions2 (list): Predicted probabilities for the second input.
    """
    # Load the first input data
    predict_x1 = pred_data1
    predict_x2  = pred_data2 if pred_data2 is not None else np.zeros_like(predict_x1, dtype=np.float16)

    # Restore the saved model for prediction
    model.load_state_dict(torch.load(model_path, map_location=device)["model"])
    print(f'Model restored from {model_path}')

    # Perform predictions on the input data
    preds_list = []
    preds_list2 = []
    predictions = []
    predictions2 = []
    print('Predicting')
    model.train(False)

    length = min(predict_x1.shape[0], predict_x2.shape[0])
    num_batches = length
    for i in range(num_batches-1):
        # Get the input batch using the pred_input_function
        inputs = pred_input_function(predict_x1, predict_x2, i)
        # Run the model to get the predictions for the current batch
        preds, preds2 = model(torch.from_numpy(inputs['x']).to(device),
                              torch.from_numpy(inputs['x2']).to(device))
        # Append the predictions to the preds_list and preds_list2
        preds_list.extend(preds.detach().cpu().numpy())
        preds_list2.extend(preds2.detach().cpu().numpy())
        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{num_batches} batches")

    # Average predictions across consecutive windows
    for i in range(len(preds_list) - (avg_pred_win - 1)):
        # Calculate the mean predictions for the current window
        mean_preds = np.mean(preds_list[i:i + avg_pred_win], axis=0)
        mean_preds2 = np.mean(preds_list2[i:i + avg_pred_win], axis=0)
        # Append the mean predictions to the final predictions lists
        predictions.append(mean_preds)
        predictions2.append(mean_preds2)

    print(f"Predictions shape: {np.shape(predictions)}")

    return predictions, predictions2

def get_prediction(preds, cutoff):
    """
    Returns the top prediction if its confidence is higher than the cutoff, otherwise returns 'noise' (8).

    Args:
        preds (numpy.ndarray): An array of prediction probabilities.
        cutoff (float): The confidence threshold for predictions.

    Returns:
        int: The index of the top prediction if its confidence is higher than the cutoff, otherwise 8 ('noise').
    """
    try:
        # Get the index of the maximum prediction
        max_pred_idx = np.argmax(preds)
        max_pred_prob = preds[max_pred_idx]

        # Check if the maximum prediction is 'tr' or 'ph'
        if max_pred_idx in [3, 5]:
            # Count in the confidence of 'trph'
            combined_prob = max_pred_prob + preds[6]
            if combined_prob >= cutoff:
                return max_pred_idx
            else:
                return 8  # 'noise'

        # Check if the maximum prediction is 'trph'
        elif max_pred_idx == 6:
            # Consider the combined probability of 'tr', 'trph', and 'ph'
            combined_prob = max_pred_prob + preds[3] + preds[5]
            if combined_prob >= cutoff:
                return max_pred_idx
            else:
                return 8  # 'noise'

        # For other predictions
        else:
            if max_pred_prob >= cutoff:
                return max_pred_idx
            else:
                return 8  # 'noise'

    except (IndexError, ValueError):
        print("Error: Invalid input or prediction probabilities.")
        return None

def merge_segments(input_segments):
    """
    Merges adjoining segments with the same labels.

    Args:
        input_segments: A list of unmerged segments, where each segment is a tuple (start_time, end_time, label)

    Returns:
        list: A list of merged segments, where each segment is a tuple (start_time, end_time, label).
    """
    if not input_segments:
        return []
        
    segments = sorted(input_segments, key=lambda x: x[0])
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

def predict_seg(pred_prob, pred_seg, cutoff=0, merge=True):
    """
    Predict labels and generate segments from prediction probabilities.

    Args:
        pred_prob (numpy.ndarray): Numpy array containing prediction probabilities.
        pred_seg (str): Path to the output file where predicted segments will be written.
        cutoff (float): Confidence threshold for predictions.
        merge (bool): Whether to merge adjacent segments with same labels.

    Returns:
        tuple: (raw_segments, merged_segments) if merge=True, otherwise just raw_segments
    """
    try:
        predictions = pred_prob
        raw_segments = []

        # Generate raw segments
        for i, pred in enumerate(predictions):
            pred_label_idx = get_prediction(pred, cutoff)  # Get the predicted label index

            if pred_label_idx != 8:  # Check if the prediction is not 'noise'
                start_time = i * 0.05 + 0.325
                end_time = (i + 1) * 0.05 + 0.325
                label = truth_values[pred_label_idx]
                raw_segments.append((round(start_time, 6), round(end_time, 6), label))

        if not raw_segments:
            print("\nWarning: all predictions are noises\n")
            return ([], []) if merge else []

        # Merge segments if requested
        if merge:
            merged_segments = merge_segments(raw_segments)
            
            # Write merged segments to file
            with codecs.open(pred_seg, 'w', 'utf-8') as save_fil:
                for start_time, end_time, label in merged_segments:
                    save_fil.write(f"{start_time:.3f}\t{end_time:.3f}\t{label}\n")
            
            print('Done!')
            return raw_segments, merged_segments
        else:
            # Write raw segments to file
            with codecs.open(pred_seg, 'w', 'utf-8') as save_fil:
                for start_time, end_time, label in raw_segments:
                    save_fil.write(f"{start_time:.3f}\t{end_time:.3f}\t{label}\n")
            
            print('Done!')
            return raw_segments

    except (IOError, ValueError) as e:
        print(f"Error: {e}")
        return ([], []) if merge else []

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predcition from two stream CNN (https://marmosetbehavior.mit.edu/) with cutoff.")
    parser.add_argument("--eval_model", type=str, default="exp/run.torch_v2/mit_data/mit_data0/mit_cnn_72-run0/bs25lr0.0003evalinterval1avgpredwin5/train/model.ckpt", help="Model path for prediction or evaluation")
    parser.add_argument("--wav_file1", type=str, default="/data/share/bin-wu/data/marmoset/vocalization/marmoset_mit/data/pair1/pair1_animal1_together.wav", help="Path of first input wav")
    # parser.add_argument("--wav_file2", type=str, default=None, help="Path of second input wav (optional)")
    parser.add_argument("--wav_file2", type=str, default="/data/share/bin-wu/data/marmoset/vocalization/marmoset_mit/data/pair1/pair1_animal2_together.wav", help="Path of second input wav (optional); use "" when not needed")
    parser.add_argument("--out_dir", type=str, default="./exp/sandbox/out", help="Output directory to store predictions after applying the cutoff.")
    parser.add_argument("--no_merge", action="store_true", help="Do not merge adjacent segments with same labels")

    args = parser.parse_args()
    print(args)

    eval_model = args.eval_model
    wav_file1 = args.wav_file1
    wav_file2 = args.wav_file2
    out_dir = args.out_dir
    should_merge = not args.no_merge  # Merge by default unless --no_merge is specified

    set_seed(2020)
    device = set_device("auto")
    print(f"Device: {device}")

    # Create spectrograms for both streams
    spec1 = create_spec_data(wav_file1)
    spec2 = create_spec_data(wav_file2) if wav_file2 else None
    
    model = TwoStreamCNNModel().to(device)
    predictions, predictions2 = predict(model, spec1, spec2, model_path=eval_model, device=device)

    # Ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)
        
    # Generate output filenames based on input wav files
    file_name1, _ = os.path.splitext(os.path.basename(wav_file1))
    if wav_file2:
        file_name2, _ = os.path.splitext(os.path.basename(wav_file2))
    else:
        file_name2 = file_name1 + "_stream2"

    # Save predictions
    out_file1 = os.path.join(out_dir, f"pred1_{file_name1}.txt")
    out_file2 = os.path.join(out_dir, f"pred2_{file_name2}.txt")
    predict_seg(predictions, out_file1, merge=should_merge)
    predict_seg(predictions2, out_file2, merge=should_merge)

    print(f"Saved predictions to:\n{out_file1}\n{out_file2}")

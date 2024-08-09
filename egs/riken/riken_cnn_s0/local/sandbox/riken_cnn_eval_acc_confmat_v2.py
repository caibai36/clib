# Implemented by bin-wu at 11:37 on 19 April 2024
# Add window_size, window_shift, and window_middle as arguments. Normal window_size would not affect the results. The window middle and window shift setting the same value would change f-score and accuracies
# Change the window_shift and window_middle into split_size. Remove the window_size.

# v1: Add confusion matrix.
# v2: Compute the metrics versus split size; update the default paths

import os

import re
import argparse
import json

from omegaconf import OmegaConf

import numpy as np
import matplotlib.pyplot as plt

default_hypo_files = [
    "exp/sys/riken2024/division_jay_half_winmid0.05size0.5shift0.05_noisekeep5/cnn-run0/bs2048lr0.0003lrdecay1avgpredwin5/eval/test_pred_230807_001_ch1.txt"
]

parser = argparse.ArgumentParser(
    description="Compute accuracy and F-score. Generate confusion matrix and metrics-versus-split.\n"
                "Prediction/hypothesis and correct/reference files are split into 50ms segments for label comparison.\n\nNote:\n"
                "Only the labels between the indices of the first and last human-annotated labels are considered.\n"
                "We extend the prediction if it is shorter than the index of the last human-annotated label.\n"
                "We convert labels that are not in the classes into 'noise'.\n"
                "Discretizes the segments in a file into 50ms chunks using a sliding window approach.\n"
                "This implementation can reproduce the results of 'accuracy_tester.py' from https://marmosetbehavior.mit.edu/",
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--hypo_files", type=str, default=default_hypo_files, nargs="+", help='A sequence of prediction files in the Audacity label format (with "begin_sec<tab>end_sec<tab>label" on each line).')
parser.add_argument("--ref_files", type=str, default=[], nargs="+", help='A sequence of reference files in the Audacity label format with the same order as prediction files.')
parser.add_argument("--info_json", type=str, default="data/riken2024_jay_family/info.json", help="(Optional) JSON file that maps uttid to key-value pairs. The keys should include 'wav' and 'aud' for locations of audio files and Audacity labels.")
parser.add_argument("--label2id_yaml", type=str, default="conf/dict/label2id.yaml", help="The YAML file that contains the label-to-labelID mapping.")
parser.add_argument("--ref_uttids", type=str, default=["230807_001_ch1"], nargs="+", help='(Optional) A sequence of uttids of reference files in the Audacity label format with the same order as prediction files.')
parser.add_argument("--split_size", type=float, default=0.05, help="The size of a spilt in seconds when we split (discretize) hypothesis and reference segment files (using a slide window) for comparison. (default: 0.05).")
parser.add_argument("--window_size", type=float, default=0.5, help="The size of the sliding window in seconds for label assignment (default: 0.5). Window sizes that are greater than the shift size and smaller than the audio size are expected to get the same evaluation result.")

args = parser.parse_args()

hypo = args.hypo_files
ref = args.ref_files
label2id = OmegaConf.load(args.label2id_yaml)
classes=set(label2id.keys())

if not ref:
    info = OmegaConf.load(args.info_json)
    ref = [info[uttid]['seg'] for uttid in args.ref_uttids]

prediction_list = hypo
correct_list = ref

def compare_segments(prediction_list, correct_list, classes, window_size=0.5, window_shift=0.05, middle_part=0.05):
    """
    Compares human-annotated segments with model-predicted segments.

    Converts each segment into a list of 50ms labels and returns boolean lists
    indicating the correctness of predictions for noise and non-noise (signal) labels.

    Args:
        prediction_list (list): List of file paths containing predicted segments.
        correct_list (list): List of file paths containing correct human-annotated segments.
        classes (list): List of possible call types.
        window_size (float): The size of the sliding window in seconds for label assignment (default: 0.5).
        window_shift (float): The shift of the sliding window in seconds for label assignment (default: 0.05).
        middle_part (float): The proportion of the middle part of the window in seconds for label assignment (default: 0.05).

    Returns:
        tuple: A tuple containing noise_correct and signal_correct lists.
               - noise_correct: A list of booleans indicating the correctness of predictions on noise labels.
               - signal_correct: A list of booleans indicating the correctness of predictions on non-noise (signal) labels.
    Note:
    Only the labels between the indices of the first and last human-annotated labels are considered.
    We extend the prediction if is shorter than the index of the last human-annotated label
    We convert labels that not in the classes into 'noise'.
    Return two lists, each list concatenates the correct lists of all given predict-correct segment file pairs.
    The given list of reference/correct files is in the same order as that of the prediction files.
    Each given segment file is in the Audacity label format (with "begin_sec<tab>end_sec<tab>label" on each line).
    """
    noise_correct = []
    signal_correct = []

    # Iterate over the prediction and correct files simultaneously
    for pred_file, corr_file in zip(prediction_list, correct_list):
        with open(pred_file, 'r') as predictions, open(corr_file, 'r') as correct:
            # Discretize the predicted segments into 50ms chunks
            lines_pred = discretize_segments(predictions, classes, window_size, window_shift, middle_part) # Convert labels not in the classes into 'noise'
            # Discretize the correct segments into 50ms chunks and get the indices of the first and last labels
            lines_corr, first, last = discretize_segments(correct, classes, window_size, window_shift, middle_part, return_indices=True)

        # Pad the predictions with 'noise' if shorter than the correct labels
        lines_pred.extend(['noise'] * (len(lines_corr) - len(lines_pred)))

        # Compare the predicted and correct labels from the first to the last label
        for pred_label, corr_label in zip(lines_pred[first:last], lines_corr[first:last]):
            if corr_label == 'noise':
                noise_correct.append(pred_label == 'noise')
            else:
                signal_correct.append(pred_label == corr_label)

    return noise_correct, signal_correct

def discretize_segments(file, classes, window_size=0.5, window_shift=0.05, middle_part=0.05, return_indices=False):
    """
    Discretizes the segments in a file into 50ms chunks using a sliding window approach.

    A sliding window with a size of 500ms and a shift of 50ms is used to iterate over the segments.
    When the middle 50ms of the window overlaps with the interval of a label in the segment file,
    the label is assigned to the window. Otherwise, 'noise' is assigned to the window.
    The label assigned to each sliding window represents the label of a 50ms chunk.

    Args:
        file (file object): File object containing the segments.
        classes (list): List of possible call types.
        window_size (float): The size of the sliding window in seconds (default: 0.5).
        window_shift (float): The shift of the sliding window in seconds (default: 0.05).
        middle_part (float): The proportion of the middle part of the window in seconds (default: 0.05).
        return_indices (bool): Whether to return the indices of the first and last labels.

    Returns:
        list: List of labels for each 50ms chunk.
        tuple (optional): Indices of the first and last labels if return_indices is True.
    Note:
    Labels not in the given classes are converted to noise.
    """
    lines = []

    # A 500ms window and its 50ms middle part
    current = 0
    start_window = current * window_shift
    middle_start = start_window + (window_size - middle_part) / 2
    middle_end = start_window + (window_size + middle_part) / 2
    first = None # the index of the first label

    # Iterate over each line in the file
    for line in file:
        # Split the line into start time, end time, and label
        start_t, end_t, label = re.split(r'\s+', line.strip())
        start_t, end_t = float(start_t), float(end_t)
        label = label.lower()

        # Slide the window until the middle 50ms overlaps with the current segment
        while start_t - middle_end > -0.001:
            lines.append('noise')
            current += 1
            start_window = current * window_shift
            middle_start = start_window + (window_size - middle_part) / 2
            middle_end = start_window + (window_size + middle_part) / 2

        # Assign the label to the window while the middle 50ms overlaps with the current segment
        while end_t - middle_start > 0.001:
            if first is None:
                # Record the index of the first non-noise label
                first = current
            # Append the label if it is in the classes list, else append 'noise'
            lines.append('noise' if label not in classes else label)
            current += 1
            start_window = current * window_shift
            middle_start = start_window + (window_size - middle_part) / 2
            middle_end = start_window + (window_size + middle_part) / 2

    # Record the index of the last label
    last = len(lines)

    if return_indices:
        return lines, first, last
    else:
        return lines

def evaluate_metrics(signal_correct, noise_correct):
    """
    Evaluates the performance metrics based on the correctness of noise and signal predictions.

    Args:
        signal_correct (list): List of booleans indicating the correctness of signal predictions.
        noise_correct (list): List of booleans indicating the correctness of noise predictions.

    Returns:
        dict: A dictionary containing the calculated metrics (accuracy_noise, accuracy_signal, accuracy_total, precision, recall, f_score).
    """
    noise_correct = np.array(noise_correct)
    signal_correct = np.array(signal_correct)

    # Calculate the accuracy for noise labels, call labels, and all labels
    accuracy_noise = np.mean(noise_correct)
    accuracy_signal = np.mean(signal_correct)
    accuracy_total = np.mean(np.concatenate((signal_correct,noise_correct)))

    # Calculate the number of true positives, false positives, and false negatives
    true_positives = np.sum(signal_correct)
    false_positives = np.sum(~noise_correct) # ~ is 'NOT' operator, e.g., ~np.array([True, True, False]) same as  np.array([False, False,  True])
    false_negatives = np.sum(~signal_correct)

    # Calculate the precision, recall, and F-score
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    # Print the evaluation metrics in a formatted string
    print("Fraction correctly classified: Noise:{:.4f}, Call:{:.4f}, Total:{:.4f}".format(accuracy_noise, accuracy_signal, accuracy_total))
    print("Recall:{:.4f}, Precision:{:.4f}, F1-score:{:.4f}".format(recall, precision, f_score))

    # Return the evaluation metrics as a dictionary
    return {
        "accuracy_noise": accuracy_noise,
        "accuracy_signal": accuracy_signal,
        "accuracy_total": accuracy_total,
        "precision": precision,
        "recall": recall,
        "f_score": f_score
    }

def compute_confusion_matrix(prediction_list, correct_list, label2id, window_size=0.5, window_shift=0.05, middle_part=0.05):
    """
    Computes the confusion matrix for different labels, including 'noise', given lists of prediction and correct files.

    Args:
        prediction_list (list): List of file paths containing predicted segments.
        correct_list (list): List of file paths containing correct human-annotated segments.
        label2id (dict): Dictionary mapping labels to their corresponding IDs.
        window_size (float): The size of the sliding window in seconds for label assignment (default: 0.5).
        window_shift (float): The shift of the sliding window in seconds for label assignment (default: 0.05).
        middle_part (float): The proportion of the middle part of the window in seconds for label assignment (default: 0.05).

    Returns:
        tuple: A tuple containing the confusion matrix and the list of labels.
    """
    classes = set(label2id.keys())
    labels = list(label2id.keys())
    confusion_matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for pred_file, corr_file in zip(prediction_list, correct_list):
        # Discretize the predicted segments into 50ms chunks
        lines_pred = discretize_segments(open(pred_file, 'r'), classes, window_size, window_shift, middle_part)
        # Discretize the correct segments into 50ms chunks and get the indices of the first and last labels
        lines_corr, first, last = discretize_segments(open(corr_file, 'r'), classes, window_size, window_shift, middle_part, return_indices=True)

        # Pad the predictions with 'noise' if shorter than the correct labels
        lines_pred.extend(['noise'] * (len(lines_corr) - len(lines_pred)))

        # Populate the confusion matrix
        for pred_label, corr_label in zip(lines_pred[first:last], lines_corr[first:last]):
            pred_idx = labels.index(pred_label)
            corr_idx = labels.index(corr_label)
            confusion_matrix[corr_idx, pred_idx] += 1

    # Remove label entries that don't have data in either predicted or referenced segments
    valid_labels = []
    valid_indices = []
    for i, label in enumerate(labels):
        if np.sum(confusion_matrix[i, :]) > 0 or np.sum(confusion_matrix[:, i]) > 0:
            valid_labels.append(label)
            valid_indices.append(i)

    # First take the rows then take the columns of the confusion matrix
    confusion_matrix = confusion_matrix[valid_indices, :][:, valid_indices]

    return confusion_matrix, valid_labels

def plot_confusion_matrix(confusion_matrix, labels, output_file):
    """
    Plots the confusion matrix as an image with labels on the axes and saves it to a file.

    Args:
        confusion_matrix (numpy.ndarray): The confusion matrix as a 2D numpy array.
        labels (list): List of labels corresponding to the confusion matrix.
        output_file (str): Path to the output file where the image will be saved.
    """
    plt.rcParams['font.family'] = 'serif'

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(confusion_matrix, cmap='Blues')

    # Add labels to the axes
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='center', fontsize=12)
    ax.set_yticklabels(labels, fontsize=12, va='center')

    # Move x-axis labels to the top
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    # Add labels for x-axis and y-axis
    ax.set_xlabel('Predicted Labels', fontsize=14)
    ax.set_ylabel('True Labels', fontsize=14)

    # Add values to each cell
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='black', fontsize=12)

    # Add a title to the plot
    ax.set_title('Confusion Matrix', fontsize=16)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# Evaluate the accuracy and the F-score for predicted and reference segment files.
noise_correct, signal_correct = compare_segments(prediction_list, correct_list, classes, window_size=args.window_size, window_shift=args.split_size, middle_part=args.split_size)
evaluate_metrics(signal_correct, noise_correct)

# Compute the confusion matrix and save it to png and yaml files
confusion_matrix, labels = compute_confusion_matrix(prediction_list, correct_list, label2id, window_size=args.window_size, window_shift=args.split_size, middle_part=args.split_size)
output_dir = os.path.dirname(prediction_list[0])
output_file = os.path.join(output_dir, "confusion_matrix.png")
# print(f"Confusion matrix saved at: {output_file}")
plot_confusion_matrix(confusion_matrix, labels, output_file)
save_file = os.path.join(output_dir, "confusion_matrix.yaml")
x = {'labels': labels, 'matrix': confusion_matrix.tolist()}
OmegaConf.save(x, save_file)

def plot_and_save_metrics(splits, results, output_file, yaml_file):
    """
    Plot metrics versus split sizes and save both the plot and data.

    This function creates a 2x3 grid of subplots, each showing how a different
    evaluation metric changes with the split size. It also saves these metrics
    and their statistics to a YAML file.

    Args:
        splits (list): List of split sizes used for evaluation.
        results (list): List of dictionaries, each containing evaluation results for a specific split size.
        output_file (str): Path to save the output plot image.
        yaml_file (str): Path to save the output YAML file containing metric data and statistics.

    The function plots and saves data for the following metrics:
    accuracy_noise, accuracy_signal, accuracy_total, precision, recall, f_score
    """
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs = axs.flatten()
    metrics = ['accuracy_noise', 'accuracy_signal', 'accuracy_total', 'precision', 'recall', 'f_score']
    metric_titles = [m.capitalize().replace('_', ' ') for m in metrics] # make titles uppercase
    
    # Prepare data for YAML file
    yaml_data = {metric: {} for metric in metrics}
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        values = [result[metric] for result in results]
        axs[i].plot(splits, values, marker='o')
        axs[i].set_title(title)
        axs[i].set_xlabel('Split (sec)')
        axs[i].set_ylabel('Value')
        
        # Add data to yaml_data
        yaml_data[metric] = {
            'values': {str(split): float(value) for split, value in zip(splits, values)},
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    # Save data to YAML file using OmegaConf
    OmegaConf.save(yaml_data, yaml_file)

# Plot and save metrics for different split sizes
splits = [i*0.01 for i in range(1, 10)]
results = []
print()
for split in splits:
    print(f"{'='*20} SPLIT SIZE: {split:.2f} seconds {'='*20}")
    noise_correct, signal_correct = compare_segments(prediction_list, correct_list, classes, window_size=args.window_size, window_shift=split, middle_part=split)
    results.append(evaluate_metrics(signal_correct, noise_correct))

metrics_output_file = os.path.join(output_dir, "metrics_vs_split.png")
metrics_yaml_file = os.path.join(output_dir, "metrics_vs_split.yaml")
plot_and_save_metrics(splits, results, metrics_output_file, metrics_yaml_file)

print(f"Confusion matrix saved at: {output_file}")
print(f"Metrics vs split saved at: {metrics_output_file}")

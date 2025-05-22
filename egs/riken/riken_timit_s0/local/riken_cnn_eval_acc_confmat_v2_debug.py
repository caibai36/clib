# Implemented by bin-wu at 11:37 on 19 April 2024
# Update to v2 to support TIMIT no noise segments

import os

import re
import argparse
import json

from omegaconf import OmegaConf

import numpy as np
import matplotlib.pyplot as plt

default_hypo_files = [
    # "exp/sys/mit_sample/mit_sample0/mit_cnn_72-run0/bs25lr0.0003evalinterval200avgpredwin5/eval/test_pred1_Athos_cutoff0.txt",
    # "exp/sys/mit_sample/mit_sample0/mit_cnn_72-run0/bs25lr0.0003evalinterval200avgpredwin5/eval/test_pred2_Porthos_cutoff0.txt",
    "exp/sys/mit_sample/division_sample_winmid0.05size0.5shift0.05_noisekeep5/cnn-run0/bs25lr0.0003lrdecay1avgpredwin5/eval/test_pred_Athos.txt" 
]

parser = argparse.ArgumentParser(
    description="Compute accuracy and F-score. Generate confusion matrix.\n"
                "Prediction/hypothesis and correct/reference files are split into 50ms segments for label comparison.\n\nNote:\n"
                "Only the labels between the indices of the first and last human-annotated labels are considered.\n"
                "We extend the prediction if it is shorter than the index of the last human-annotated label.\n"
                "We convert labels that are not in the classes into 'noise'.\n"
                "Discretizes the segments in a file into 50ms chunks using a sliding window approach.\n"
                "This implementation can reproduce the results of 'accuracy_tester.py' from https://marmosetbehavior.mit.edu/",
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--hypo_files", type=str, default=default_hypo_files, nargs="+", help='A sequence of prediction files in the Audacity label format (with "begin_sec<tab>end_sec<tab>label" on each line).')
parser.add_argument("--ref_files", type=str, default=[], nargs="+", help='A sequence of reference files in the Audacity label format with the same order as prediction files.')
parser.add_argument("--info_json", type=str, default="data/mit_sample0/info.json", help="(Optional) JSON file that maps uttid to key-value pairs. The keys should include 'wav' and 'aud' for locations of audio files and Audacity labels.")
parser.add_argument("--label2id_yaml", type=str, default="conf/dict/label2id_marmoset.yaml", help="The YAML file that contains the label-to-labelID mapping.")
parser.add_argument("--ref_uttids", type=str, default=["Athos"], nargs="+", help='(Optional) A sequence of uttids of reference files in the Audacity label format with the same order as prediction files.')
parser.add_argument("--debug", action="store_true", help="Enable debug output for detailed processing information")

args = parser.parse_args()

hypo = args.hypo_files
ref = args.ref_files
label2id = OmegaConf.load(args.label2id_yaml)
classes=set(label2id.keys())

if args.debug:
    print(f"Debug: Loaded {len(classes)} classes from {args.label2id_yaml}")
    print(f"Debug: Classes include: {sorted(list(classes))[:10]}...")  # Show first 10 classes
    print(f"Debug: 'noise' in classes: {'noise' in classes}")
    print(f"Debug: 'sil' in classes: {'sil' in classes}")
    print("First two files")
    print(hypo[:2])
    print(ref[:2])

print(f"Number of hyp files {len(hypo)}")
print(f"Number of ref files {len(ref)}")

if not ref:
    info = OmegaConf.load(args.info_json)
    ref = [info[uttid]['seg'] for uttid in args.ref_uttids]

prediction_list = hypo
correct_list = ref

def compare_segments(prediction_list, correct_list, classes, debug=False):
    """
    Compares human-annotated segments with model-predicted segments.

    Converts each segment into a list of 50ms labels and returns boolean lists
    indicating the correctness of predictions for noise and non-noise (signal) labels.

    Args:
        prediction_list (list): List of file paths containing predicted segments.
        correct_list (list): List of file paths containing correct human-annotated segments.
        classes (list): List of possible call types.
        debug (bool): Whether to print debug information.

    Returns:
        tuple: A tuple containing noise_correct and signal_correct lists.
               - noise_correct: A list of booleans indicating the correctness of predictions on noise labels.
               - signal_correct: A list of booleans indicating the correctness of predictions on non-noise (signal) labels.
    Note:
    Only the labels between the indices of the first and last human-annoated labels are considered.
    We extend the prediction if is shorter than the index of the last human-annoated label
    We convert labels that not in the classes into 'noise'.
    Return two lists, each list concatenates the correct lists of all given predict-correct segment file pairs.
    The given list of reference/correct files is in the same order as that of the prediction files.
    Each given segment file is in the Audacity label format (with "begin_sec<tab>end_sec<tab>label" on each line).
    """
    noise_correct = []
    signal_correct = []
    
    # Debug counters
    total_files_processed = 0
    files_with_valid_segments = 0
    total_segments_compared = 0

    # Iterate over the prediction and correct files simultaneously
    for i, (pred_file, corr_file) in enumerate(zip(prediction_list, correct_list)):
        total_files_processed += 1
        
        if debug and i < 3:  # Debug first 3 files only when debug is enabled
            print(f"\nDebug: Processing file {i+1}: {os.path.basename(pred_file)}")
        
        try:
            with open(pred_file, 'r') as predictions, open(corr_file, 'r') as correct:
                # Discretize the predicted segments into 50ms chunks
                lines_pred = discretize_segments(predictions, classes, debug=(debug and i<3))
                # Discretize the correct segments into 50ms chunks and get the indices of the first and last labels
                lines_corr, first, last = discretize_segments(correct, classes, return_indices=True, debug=(debug and i<3))

            if debug and i < 3:
                print(f"  Pred segments: {len(lines_pred)}, Corr segments: {len(lines_corr)}")
                print(f"  Valid range: first={first}, last={last}")
                if len(lines_pred) > 0:
                    print(f"  Sample pred: {lines_pred[:5]}")
                if len(lines_corr) > 0:
                    print(f"  Sample corr: {lines_corr[:5]}")

            # Pad the predictions with 'noise' if shorter than the correct labels
            lines_pred.extend(['noise'] * (len(lines_corr) - len(lines_pred)))

            # Compare the predicted and correct labels from the first to the last label
            if first is not None and last is not None and last > first:
                files_with_valid_segments += 1
                segment_count = 0
                for pred_label, corr_label in zip(lines_pred[first:last], lines_corr[first:last]):
                    segment_count += 1
                    total_segments_compared += 1
                    if corr_label == 'noise':
                        noise_correct.append(pred_label == 'noise')
                    else:
                        signal_correct.append(pred_label == corr_label)
                
                if debug and i < 3:
                    print(f"  Compared {segment_count} segments")
            else:
                if debug and i < 3:
                    print(f"  Skipped - no valid segment range")
                    
        except Exception as e:
            if debug or i < 5:  # Show errors for first 5 files even without debug
                print(f"Error processing {pred_file}: {e}")
            continue

    if debug:
        print(f"\nDebug Summary:")
        print(f"  Total files processed: {total_files_processed}")
        print(f"  Files with valid segments: {files_with_valid_segments}")
        print(f"  Total segments compared: {total_segments_compared}")
        print(f"  Noise segments: {len(noise_correct)}")
        print(f"  Signal segments: {len(signal_correct)}")

    return noise_correct, signal_correct

def discretize_segments(file, classes, return_indices=False, debug=False):
    """
    Discretizes the segments in a file into 50ms chunks using a sliding window approach.

    A sliding window with a size of 500ms and a shift of 50ms is used to iterate over the segments.
    When the middle 50ms of the window overlaps with the interval of a label in the segment file,
    the label is assigned to the window. Otherwise, 'noise' is assigned to the window.
    The label assigned to each sliding window represents the label of a 50ms chunk.

    Args:
        file (file object): File object containing the segments.
        classes (list): List of possible call types.
        return_indices (bool): Whether to return the indices of the first and last labels.
        debug (bool): Whether to print debug information.

    Returns:
        list: List of labels for each 50ms chunk.
        tuple (optional): Indices of the first and last labels if return_indices is True.
    Note:
    Labels not in the given classes are converted to noise.
    """
    lines = []
    raw_labels_found = set()

    # A 500ms window and its 50ms middle part
    current = 0
    start_window = current * 0.05
    middle_start = start_window + 0.225
    middle_end = start_window + 0.275
    first = None # the index of the first label

    line_count = 0
    # Iterate over each line in the file
    for line in file:
        line = line.strip()
        if not line:
            continue
            
        line_count += 1
        try:
            # Split the line into start time, end time, and label
            parts = re.split(r'\s+', line)
            if len(parts) < 3:
                if debug:
                    print(f"    Warning: Line {line_count} has only {len(parts)} parts: {parts}")
                continue
                
            start_t, end_t = float(parts[0]), float(parts[1])
            label = parts[2].lower()
            raw_labels_found.add(label)

            if debug and line_count <= 3:
                print(f"    Line {line_count}: {start_t:.3f}-{end_t:.3f} '{label}' -> in_classes={label in classes}")

            # Slide the window until the middle 50ms overlaps with the current segment
            while start_t - middle_end > -0.001:
                lines.append('noise')
                current += 1
                start_window = current * 0.05
                middle_start = start_window + 0.225
                middle_end = start_window + 0.275

            # Assign the label to the window while the middle 50ms overlaps with the current segment
            while end_t - middle_start > 0.001:
                if first is None:
                    # Record the index of the first non-noise label
                    first = current
                # Append the label if it is in the classes list, else append 'noise'
                final_label = label if label in classes else 'noise'
                lines.append(final_label)
                current += 1
                start_window = current * 0.05
                middle_start = start_window + 0.225
                middle_end = start_window + 0.275
                
        except (ValueError, IndexError) as e:
            if debug:
                print(f"    Error parsing line {line_count}: '{line}' -> {e}")
            continue

    # Record the index of the last label
    last = len(lines)

    if debug:
        print(f"    Raw labels found: {sorted(raw_labels_found)}")
        print(f"    Labels in classes: {[l for l in raw_labels_found if l in classes]}")
        print(f"    Labels converted to noise: {[l for l in raw_labels_found if l not in classes]}")
        print(f"    Discretized to {len(lines)} chunks, valid range {first}-{last}")
        if len(lines) > 0:
            unique_labels = set(lines)
            print(f"    Unique discretized labels: {sorted(unique_labels)}")

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
    # Handle empty arrays to avoid numpy warnings
    if len(noise_correct) == 0 and len(signal_correct) == 0:
        print("Warning: No segments found for evaluation!")
        return {
            "accuracy_noise": 0.0,
            "accuracy_signal": 0.0,
            "accuracy_total": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f_score": 0.0
        }
    
    # Convert to numpy arrays
    noise_correct = np.array(noise_correct) if len(noise_correct) > 0 else np.array([])
    signal_correct = np.array(signal_correct) if len(signal_correct) > 0 else np.array([])

    # Calculate the accuracy for noise labels, call labels, and all labels
    accuracy_noise = np.mean(noise_correct) if len(noise_correct) > 0 else 0.0
    accuracy_signal = np.mean(signal_correct) if len(signal_correct) > 0 else 0.0
    
    if len(noise_correct) > 0 and len(signal_correct) > 0:
        accuracy_total = np.mean(np.concatenate((signal_correct, noise_correct)))
    elif len(noise_correct) > 0:
        accuracy_total = accuracy_noise
    elif len(signal_correct) > 0:
        accuracy_total = accuracy_signal
    else:
        accuracy_total = 0.0

    # Calculate the number of true positives, false positives, and false negatives
    true_positives = np.sum(signal_correct) if len(signal_correct) > 0 else 0
    false_positives = np.sum(~noise_correct) if len(noise_correct) > 0 else 0
    false_negatives = np.sum(~signal_correct) if len(signal_correct) > 0 else 0

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

def compute_confusion_matrix(prediction_list, correct_list, label2id):
    """
    Computes the confusion matrix for different labels, including 'noise', given lists of prediction and correct files.

    Args:
        prediction_list (list): List of file paths containing predicted segments.
        correct_list (list): List of file paths containing correct human-annotated segments.
        label2id (dict): Dictionary mapping labels to their corresponding IDs.

    Returns:
        tuple: A tuple containing the confusion matrix and the list of labels.
    """
    classes = set(label2id.keys())
    labels = list(label2id.keys())
    confusion_matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for pred_file, corr_file in zip(prediction_list, correct_list):
        # Discretize the predicted segments into 50ms chunks
        lines_pred = discretize_segments(open(pred_file, 'r'), classes)
        # Discretize the correct segments into 50ms chunks and get the indices of the first and last labels
        lines_corr, first, last = discretize_segments(open(corr_file, 'r'), classes, return_indices=True)

        # Pad the predictions with 'noise' if shorter than the correct labels
        lines_pred.extend(['noise'] * (len(lines_corr) - len(lines_pred)))

        # Populate the confusion matrix
        if first is not None and last is not None and last > first:
            for pred_label, corr_label in zip(lines_pred[first:last], lines_corr[first:last]):
                if pred_label in labels and corr_label in labels:
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
    ax.set_xlabel('Predicted labels', fontsize=14)
    ax.set_ylabel('True labels', fontsize=14)

    # Add values to each cell
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='black', fontsize=12)

    # Add a title to the plot
    ax.set_title('Confusion matrix', fontsize=16)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# Evaluate the accuracy and the F-score for predicted and reference segment files.
noise_correct, signal_correct = compare_segments(prediction_list, correct_list, classes, debug=args.debug)
evaluate_metrics(signal_correct, noise_correct)

# Compute the confusion matrix and save it to png and yaml files
confusion_matrix, labels = compute_confusion_matrix(prediction_list, correct_list, label2id)

output_dir = os.path.dirname(prediction_list[0])
output_file = os.path.join(output_dir, "confusion_matrix.png")
print(f"Confusion matrix saved at: {os.path.abspath(output_file)}")
plot_confusion_matrix(confusion_matrix, labels, output_file)

save_file = os.path.join(output_dir, "confusion_matrix.yaml")
x = {'labels': labels, 'matrix': confusion_matrix.tolist()}
OmegaConf.save(x, save_file)

# Implemented by bin-wu at 18:33 on 6 April 2024 from the MIT's implementation
import os
import re
import argparse
import json
import numpy as np

classes=['chi','cha','ek','noise','ot','ph','tr','trph','ts','tw']

default_hypo_files = [
    "exp/sys/mit_sample/mit_sample0/mit_cnn_72-run0/bs25lr0.0003evalinterval200avgpredwin5/eval/test_pred1_Athos_cutoff0.txt",
    "exp/sys/mit_sample/mit_sample0/mit_cnn_72-run0/bs25lr0.0003evalinterval200avgpredwin5/eval/test_pred2_Porthos_cutoff0.txt"
]

parser = argparse.ArgumentParser(
    description="Compute accuracy and F-score. Prediction/hypothesis and correct/reference files are split into 50ms segments for label comparison.\n\nNote:\n"
                "Only the labels between the indices of the first and last human-annotated labels are considered.\n"
                "We extend the prediction if it is shorter than the index of the last human-annotated label.\n"
                "We convert labels that are not in the classes into 'noise'.\n"
                "Discretizes the segments in a file into 50ms chunks using a sliding window approach.\n"
                "This implementation can reproduce the results of 'accuracy_tester.py' from https://marmosetbehavior.mit.edu/",
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--hypo_files", type=str, default=default_hypo_files, nargs="+", help='A sequence of prediction files in the Audacity label format (with "begin_sec<tab>end_sec<tab>label" on each line).')
parser.add_argument("--ref_files", type=str, default=[], nargs="+", help='A sequence of reference files in the Audacity label format with the same order as prediction files.')
parser.add_argument("--info_json", type=str, default="data/mit_sample/info.json", help="(Optional) JSON file that maps uttid to key-value pairs. The keys should include 'wav' and 'aud' for locations of audio files and Audacity labels.")
parser.add_argument("--ref_uttids", type=str, default=["Athos", "Porthos"], nargs="+", help='(Optional) A sequence of uttids of reference files in the Audacity label format with the same order as prediction files.')

args = parser.parse_args()

hypo = args.hypo_files
ref = args.ref_files

if not ref:
    with open(args.info_json, 'r') as f:
        info = json.load(f)
    ref = [info[uttid]['aud'] for uttid in args.ref_uttids]

prediction_list = hypo
correct_list = ref

def compare_segments(prediction_list, correct_list, classes):
    """
    Compares human-annotated segments with model-predicted segments.

    Converts each segment into a list of 50ms labels and returns boolean lists
    indicating the correctness of predictions for noise and non-noise (signal) labels.

    Args:
        prediction_list (list): List of file paths containing predicted segments.
        correct_list (list): List of file paths containing correct human-annotated segments.
        classes (list): List of possible call types.

    Returns:
        tuple: A tuple containing noise_correct, signal_correct, and non_noise_correct lists.
    """
    noise_correct = []
    signal_correct = []
    non_noise_correct = []

    for pred_file, corr_file in zip(prediction_list, correct_list):
        with open(pred_file, 'r') as predictions, open(corr_file, 'r') as correct:
            lines_pred = discretize_segments(predictions, classes)
            lines_corr, first, last = discretize_segments(correct, classes, return_indices=True)

        lines_pred.extend(['noise'] * (len(lines_corr) - len(lines_pred)))

        for pred_label, corr_label in zip(lines_pred[first:last], lines_corr[first:last]):
            if corr_label == 'noise':
                noise_correct.append(pred_label == 'noise')
            else:
                signal_correct.append(pred_label == corr_label)
                non_noise_correct.append(pred_label != 'noise')

    return noise_correct, signal_correct, non_noise_correct

def discretize_segments(file, classes, return_indices=False):
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

    Returns:
        list: List of labels for each 50ms chunk.
        tuple (optional): Indices of the first and last labels if return_indices is True.
    """
    lines = []

    current = 0
    start_window = current * 0.05
    middle_start = start_window + 0.225
    middle_end = start_window + 0.275
    first = None

    for line in file:
        start_t, end_t, label = re.split(r'\s+', line.strip())
        start_t, end_t = float(start_t), float(end_t)
        label = label.lower()

        while start_t - middle_end > -0.001:
            lines.append('noise')
            current += 1
            start_window = current * 0.05
            middle_start = start_window + 0.225
            middle_end = start_window + 0.275

        while end_t - middle_start > 0.001:
            if first is None:
                first = current
            lines.append('noise' if label not in classes else label)
            current += 1
            start_window = current * 0.05
            middle_start = start_window + 0.225
            middle_end = start_window + 0.275

    last = len(lines)

    if return_indices:
        return lines, first, last
    else:
        return lines

def evaluate_metrics(signal_correct, noise_correct, non_noise_correct):
    """
    Evaluates the performance metrics based on the correctness of predictions.

    Args:
        signal_correct (list): List of booleans indicating the correctness of signal predictions.
        noise_correct (list): List of booleans indicating the correctness of noise predictions.
        non_noise_correct (list): List of booleans indicating the correctness of non-noise predictions.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    noise_correct = np.array(noise_correct)
    signal_correct = np.array(signal_correct)
    non_noise_correct = np.array(non_noise_correct)

    accuracy_noise = np.mean(noise_correct)
    accuracy_signal = np.mean(signal_correct)
    accuracy_total = np.mean(np.concatenate((signal_correct, noise_correct)))
    accuracy_non_noise = np.mean(non_noise_correct)

    true_positives = np.sum(signal_correct)
    false_positives = np.sum(~noise_correct)
    false_negatives = np.sum(~signal_correct)

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    print("Fraction correctly classified: Noise:{:.4f}, Call:{:.4f}, Total:{:.4f}".format(
        accuracy_noise, accuracy_signal, accuracy_total))
    print("Caller accuracy:{:.4f}".format(accuracy_non_noise))
    print("Recall:{:.4f}, Precision:{:.4f}, F1-score:{:.4f}".format(
        recall, precision, f_score))

    return {
        "accuracy_noise": accuracy_noise,
        "accuracy_signal": accuracy_signal,
        "accuracy_total": accuracy_total,
        "accuracy_non_noise": accuracy_non_noise,
        "precision": precision,
        "recall": recall,
        "f_score": f_score
    }

noise_correct, signal_correct, non_noise_correct = compare_segments(prediction_list, correct_list, classes)
evaluate_metrics(signal_correct, noise_correct, non_noise_correct)

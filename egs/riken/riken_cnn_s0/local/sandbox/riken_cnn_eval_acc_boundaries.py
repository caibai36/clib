# Implemented by bin-wu at 16:57 on 20 April 2024

import os
import re
import argparse
import json
from omegaconf import OmegaConf
import numpy as np

def calculate_boundary_accuracy(ref_segments, pred_segments, tolerance=0.1):
    """
    Calculate the boundary accuracy between predicted and reference segments.

    This function measures how well the predicted segment boundaries (start and end times)
    align with the reference segment boundaries. It calculates precision, recall, and
    F1-score based on a defined tolerance level.

    Args:
        ref_segments (list of tuples): Reference segments in the format [(start, end, label), ...].
        pred_segments (list of tuples): Predicted segments in the format [(start, end, label), ...].
        tolerance (float): Maximum allowed deviation (in seconds) for a boundary to be considered correct.

    Returns:
        correct_count (int): Number of correctly predicted boundaries within tolerance
        pred_count (int): Total number of predicted boundaries
        ref_count (int): Total number of reference boundaries
    """
    correct_boundaries = 0
    # Check each predicted segment against each reference segment
    for pred_start, pred_end, _ in pred_segments:
        for ref_start, ref_end, _ in ref_segments:
            # Check if both start and end times fall within the allowed tolerance
            if abs(pred_start - ref_start) <= tolerance and abs(pred_end - ref_end) <= tolerance:
                correct_boundaries += 1
                break  # Avoid counting any more once a match is found

    return correct_boundaries, len(pred_segments), len(ref_segments)

default_hypo_files = ["exp/sys/mit_sample/division_sample_winmid0.05size0.5shift0.05_noisekeep5/cnn-run0/bs25lr0.0003lrdecay1avgpredwin5/eval/test_pred_Athos.txt"]

parser = argparse.ArgumentParser(
    description="Compute boundary F-score. Prediction/hypothesis and correct/reference files are split into 50ms segments for label comparison.",
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--hypo_files", type=str, default=default_hypo_files, nargs="+",
    help='A sequence of prediction files in the Audacity label format (with "begin_sec<tab>end_sec<tab>label" on each line).')
parser.add_argument("--ref_files", type=str, default=[], nargs="+",
    help='A sequence of reference files in the Audacity label format with the same order as prediction files.')
parser.add_argument("--info_json", type=str, default="data/mit_sample0/info.json",
    help="JSON file that maps uttid to key-value pairs. The keys should include 'wav' and 'seg' for locations of audio files and Audacity labels.")
parser.add_argument("--ref_uttid", type=str, default="Athos",
    help='Utterance ID of the reference file.')
parser.add_argument("--tolerance", type=float, default=0.1,
    help="Maximum allowed deviation (in seconds) for a boundary to be considered correct.")

args = parser.parse_args()

# If ref_files not provided, use info_json to get reference files
if not args.ref_files:
    info = OmegaConf.load(args.info_json)
    args.ref_files = [info[args.ref_uttid]['seg']] * len(args.hypo_files)

if len(args.hypo_files) != len(args.ref_files):
    raise ValueError("Number of hypothesis files must match number of reference files")

total_correct = 0
total_pred = 0
total_ref = 0

# Process each pair of hypothesis and reference files
for hypo_file, ref_file in zip(args.hypo_files, args.ref_files):
    # Read prediction and reference files
    with open(hypo_file, 'r') as f:
        pred_segments = [tuple(line.strip().split('\t')) for line in f]
        pred_segments = [(float(start), float(end), label) for start, end, label in pred_segments]

    with open(ref_file, 'r') as f:
        ref_segments = [tuple(line.strip().split('\t')) for line in f]
        ref_segments = [(float(start), float(end), label) for start, end, label in ref_segments]

    # Calculate boundary accuracy for this file pair
    correct, pred, ref = calculate_boundary_accuracy(ref_segments, pred_segments, tolerance=args.tolerance)

    # Accumulate totals
    total_correct += correct
    total_pred += pred
    total_ref += ref

# Calculate final metrics
boundary_precision = total_correct / total_pred if total_pred > 0 else 0
boundary_recall = total_correct / total_ref if total_ref > 0 else 0
boundary_f1 = 2 * (boundary_precision * boundary_recall) / (boundary_precision + boundary_recall) if (boundary_precision + boundary_recall) > 0 else 0

# Print results
print("Boundary (tolerance {}ms) Recall:{:.4f}, Precision:{:.4f}, F1-score:{:.4f}".format(
    args.tolerance*1000, boundary_recall, boundary_precision, boundary_f1))

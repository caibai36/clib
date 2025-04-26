# Implemented by bin-wu at 16:57 on 20 April 2024

import os
import re
import argparse
import json
from omegaconf import OmegaConf
import numpy as np

# # Wrong version double counts the references
# def calculate_boundary_accuracy(ref_segments, pred_segments, tolerance=0.1):
#     """
#     Calculate the boundary accuracy between predicted and reference segments.

#     This function measures how well the predicted segment boundaries (start and end times)
#     align with the reference segment boundaries. It calculates precision, recall, and
#     F1-score based on a defined tolerance level.

#     Args:
#         ref_segments (list of tuples): Reference segments in the format [(start, end, label), ...].
#         pred_segments (list of tuples): Predicted segments in the format [(start, end, label), ...].
#         tolerance (float): Maximum allowed deviation (in seconds) for a boundary to be considered correct.

#     Returns:
#         precision (float): The proportion of predicted boundaries that are correct within the tolerance.
#         recall (float): The proportion of reference boundaries that are correctly predicted within the tolerance.
#         f1_score (float): The harmonic mean of precision and recall.
#     """
#     correct_boundaries = 0
#     # Check each predicted segment against each reference segment
#     for pred_start, pred_end, _ in pred_segments:
#         for ref_start, ref_end, _ in ref_segments:
#             # Check if both start and end times fall within the allowed tolerance
#             if abs(pred_start - ref_start) <= tolerance and abs(pred_end - ref_end) <= tolerance:
#                 correct_boundaries += 1
#                 break  # Avoid counting any more once a match is found
    
#     # Calculate precision and recall based on the matches
#     precision = correct_boundaries / len(pred_segments)
#     recall = correct_boundaries / len(ref_segments)
#     # Calculate F1-score to balance precision and recall
#     f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
#     return precision, recall, f1_score

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
        precision (float): The proportion of predicted segments that are correct within the tolerance.
        recall (float): The proportion of reference segments that are correctly matched within the tolerance.
        f1_score (float): The harmonic mean of precision and recall.
    """
    correct_segments = 0
    matched_refs = set()  # Track which reference segments have been matched
    
    # Check each predicted segment against each reference segment
    for i, (pred_start, pred_end, _) in enumerate(pred_segments):
        for j, (ref_start, ref_end, _) in enumerate(ref_segments):
            # Skip already matched reference segments
            if j in matched_refs:
                continue
                
            # Check if both start and end times fall within the allowed tolerance
            if abs(pred_start - ref_start) <= tolerance and abs(pred_end - ref_end) <= tolerance:
                correct_segments += 1
                matched_refs.add(j)  # Mark this reference segment as matched
                break  # Move to the next predicted segment
    
    # Calculate precision and recall
    precision = correct_segments / len(pred_segments) if pred_segments else 0
    recall = correct_segments / len(ref_segments) if ref_segments else 0
    
    # Calculate F1-score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return precision, recall, f1_score

default_hypo_file = "exp/sys/mit_sample/division_sample_winmid0.05size0.5shift0.05_noisekeep5/cnn-run0/bs25lr0.0003lrdecay1avgpredwin5/eval/test_pred_Athos.txt"

parser = argparse.ArgumentParser(
    description="Compute boundary F-score. Prediction/hypothesis and correct/reference files are split into 50ms segments for label comparison.",
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--hypo_file", type=str, default=default_hypo_file, help='Prediction file in the Audacity label format (with "begin_sec<tab>end_sec<tab>label" on each line).')
parser.add_argument("--ref_file", type=str, default=None, help='Reference file in the Audacity label format.')
parser.add_argument("--info_json", type=str, default="data/mit_sample0/info.json", help="JSON file that maps uttid to key-value pairs. The keys should include 'wav' and 'seg' for locations of audio files and Audacity labels.")
parser.add_argument("--ref_uttid", type=str, default="Athos", help='Utterance ID of the reference file.')
parser.add_argument("--tolerance", type=float, default=0.1, help="Maximum allowed deviation (in seconds) for a boundary to be considered correct.")

args = parser.parse_args()

ref_file = args.ref_file

if not ref_file:
    info = OmegaConf.load(args.info_json)
    ref_file = info[args.ref_uttid]['seg']

hypo_file = args.hypo_file

# Read prediction and reference files
with open(hypo_file, 'r') as f:
    pred_segments = [tuple(line.strip().split('\t')) for line in f]
    pred_segments = [(float(start), float(end), label) for start, end, label in pred_segments]

with open(ref_file, 'r') as f:
    ref_segments = [tuple(line.strip().split('\t')) for line in f]
    ref_segments = [(float(start), float(end), label) for start, end, label in ref_segments]

# Calculate boundary accuracy
boundary_precision, boundary_recall, boundary_f1 = calculate_boundary_accuracy(ref_segments, pred_segments, tolerance=args.tolerance)

# Print results
print("Boundary (tolerance {}ms) Recall:{:.4f}, Precision:{:.4f}, F1-score:{:.4f}".format(args.tolerance*1000,boundary_recall, boundary_precision, boundary_f1))

# Implemented by bin-wu at 23:23 on 18 April 2024

import os
import argparse
import codecs
from omegaconf import OmegaConf
import numpy as np

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

def predict(pred_prob, label2id):
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

default_pred_files = [
    # "exp/sys/mit_sample/mit_sample0/mit_cnn_72-run0/bs25lr0.0003evalinterval200avgpredwin5/eval/test_pred1_Athos.npy",
    # "exp/sys/mit_sample/mit_sample0/mit_cnn_72-run0/bs25lr0.0003evalinterval200avgpredwin5/eval/test_pred2_Porthos.npy",
    "exp/sys/mit_sample/division_sample_winmid0.05size0.5shift0.05_noisekeep5/cnn-run0/bs25lr0.0003lrdecay1avgpredwin5/eval/test_pred_Athos.npy"
]

parser = argparse.ArgumentParser(description="Cutoff the predictions (reference: 'cutoff_predictor_single.py' from https://marmosetbehavior.mit.edu/).")
parser.add_argument("--label2id_yaml", type=str, default="conf/dict/label2id_marmoset.yaml", help="The YAML file that contains the label-to-labelID mapping.")
parser.add_argument("--pred_files", type=str, default=default_pred_files, nargs="+", help="Sequence of prediction files that need to be cut off.")
parser.add_argument("--out_dir", type=str, default="exp/sys/mit_sample/division_sample_winmid0.05size0.5shift0.05_noisekeep5/cnn-run0/bs25lr0.0003lrdecay1avgpredwin5/eval", help="Output directory to store predictions after applying the cutoff.")
parser.add_argument("--save_raw_segments", action="store_true", help="Save the raw segment file.")

args = parser.parse_args()
print(args)

label2id = OmegaConf.load(args.label2id_yaml)

for pred_file in args.pred_files:
    file_name, _ = os.path.splitext(os.path.basename(pred_file))
    raw_seg_file = os.path.join(args.out_dir, file_name+"_raw.txt")
    merged_seg_file = os.path.join(args.out_dir, file_name+".txt")

    print(f"Pred file: {os.path.abspath(pred_file)}")
    print(f"Merged seg file: {os.path.abspath(merged_seg_file)}")
    if args.save_raw_segments:
        print(f"Raw seg file: {os.path.abspath(raw_seg_file)}")
    
    # Open the prediction probabilities file in binary mode
    pred_prob = np.load(pred_file)  # Load the prediction probabilities

    raw_segments, merged_segments = predict(pred_prob, label2id)

    # Open the output file in write mode with UTF-8 encoding for Audacity compatibility
    if args.save_raw_segments:
        with codecs.open(raw_seg_file, 'w', 'utf-8') as f_raw:
            for start_time, end_time, label in raw_segments:
                f_raw.write(f"{start_time}\t{end_time}\t{label}\n")

    with codecs.open(merged_seg_file, 'w', 'utf-8') as f_merged:
        for start_time, end_time, label in merged_segments:
            f_merged.write(f"{start_time}\t{end_time}\t{label}\n")

# Implemented by bin-wu at 20:42 on 5 April 2024 from the MIT's implementation

import os
import argparse
import codecs

import numpy as np

truth_values=['cha','chi','ek','ph','ts','tr','trph','tw','noise']

def get_prediction(preds, cutoff):
    """
    Returns the top prediction if its confidence is higher than the cutoff, otherwise returns 'noise' (8).

    Args:
        preds (numpy.ndarray): An array of prediction probabilities.
        cutoff (float, optional): The confidence threshold for predictions. Defaults to 0.5.

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

def predict(pred_prob, pred_seg, cutoff):
    """
    Predict labels from the given prediction probabilities and write the start time, end time, and label to a segment file.

    Args:
        pred_prob (str): Path to the file containing prediction probabilities.
        pred_seg (str): Path to the output file where predicted segments will be written.
        cutoff (float): Confidence threshold for predictions.
    """
    try:
        # Open the prediction probabilities file in binary mode
        with open(pred_prob, 'rb') as fil1:
            predictions = np.load(fil1)  # Load the prediction probabilities

        # Open the output file in write mode with UTF-8 encoding for Audacity compatibility
        with codecs.open(pred_seg, 'w', 'utf-8') as save_fil:
            for i, pred in enumerate(predictions):
                pred_label_idx = get_prediction(pred, cutoff)  # Get the predicted label index

                if pred_label_idx != 8:  # Check if the prediction is not 'noise'
                    start_time = i * 0.05 + 0.325 # Keep same as 'cutoff_predictor_single.py' from https://marmosetbehavior.mit.edu/; 0.325 should be 0.225 (start of mid-50ms of 500ms)
                    end_time = (i + 1) * 0.05 + 0.325
                    label = truth_values[pred_label_idx]
                    # Write the start time, end time, and label to the output file
                    save_fil.write(f"{start_time:.3f}\t{end_time:.3f}\t{label}\n")

        print('Done!')

    except (IOError, ValueError) as e:
        print(f"Error: {e}")

default_pred_files = [
    "exp/sys/mit_sample/mit_sample0/mit_cnn_72-run0/bs25lr0.0003evalinterval200avgpredwin5/eval/test_pred1_Athos.npy",
    "exp/sys/mit_sample/mit_sample0/mit_cnn_72-run0/bs25lr0.0003evalinterval200avgpredwin5/eval/test_pred2_Porthos.npy"
]

parser = argparse.ArgumentParser(description="Cutoff the predictions (reference: 'cutoff_predictor_single.py' from https://marmosetbehavior.mit.edu/).")
parser.add_argument("--cutoffs", type=float, default=[0, 0.7, 0.8], nargs="+", help="Values of cutoff (e.g., --cutoffs 0.7 0.8).")
parser.add_argument("--pred_files", type=str, default=default_pred_files, nargs="+", help="Sequence of prediction files that need to be cut off.")
parser.add_argument("--out_dir", type=str, default="exp/sys/mit_sample/mit_sample0/mit_cnn_72-run0/bs25lr0.0003evalinterval200avgpredwin5/eval", help="Output directory to store predictions after applying the cutoff.")

args = parser.parse_args()

for cutoff in args.cutoffs:
    for pred_file in args.pred_files:
        file_name, _ = os.path.splitext(os.path.basename(pred_file))
        if int(cutoff) == cutoff:
            cutoff_file_name = f"{file_name}_cutoff{int(cutoff)}.txt" # Avoid float converting cutoff 0 into cutoff 0.0
        else:
            cutoff_file_name = f"{file_name}_cutoff{cutoff}.txt"
        cutoff_file = os.path.join(args.out_dir, cutoff_file_name)

        print(f"Pred file: {pred_file}")
        print(f"Cutoff file: {cutoff_file}")
        predict(pred_file, cutoff_file, cutoff=cutoff)

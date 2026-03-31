#!/usr/bin/env python3
"""
Filter spectrogram features to extract first 15 weeks of data using a mask file.
"""

import argparse
import numpy as np
import os


def load_mask(mask_file):
    """Load mask file and return boolean array."""
    mask = []
    with open(mask_file, 'r') as f:
        for line in f:
            mask.append(int(line.strip()) == 1)
    return np.array(mask)


def filter_spectrogram(input_npy, mask_file, output_npy):
    """
    Filter spectrogram features using mask.

    Args:
        input_npy: Path to input .npy file with shape (num_records, feat_x, feat_y)
        mask_file: Path to mask text file with 1/0 for each record
        output_npy: Path to output filtered .npy file
    """
    # Load input data
    print(f"Loading input data from: {input_npy}")
    data = np.load(input_npy)
    print(f"Input shape: {data.shape}")

    # Load mask
    print(f"Loading mask from: {mask_file}")
    mask = load_mask(mask_file)
    print(f"Mask shape: {mask.shape}")
    print(f"Number of True values in mask: {np.sum(mask)}")

    # Verify mask length matches data
    if len(mask) != data.shape[0]:
        raise ValueError(
            f"Mask length ({len(mask)}) does not match "
            f"number of records in data ({data.shape[0]})"
        )

    # Apply mask
    print("Applying mask...")
    filtered_data = data[mask]
    print(f"Output shape: {filtered_data.shape}")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_npy)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    # Save filtered data
    print(f"Saving filtered data to: {output_npy}")
    np.save(output_npy, filtered_data)
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description='Filter spectrogram features to extract first 15 weeks using mask file',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Default paths for b2_f1
    default_input = '/data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data/division_nas5_b2_f1_mae_vit_labels_for_mae_pretrained_all_days/train_input.npy'
    default_mask = 'exp/mae_feat_dim_reduction/python_script_nas5/b2_f1_15weeks/mask_b2_f1_15weeks.txt'
    default_output = '/data04/share/bin-wu/feat/riken/riken_cnn_s0/exp/data_15w/b2_f1_15weeks/spec_raw.npy'

    parser.add_argument(
        '--input',
        type=str,
        default=default_input,
        help='Path to input .npy file (train_input.npy)'
    )

    parser.add_argument(
        '--mask',
        type=str,
        default=default_mask,
        help='Path to mask text file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=default_output,
        help='Path to output .npy file (spec_raw.npy)'
    )

    args = parser.parse_args()

    # Verify input files exist
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not os.path.exists(args.mask):
        raise FileNotFoundError(f"Mask file not found: {args.mask}")

    # Process
    filter_spectrogram(args.input, args.mask, args.output)


if __name__ == '__main__':
    main()

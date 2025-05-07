#!/usr/bin/env python3
import os
import json
import yaml
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare NTT Infant data for Kaldi")
    parser.add_argument("--data_info_json", type=str, default="data/ntt_infant/info.json",
                        help="Path to the data info JSON file")
    parser.add_argument("--division_yaml", type=str, default="conf/data/division_ntt.yaml",
                        help="Path to dataset division YAML file")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory for Kaldi data")
    return parser.parse_args()

def prepare_kaldi_data(data_info, ids, output_dir):
    """Prepare Kaldi data files"""
    os.makedirs(output_dir, exist_ok=True)

    # Create wav.scp, text, segments, utt2spk files
    with open(os.path.join(output_dir, "wav.scp"), "w") as wav_scp, \
         open(os.path.join(output_dir, "text"), "w") as text_file, \
         open(os.path.join(output_dir, "segments"), "w") as segments_file, \
         open(os.path.join(output_dir, "utt2spk"), "w") as utt2spk_file:

        for id in sorted(ids):
            # Get paths from data_info
            wav_path = data_info[id]["wav"]
            token_path = data_info[id]["token"]

            # Get speaker ID (first part of ID, e.g., "kk" from "kk001_1")
            speaker_id = id.split('_')[0]

            # Write wav.scp entry
            wav_scp.write(f"{id} {wav_path}\n")

            # Process token file to get segments and transcriptions
            try:
                with open(token_path, "r") as f:
                    token_lines = f.readlines()

                # Process each line in the token file
                for seg_idx, line in enumerate(token_lines):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        start_time, end_time = float(parts[0]), float(parts[1])
                        tokens = " ".join(parts[2:])

                        # Create unique utterance ID for this segment
                        utt_id = f"{id}_{seg_idx+1:04d}"

                        # Write to segments file
                        segments_file.write(f"{utt_id} {id} {start_time:.3f} {end_time:.3f}\n")

                        # Write to text file
                        text_file.write(f"{utt_id} {tokens}\n")

                        # Write to utt2spk file
                        utt2spk_file.write(f"{utt_id} {speaker_id}\n")

            except Exception as e:
                print(f"Error processing token file {token_path}: {e}")

    # Generate spk2utt file
    os.system(f"utils/utt2spk_to_spk2utt.pl {os.path.join(output_dir, 'utt2spk')} > {os.path.join(output_dir, 'spk2utt')}")

    return output_dir

def main():
    args = parse_args()

    # Load data info
    print(f"Loading data info from {args.data_info_json}")
    with open(args.data_info_json, "r") as f:
        data_info = json.load(f)

    # Load division yaml
    print(f"Loading division info from {args.division_yaml}")
    with open(args.division_yaml, "r") as f:
        division = yaml.safe_load(f)

    # Prepare each dataset
    for subset in ["train", "dev", "test"]:
        output_path = os.path.join(args.output_dir, subset)
        prepare_kaldi_data(data_info, division[subset], output_path)
        print(f"Prepared {subset} data: {len(division[subset])} recordings")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Combine Kaldi datasets")
    parser.add_argument("--input_dirs", type=str, nargs='+', required=True,
                        help="Input data directories to combine")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for combined data")
    return parser.parse_args()

def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Use Kaldi's utils/combine_data.sh to merge the datasets
    cmd = ["utils/combine_data.sh", args.output_dir] + args.input_dirs
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    print(f"Combined data written to {args.output_dir}")

if __name__ == "__main__":
    main()

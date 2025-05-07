#!/usr/bin/env python3
import os
import yaml
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dictionary for NTT Infant data")
    parser.add_argument("--dict_yaml", type=str, default="conf/dict/ntt_label2id.yaml",
                        help="Path to the NTT label2id YAML file")
    parser.add_argument("--output_dir", type=str, default="data/local/dict",
                        help="Output directory for Kaldi dictionary")
    return parser.parse_args()

def prepare_dictionary(dict_yaml, output_dir):
    """Prepare Kaldi dictionary files"""
    os.makedirs(output_dir, exist_ok=True)

    # Load dictionary from YAML
    with open(dict_yaml, "r") as f:
        label_dict = yaml.safe_load(f)

    # Extract all regular tokens (excluding special tokens)
    regular_tokens = []
    special_tokens = []

    for token in label_dict.keys():
        if token.startswith("<") and token.endswith(">"):
            special_tokens.append(token)
        elif not token.startswith("<") and len(token) > 0:
            regular_tokens.append(token)

    # Write lexicon.txt
    with open(os.path.join(output_dir, "lexicon.txt"), "w") as f:
        # Add silence and special tokens
        f.write("<UNK> SPN\n")  # Unknown word
        f.write("<SIL> SIL\n")  # Silence

        # Add special tokens mapping to themselves
        for token in special_tokens:
            if token not in ["<unk>", "<pad>", "<sos>", "<eos>", "<period>", "<space>"]:
                token_without_brackets = token[1:-1]  # Remove < and >
                f.write(f"{token} {token_without_brackets}\n")

        # Add regular tokens mapping to themselves (Japanese kana)
        for token in regular_tokens:
            f.write(f"{token} {token}\n")

    # Write silence_phones.txt
    with open(os.path.join(output_dir, "silence_phones.txt"), "w") as f:
        f.write("SIL\n")
        f.write("SPN\n")

    # Write nonsilence_phones.txt
    with open(os.path.join(output_dir, "nonsilence_phones.txt"), "w") as f:
        # Add special tokens
        for token in special_tokens:
            if token not in ["<unk>", "<pad>", "<sos>", "<eos>", "<period>", "<space>"]:
                token_without_brackets = token[1:-1]  # Remove < and >
                f.write(f"{token_without_brackets}\n")

        # Add regular tokens
        for token in regular_tokens:
            f.write(f"{token}\n")

    # Write optional_silence.txt
    with open(os.path.join(output_dir, "optional_silence.txt"), "w") as f:
        f.write("SIL\n")

    # Write extra_questions.txt (empty for now)
    with open(os.path.join(output_dir, "extra_questions.txt"), "w") as f:
        pass

    return output_dir

def main():
    args = parse_args()
    prepare_dictionary(args.dict_yaml, args.output_dir)
    print(f"Dictionary prepared in {args.output_dir}")

if __name__ == "__main__":
    main()

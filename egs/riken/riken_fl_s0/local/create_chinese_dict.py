#!/usr/bin/env python3
"""
Generate a label-to-id mapping dictionary for Chinese phonemes and their combinations.

This script reads a pinyin mapping file and creates a comprehensive dictionary that includes:
- Special tokens (<unk>, <pad>, <sos>, etc.)
- Initial consonants
- Finals with and without tones
- Full syllables (initial_final) with and without tones
- Extended phoneme combinations with multiple X tokens
- Additional mask tokens
"""

import argparse
from typing import List, Set, Dict
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate label-to-id mapping dictionary")
    parser.add_argument("--input", type=str, default="conf/data/PY2IF+Tone.txt",
                       help="Input pinyin mapping file path")
    parser.add_argument("--output", type=str, default="conf/dict/chinese_all_label2id.txt",
                       help="Output dictionary file path")
    parser.add_argument("--x_tokens", nargs='+', default=["<mask>"],
                       help="List of tokens to use as phoneme extension placeholders")
    parser.add_argument("--num_extra_masks", type=int, default=9,
                       help="Number of additional mask tokens to add")
    return parser.parse_args()

def read_pinyin_file(file_path: str) -> tuple[Set[str], Set[str], Set[str]]:
    """
    Read the pinyin mapping file and extract initials, finals, and tones.
    
    Args:
        file_path: Path to the pinyin mapping file
        
    Returns:
        Tuple of sets containing initials, finals (without tones), and tones
    """
    initials = set()
    finals = set()
    tones = set()
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                initials.add(parts[1])
                finals.add(parts[2])
                tones.add(parts[3])
    
    return initials, finals, tones

def generate_dictionary(initials: Set[str], finals: Set[str], tones: Set[str], x_tokens: List[str]) -> Dict[str, int]:
    """
    Generate the complete label-to-id mapping dictionary.
    
    Args:
        initials: Set of initial consonants
        finals: Set of finals without tones
        tones: Set of tone numbers
        x_tokens: List of tokens to use as phoneme extension placeholders
        
    Returns:
        Dictionary mapping labels to IDs
    """
    label2id = {}
    current_id = 0
    
    # Add special tokens
    special_tokens = ['<unk>', '<pad>', '<sos>', '<eos>', '<period>', '<space>', '<mask>']
    for token in special_tokens:
        label2id[token] = current_id
        current_id += 1
    
    # Add initials
    for initial in sorted(initials):
        label2id[initial] = current_id
        current_id += 1
    
    # Add finals without and with tones
    for final in sorted(finals):
        label2id[final] = current_id
        current_id += 1
        for tone in sorted(tones):
            label2id[f"{final}{tone}"] = current_id
            current_id += 1
    
    # Add full syllables without and with tones
    for initial in sorted(initials):
        for final in sorted(finals):
            label2id[f"{initial}_{final}"] = current_id
            current_id += 1
            for tone in sorted(tones):
                label2id[f"{initial}_{final}{tone}"] = current_id
                current_id += 1
    
    # Add combinations for each X token
    for x_token in x_tokens:
        # X as initial
        for final in sorted(finals):
            label2id[f"{x_token}_{final}"] = current_id
            current_id += 1
            for tone in sorted(tones):
                label2id[f"{x_token}_{final}{tone}"] = current_id
                current_id += 1
        
        # X as final
        for initial in sorted(initials):
            label2id[f"{initial}_{x_token}"] = current_id
            current_id += 1
            for tone in sorted(tones):
                label2id[f"{initial}_{x_token}{tone}"] = current_id
                current_id += 1
        
        # X alone
        label2id[x_token] = current_id
        current_id += 1
        
        # X_X combinations
        for x_token2 in x_tokens:
            label2id[f"{x_token}_{x_token2}"] = current_id
            current_id += 1
    
    return label2id

def main():
    args = parse_args()
    
    # Read input file
    initials, finals, tones = read_pinyin_file(args.input)
    
    # Generate dictionary
    label2id = generate_dictionary(initials, finals, tones, args.x_tokens)
    
    # Add additional mask tokens if they're not already included
    current_id = max(label2id.values()) + 1
    for i in range(1, args.num_extra_masks + 1):
        mask_token = f"<mask{i}>"
        if mask_token not in label2id:
            label2id[mask_token] = current_id
            current_id += 1
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write dictionary to file
    with open(output_path, 'w') as f:
        for label, idx in sorted(label2id.items(), key=lambda x: x[1]):
            f.write(f"{label}: {idx}\n")

if __name__ == "__main__":
    main()

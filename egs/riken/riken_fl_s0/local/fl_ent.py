"""
Functional Load Calculator for Phoneme Pairs

This script calculates the functional load between two phonemes based on n-gram entropy.
Functional load represents the information loss that would occur if two phonemes were merged.
Higher functional load indicates the contrast between phonemes is more important in the language.

For syllable-level calculation:
- Input format: initial1 final1 initial2 final2 ...
- Original entropy: Computed after joining adjacent initial-final pairs into syllables
- Merged entropy: First merges at phoneme level, then joins into syllables

Note on entropy calculation:
- Uses count-based approach for better numerical stability
- Directly calculates probabilities from counts
- Properly handles zero counts (0 * log(0) = 0)
"""

import argparse
import numpy as np
from collections import defaultdict

def parse_syllable(syllable, connector='_'):
    """
    Parse a syllable string into list of phonemes.

    Args:
        syllable (str): Syllable string with phonemes connected by connector
        connector (str): Character used to connect phonemes in syllable

    Returns:
        list: List of phonemes in the syllable

    Example:
        'p_eng2' -> ['p', 'eng2']
        '@_m_iu5' -> ['@', 'm', 'iu5']
    """
    return syllable.split(connector)

def merge_syllable_phonemes(syllable_phonemes, phoneme1, phoneme2, connector='_'):
    """
    Merge phonemes in a syllable by replacing phoneme2 with phoneme1.

    Args:
        syllable_phonemes (list): List of phonemes in a syllable
        phoneme1 (str): Target phoneme to merge into
        phoneme2 (str): Source phoneme to be replaced
        connector (str): Character used to join phonemes back into syllable

    Returns:
        str: Merged syllable string

    Example:
        (['@', 'm', 'iu5'], 'p', 'm') -> '@_p_iu5'
    """
    merged = [phoneme1 if p == phoneme2 else p for p in syllable_phonemes]
    return connector.join(merged)

def get_ngrams(sequence, n):
    """
    Generate n-grams from a sequence.

    Args:
        sequence (list): List of items (phonemes or syllables)
        n (int): Size of n-grams to generate

    Returns:
        list: List of n-gram tuples
    """
    return [tuple(sequence[i:i+n]) for i in range(len(sequence)-n+1)]

def calculate_entropy(counts):
    """
    Calculate Shannon entropy from counts using numerically stable method.
    Converts counts directly to probabilities and handles zero counts properly.

    Args:
        counts (list): List of n-gram counts

    Returns:
        float: Entropy value in bits

    Note:
        - Uses count-based approach similar to count2entropy.cpp
        - Handles 0 * log(0) = 0 case explicitly
        - Uses direct log2 calculation for better numerical stability
    """
    if not counts:
        return 0.0

    total = sum(counts)
    entropy = 0.0

    # Calculate -sum(p * log2(p)) directly from counts
    for count in counts:
        if count > 0:  # Handle 0 * log(0) = 0 case
            prob = count / total
            entropy -= prob * np.log2(prob)

    return entropy

def calculate_functional_load(corpus_path, phoneme1, phoneme2, n=2, connector='_'):
    """
    Calculate phoneme-level functional load using n-gram entropy.
    Now handles syllables with multiple phonemes connected by connector character.

    Args:
        corpus_path (str): Path to corpus file with syllable sequences
        phoneme1 (str): First phoneme in the pair
        phoneme2 (str): Second phoneme in the pair
        n (int): Size of n-grams to use
        connector (str): Character used to connect phonemes in syllables

    Returns:
        tuple: (phoneme1, phoneme2, normalized_fl, absolute_fl_diff, original_entropy, merged_entropy)
    """
    with open(corpus_path, 'r') as f:
        corpus = f.read().strip().split('\n')

    # Split syllables into individual phonemes for original counts
    original_counts = defaultdict(int)
    for line in corpus:
        # Split syllables and then split each syllable into phonemes
        syllables = line.strip().split()
        phonemes = []
        for syllable in syllables:
            phonemes.extend(parse_syllable(syllable, connector))

        ngrams = get_ngrams(phonemes, n)
        for ngram in ngrams:
            original_counts[ngram] += 1

    original_entropy = calculate_entropy(list(original_counts.values()))

    # Generate merged n-grams
    merged_counts = defaultdict(int)
    for ngram, count in original_counts.items():
        merged_ngram = tuple(phoneme1 if p == phoneme2 else p for p in ngram)
        merged_counts[merged_ngram] += count

    merged_entropy = calculate_entropy(list(merged_counts.values()))

    fl_diff = original_entropy - merged_entropy
    fl = fl_diff / original_entropy if original_entropy != 0 else 0.0

    return phoneme1, phoneme2, fl, fl_diff, original_entropy, merged_entropy

def calculate_syllable_functional_load(corpus_path, phoneme1, phoneme2, n=2, connector='_'):
    """
    Calculate syllable-level functional load using n-gram entropy.
    Handles syllables with multiple phonemes connected by connector character.

    Args:
        corpus_path (str): Path to corpus file with syllable sequences
        phoneme1 (str): First phoneme in the pair
        phoneme2 (str): Second phoneme in the pair
        n (int): Size of n-grams to use
        connector (str): Character used to connect phonemes in syllables

    Returns:
        tuple: (phoneme1, phoneme2, normalized_fl, absolute_fl_diff, original_entropy, merged_entropy)
    """
    with open(corpus_path, 'r') as f:
        corpus = f.read().strip().split('\n')

    # Count original syllable n-grams
    original_counts = defaultdict(int)
    for line in corpus:
        syllables = line.strip().split()
        ngrams = get_ngrams(syllables, n)
        for ngram in ngrams:
            original_counts[ngram] += 1

    original_entropy = calculate_entropy(list(original_counts.values()))

    # Generate merged syllable n-grams
    merged_counts = defaultdict(int)
    for line in corpus:
        syllables = line.strip().split()
        merged_syllables = []
        for syllable in syllables:
            # Parse syllable into phonemes, merge, then reconstruct
            phonemes = parse_syllable(syllable, connector)
            merged_syllable = merge_syllable_phonemes(phonemes, phoneme1, phoneme2, connector)
            merged_syllables.append(merged_syllable)

        ngrams = get_ngrams(merged_syllables, n)
        for ngram in ngrams:
            merged_counts[ngram] += 1

    merged_entropy = calculate_entropy(list(merged_counts.values()))

    fl_diff = original_entropy - merged_entropy
    fl = fl_diff / original_entropy if original_entropy != 0 else 0.0

    return phoneme1, phoneme2, fl, fl_diff, original_entropy, merged_entropy

def main():
    """
    Main function to handle command-line arguments and run the functional load calculation.
    Supports both phoneme-level and syllable-level calculations with configurable syllable connector.
    """
    parser = argparse.ArgumentParser(
        description='Calculate functional load between two phonemes in a corpus'
    )
    parser.add_argument('--corpus',
                       default='exp/data/sample_tv07_phoneme.txt',
                       help='Path to corpus file with phoneme sequences')
    parser.add_argument('--phoneme1',
                       default='c',
                       help='First phoneme of the pair')
    parser.add_argument('--phoneme2',
                       default='sh',
                       help='Second phoneme of the pair')
    parser.add_argument('--ngram',
                       type=int,
                       default=2,
                       help='Size of n-grams to use in calculation')
    parser.add_argument('--syllable_fl',
                       action='store_true',
                       help='Calculate functional load at syllable level')
    parser.add_argument('--connector',
                       default='_',
                       help='Character used to connect phonemes in syllables')

    args = parser.parse_args()

    if args.syllable_fl:
        phoneme1, phoneme2, fl, fl_diff, original_entropy, merged_entropy = \
            calculate_syllable_functional_load(args.corpus, args.phoneme1, args.phoneme2,
                                            args.ngram, args.connector)
    else:
        phoneme1, phoneme2, fl, fl_diff, original_entropy, merged_entropy = \
            calculate_functional_load(args.corpus, args.phoneme1, args.phoneme2,
                                   args.ngram, args.connector)

    print(f"{phoneme1},{phoneme2},{fl:.4f},{fl_diff:.4f},{original_entropy:.4f},{merged_entropy:.4f}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
This script converts Pinyin sequences with word boundaries to syllable-based phoneme sequences.
It removes word boundary markers (#) and converts each syllable to its corresponding
initial and final components connected with a specified symbol (default '_'),
optionally including tone markers.

Example with tones:
    Input:  xian1#sheng5 men5
    Output: x_ian1 sh_eng5 m_en5
Example without tones:
    Input:  xian1#sheng5 men5
    Output: x_ian sh_eng m_en
"""

import argparse

def load_py2if_mapping(mapping_file):
    """[previous docstring remains the same]"""
    py2if = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                py2if[parts[0]] = (parts[1], parts[2], parts[3])
    return py2if

def convert_line(line, py2if_map, no_tone=False, remove_null_initial=False, connector='_'):
    """
    Convert a single line of Pinyin text to syllable-based phoneme sequence.

    Args:
        line (str): Input line containing Pinyin syllables with word boundaries
        py2if_map (dict): Mapping dictionary from load_py2if_mapping()
        no_tone (bool): If True, remove tone numbers from output
        remove_null_initial (bool): If True, remove @ symbols from initials
        connector (str): Symbol to connect initial and final components (default: '_')

    Returns:
        str: Space-separated sequence of connected syllable components with or without tones

    Example with remove_null_initial=True:
        Input:  "peng2#iu5 men5"
        Output: "p_eng2 iu5 m_en5"
    Example with remove_null_initial=False:
        Input:  "peng2#iu5 men5"
        Output: "p_eng2 @_iu5 m_en5"
    Example with tones:
        Input:  "xian1#sheng5 men5"
        Output: "x_ian1 sh_eng5 m_en5"
    Example without tones:
        Input:  "xian1#sheng5 men5"
        Output: "x_ian sh_eng m_en"
    """
    line = line.strip()
    words = line.split()
    result = []

    for word in words:
        # Split word into syllables at word boundaries
        syllables = word.split('#')
        for syllable in syllables:
            if syllable in py2if_map:
                initial, final, tone = py2if_map[syllable]

                # Handle syllables with or without null initials
                if initial == '@':
                    if remove_null_initial:
                        # If removing null initials, just add the final with tone
                        result.append(final + ('' if no_tone else tone))
                    else:
                        # Keep the @ symbol if not removing null initials
                        result.append(f"@{connector}{final}{'' if no_tone else tone}")
                else:
                    # Normal case: connect initial and final with connector
                    result.append(f"{initial}{connector}{final}{'' if no_tone else tone}")
            else:
                print(f"Warning: syllable {syllable} not found in mapping")
                result.append(syllable)

    return ' '.join(result)

def main():
    """
    Main function to handle command-line arguments and process the corpus.

    Command-line arguments:
        --dict: Path to PY2IF+Tone mapping file
        --corpus: Path to input corpus file
        --phoneme_corpus: Path to output phoneme corpus file
        --no_tone: Remove tone numbers from output
        --remove_null_initial: Remove @ symbols from initials
        --connector: Symbol to connect initial and final components (default: '_')
    """
    parser = argparse.ArgumentParser(
        description='Convert Pinyin sequences to syllable-based phoneme sequences')
    parser.add_argument('--dict', default='conf/data/PY2IF+Tone.txt',
                      help='Path to PY2IF+Tone mapping file')
    parser.add_argument('--corpus', default='conf/data/sample_PD99ToneSeg.txt',
                      help='Path to input corpus file')
    parser.add_argument('--phoneme_corpus',
                      default='exp/data/sample_PD99ToneSeg_phoneme.txt',
                      help='Path to output phoneme corpus file')
    parser.add_argument('--no_tone', action='store_true',
                      help='Remove tone numbers from output')
    parser.add_argument('--remove_null_initial', action='store_true',
                      help='Remove @ symbols from initials')
    parser.add_argument('--connector', default='_',
                      help='Symbol to connect initial and final components')

    args = parser.parse_args()

    py2if_map = load_py2if_mapping(args.dict)

    with open(args.corpus, 'r') as fin, open(args.phoneme_corpus, 'w') as fout:
        for line in fin:
            converted_line = convert_line(line, py2if_map,
                                       no_tone=args.no_tone,
                                       remove_null_initial=args.remove_null_initial,
                                       connector=args.connector)
            fout.write(converted_line + '\n')

if __name__ == '__main__':
    main()

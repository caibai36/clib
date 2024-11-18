#!/usr/bin/env python3
"""
Script to pair and sort input strings of format pXaY_ID where:
- X is a number (prefix)
- Y is either 1 or 2 (pair identifier)
- ID is the base identifier

Input example: p1a1_240711-0985 p2a1_240711-0989 p1a2_240711-0985 p2a2_240711-0989
Output example: p1a1_240711-0985 p1a2_240711-0985 p2a1_240711-0989 p2a2_240711-0989
"""

import sys
import re
from collections import defaultdict

def parse_string(name):
    """
    Parse string to extract prefix and base_id
    Example: p1a1_240711-0985 -> ('p1', '240711-0985', 'a1')
    """
    match = re.match(r'(p\d)(a[12])_(.+)', name)
    if match:
        prefix, a_type, base_id = match.groups()
        return prefix, base_id, a_type
    return None

def pair_strings(input_str):
    """
    Pair and sort input strings
    Returns:
        tuple: (output_string, total_count, preserved_count)
    """
    # Split input into list
    names = input_str.strip().split()
    total_count = len(names)
    
    # Group entries by prefix and base_id
    pairs = defaultdict(dict)
    for name in names:
        parsed = parse_string(name)
        if parsed:
            prefix, base_id, a_type = parsed
            key = f"{prefix}_{base_id}"
            pairs[key][a_type] = name
    
    # Build output with paired entries
    output = []
    incomplete_pairs = []
    preserved_count = 0
    
    for key in sorted(pairs.keys()):
        pair = pairs[key]
        
        # Check for complete pair
        if 'a1' in pair and 'a2' in pair:
            output.extend([pair['a1'], pair['a2']])
            preserved_count += 2
        else:
            # Store incomplete pairs for reporting
            if 'a1' not in pair:
                incomplete_pairs.append(f"No a1 entry found for {pair.get('a2')}")
            if 'a2' not in pair:
                incomplete_pairs.append(f"No a2 entry found for {pair['a1']}")
    
    # Print statistics
    print(f"\nStatistics:", file=sys.stderr)
    print(f"Total utterances: {total_count}", file=sys.stderr)
    print(f"Preserved utterances: {preserved_count}", file=sys.stderr)
    print(f"Ignored utterances: {total_count - preserved_count}", file=sys.stderr)
    
    # Print warnings for incomplete pairs
    if incomplete_pairs:
        print("\nWarnings:", file=sys.stderr)
        for warning in incomplete_pairs:
            print(f"Warning: {warning}", file=sys.stderr)
    
    return ' '.join(output)

def main():
    # Read from stdin if no arguments provided
    if len(sys.argv) == 1:
        input_str = sys.stdin.read()
    else:
        input_str = ' '.join(sys.argv[1:])
    
    # Process and print result
    result = pair_strings(input_str)
    print(result)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script to extract speaker information from .tag files and create speaker.csv
This file can be merged with all_kana_comment_token.csv using session_id as the key
"""

import os
import sys
import csv
import argparse
from pathlib import Path
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Mapping dictionaries based on documentation
SPEAKER_MAP = {
    'C': 'child',
    'F': 'father',
    'M': 'mother',
    'Z': 'unknown'
}

NOISE_MAP = {
    'I': 'quiet',
    'N': 'noisy'
}

LOUDNESS_MAP = {
    'L': 'too_low',
    'T': 'too_high'
}

DIRECTION_MAP = {
    'A': 'adult_to_adult',
    'B': 'adult_to_child'
}

def parse_tag_file(tag_path):
    """
    Parse a .tag file and extract speaker properties
    Format: Speaker + <space> + Noise + <space> + Loudness(opt) + <space> + Direction(opt)

    Returns dict with: speaker, noise, loudness, direction, original_tag
    """
    try:
        with open(tag_path, 'r', encoding='shift_jis') as f:
            content = f.read().strip()
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning(f"Error reading {tag_path}: {e}")
        return None

    if not content:
        return None

    parts = content.split()
    result = {
        'original_tag': content,
        'speaker': 'unknown',
        'noise': '',
        'loudness': '',
        'direction': ''
    }

    if len(parts) == 0:
        return result

    # First part is always speaker
    speaker_code = parts[0]
    if speaker_code in SPEAKER_MAP:
        result['speaker'] = SPEAKER_MAP[speaker_code]
    elif speaker_code.startswith('C') and len(speaker_code) > 1:
        result['speaker'] = 'other_child'
    elif speaker_code.startswith('O') and len(speaker_code) > 1:
        result['speaker'] = 'other_person'
    else:
        result['speaker'] = 'unknown'
        logger.warning(f"Unknown speaker code: {speaker_code} in {tag_path}")

    # Second part is noise level (if exists)
    if len(parts) > 1:
        noise_code = parts[1]
        result['noise'] = NOISE_MAP.get(noise_code, '')

    # Remaining parts can be loudness and/or direction
    for i in range(2, len(parts)):
        code = parts[i]
        if code in LOUDNESS_MAP:
            result['loudness'] = LOUDNESS_MAP[code]
        elif code in DIRECTION_MAP:
            result['direction'] = DIRECTION_MAP[code]

    return result

def extract_session_id(filename):
    """Extract session_id from filename (same logic as in bash script)"""
    import re

    # For format like sk-05_1_0001
    if re.match(r'^[a-z]{2}-[0-9]{2}_[1-9]', filename):
        parts = filename.split('_')
        return f"{parts[0]}_{parts[1]}"
    # For format like sa001_2_0310 or sk000_1_0357
    elif re.match(r'^[a-z]{2}[0-9]{3}_[1-9]', filename):
        parts = filename.split('_')
        return f"{parts[0]}_{parts[1]}"
    else:
        # Default case: take first 7 characters
        return filename[:7]

def process_infant_data(infant, base_path):
    """Process all tag files for one infant"""
    infant_path = Path(base_path) / infant

    if not infant_path.exists():
        logger.error(f"Path does not exist: {infant_path}")
        return []

    results = []
    tag_files = list(infant_path.rglob("*.tag"))

    logger.info(f"Processing {len(tag_files)} tag files for infant: {infant}")

    sessions_found = set()

    for tag_file in tag_files:
        filename = tag_file.stem  # filename without extension
        session_id = extract_session_id(filename)
        sessions_found.add(session_id)

        tag_info = parse_tag_file(tag_file)

        if tag_info is None:
            logger.warning(f"Could not parse tag file: {tag_file}")
            continue

        result = {
            'session_id': session_id,
            'session': filename,
            'speaker': tag_info['speaker'],
            'noise': tag_info['noise'],
            'loudness': tag_info['loudness'],
            'direction': tag_info['direction'],
            'original_tag': tag_info['original_tag']
        }

        results.append(result)

    logger.info(f"Found speaker info for {len(results)} files in {len(sessions_found)} sessions for {infant}")

    return results

def main():
    """Main function to process all infants and create speaker.csv"""
    parser = argparse.ArgumentParser(
        description='Extract speaker information from .tag files and create speaker.csv'
    )
    parser.add_argument(
        '--item_path',
        type=str,
        default='/data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/data/item',
        help='Path to the item directory containing infant data (default: /data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/data/item)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/local',
        help='Output directory for speaker.csv (default: data/local)'
    )
    parser.add_argument(
        '--infants',
        type=str,
        nargs='+',
        default=['sa', 'kk', 'ma', 'mk', 'sk'],
        help='List of infant IDs to process (default: sa kk ma mk sk)'
    )

    args = parser.parse_args()

    # Verify base path exists
    base_path = Path(args.item_path)
    if not base_path.exists():
        logger.error(f"Item path does not exist: {base_path}")
        sys.exit(1)

    logger.info(f"Item path: {base_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Processing infants: {args.infants}")

    all_results = []
    sessions_per_infant = defaultdict(set)

    for infant in args.infants:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing infant: {infant}")
        logger.info(f"{'='*60}")

        results = process_infant_data(infant, base_path)
        all_results.extend(results)

        for result in results:
            sessions_per_infant[infant].add(result['session_id'])

    # Write to CSV
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, 'speaker.csv')

    fieldnames = ['session_id', 'session', 'speaker', 'noise', 'loudness', 'direction', 'original_tag']

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    logger.info(f"\n{'='*60}")
    logger.info(f"Created {output_file} with {len(all_results)} entries")

    # Summary statistics
    for infant in args.infants:
        logger.info(f"{infant}: {len(sessions_per_infant[infant])} unique sessions")

    logger.info(f"\nDone! Speaker information saved to {output_file}")

if __name__ == "__main__":
    main()

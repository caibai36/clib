#!/usr/bin/env python3
"""
Merge speaker.csv with all_kana_comment_token.csv
"""

import pandas as pd
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def merge_speaker_info(main_csv, speaker_csv, output_csv):
    """Merge speaker information with main data file"""

    # Check if files exist
    if not Path(main_csv).exists():
        logger.error(f"Main CSV file does not exist: {main_csv}")
        sys.exit(1)

    if not Path(speaker_csv).exists():
        logger.error(f"Speaker CSV file does not exist: {speaker_csv}")
        sys.exit(1)

    # Read the files
    logger.info(f"Reading {main_csv}...")
    main_df = pd.read_csv(main_csv)

    logger.info(f"Reading {speaker_csv}...")
    speaker_df = pd.read_csv(speaker_csv)

    logger.info(f"Main data: {len(main_df)} rows")
    logger.info(f"Speaker data: {len(speaker_df)} rows")

    # Merge on session (the full session name like kk001_1_0001)
    merged_df = main_df.merge(
        speaker_df[['session', 'speaker', 'noise', 'loudness', 'direction', 'original_tag']],
        on='session',
        how='left'
    )

    # Check for missing speaker info
    missing_speaker = merged_df[merged_df['speaker'].isna()]
    if len(missing_speaker) > 0:
        logger.warning(f"\n{'='*60}")
        logger.warning(f"WARNING: {len(missing_speaker)} sessions have no speaker information!")
        logger.warning(f"{'='*60}")
        logger.warning(f"Missing sessions by session_id:")
        for session_id in missing_speaker['session_id'].unique():
            count = len(missing_speaker[missing_speaker['session_id'] == session_id])
            logger.warning(f"  {session_id}: {count} files")

        # Log some example missing sessions
        logger.warning(f"\nExample missing sessions:")
        for session in missing_speaker['session'].head(10):
            logger.warning(f"  {session}")
    else:
        logger.info("✓ All sessions have speaker information!")

    # Save merged file
    merged_df.to_csv(output_csv, index=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"Merged file saved to {output_csv}")
    logger.info(f"Total rows: {len(merged_df)}")
    logger.info(f"{'='*60}")

    # Print summary statistics
    logger.info("\nSpeaker distribution:")
    print(merged_df['speaker'].value_counts(dropna=False))

    logger.info("\nNoise distribution:")
    print(merged_df['noise'].value_counts(dropna=False))

    logger.info("\nDirection distribution:")
    print(merged_df['direction'].value_counts(dropna=False))

    logger.info("\nLoudness distribution:")
    print(merged_df['loudness'].value_counts(dropna=False))

def main():
    parser = argparse.ArgumentParser(
        description='Merge speaker.csv with all_kana_comment_token.csv'
    )
    parser.add_argument(
        '--main_csv',
        type=str,
        default='/data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/processed/all_kana_comment_token.csv',
        help='Path to main CSV file with kana and comment tokens'
    )
    parser.add_argument(
        '--speaker_csv',
        type=str,
        default='data/local/speaker.csv',
        help='Path to speaker CSV file (default: data/local/speaker.csv)'
    )
    parser.add_argument(
        '--output_csv',
        type=str,
        default='data/local/all_kana_comment_token_speaker.csv',
        help='Path to output merged CSV file (default: data/local/all_kana_comment_token_speaker.csv)'
    )

    args = parser.parse_args()

    logger.info(f"Main CSV: {args.main_csv}")
    logger.info(f"Speaker CSV: {args.speaker_csv}")
    logger.info(f"Output CSV: {args.output_csv}")

    merge_speaker_info(args.main_csv, args.speaker_csv, args.output_csv)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create a new CSV file with an additional kana_comment_token column.

This script reads data/local/all_kana_token.csv and adds a new column called
kana_comment_token. If kana_token is empty and comment matches a key in
comment_rep.txt, it uses the replacement value. Otherwise, it copies the
kana_token value.
"""

import csv
import sys

def load_comment_replacements(file_path="conf/comment_rep.txt"):
    """
    Load comment replacement mappings from file.

    Args:
        file_path: Path to the comment replacement file

    Returns:
        Dictionary mapping comment strings to their replacements
    """
    comment_rep = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(': ', 1)
                if len(parts) == 2:
                    comment_rep[parts[0]] = parts[1]
    except FileNotFoundError:
        print(f"Error: Comment replacement file {file_path} not found!", file=sys.stderr)
        sys.exit(1)

    return comment_rep

def process_csv(input_file="data/local/all_kana_token.csv",
                output_file="data/local/all_kana_comment_token.csv",
                comment_rep_file="conf/comment_rep.txt"):
    """
    Process the CSV file and create a new one with the kana_comment_token column.

    Args:
        input_file: Path to the input CSV file
        output_file: Path to the output CSV file
        comment_rep_file: Path to the comment replacement file
    """
    # Load comment replacements
    comment_rep = load_comment_replacements(comment_rep_file)

    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8', newline='') as f_out:

            reader = csv.DictReader(f_in)
            # Add the new column to the fieldnames
            fieldnames = reader.fieldnames + ['kana_comment_token']

            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                # If kana_token is empty and comment is in comment_rep
                if (not row['kana_token'] or row['kana_token'].strip() == '') and row['comment'] in comment_rep:
                    row['kana_comment_token'] = comment_rep[row['comment']]
                else:
                    # Copy kana_token to kana_comment_token
                    row['kana_comment_token'] = row['kana_token']

                writer.writerow(row)

        print(f"Successfully created {output_file}")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing CSV: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """Main function to process the CSV file."""
    # Use default paths for simplicity
    process_csv()

if __name__ == "__main__":
    main()

import pandas as pd
import json
import argparse

def process_compound_calls(input_csv, output_txt, output_json=None, output_csv=None,
                         compound_threshold=0.02, utterance_threshold=1.0,
                         process_compound=True):
    """Process call annotations to generate compound calls and utterances.

    Args:
        input_csv (str): Path to input CSV file
        output_txt (str): Path to output text file for utterances
        output_json (str, optional): Path to output JSON file for detailed info
        output_csv (str, optional): Path to output CSV file with utterance info
        compound_threshold (float): Max time gap (seconds) for compound calls
        utterance_threshold (float): Max time gap (seconds) for utterance
        process_compound (bool): Whether to process compound calls

    The function generates three output files:
    1. Text file: Contains utterances (space-separated calls)
    2. JSON file (optional): Contains detailed utterance information
    3. CSV file (optional): Contains utterance-level information with metadata

    Example input CSV format:
    dataid,audioid,age_days,age_weeks,begin_sec,end_sec,label,duration,ext_cut_begin_sec,ext_cut_end_sec
    b0_f1,230724_001,0,0,9.77,9.83,tr,0.06,9.55,10.05
    b0_f1,230724_001,11.72,11.79,ek,0.07,11.50,12.00
    """
    # Read CSV file
    df = pd.read_csv(input_csv)

    # Create age lookup dictionary for each audio_id
    age_info = df.groupby('audioid').agg({
        'age_days': 'first',
        'age_weeks': 'first'
    }).to_dict('index')

    # Replace u-x labels with x
    label_replacements = {
        'u-pp': 'pp', 'u-ct': 'ct', 'u-cr': 'cr', 'u-cp': 'cp',
        'u-ek': 'ek', 'u-ph': 'ph', 'u-tr': 'tr', 'u-ts': 'ts',
        'u-se': 'se', 'u-ok': 'ok', 'u-tw': 'tw'
    }
    df['label'] = df['label'].replace(label_replacements)

    utterances = []
    utterances_info = []

    for audio_id in df['audioid'].unique():
        audio_df = df[df['audioid'] == audio_id].sort_values('begin_sec')

        current_utterance = []
        current_utterance_info = {
            'audio_id': audio_id,
            'age_days': age_info[audio_id]['age_days'],
            'age_weeks': age_info[audio_id]['age_weeks'],
            'calls': [],
            'begin_sec': None,
            'end_sec': None
        }

        i = 0
        while i < len(audio_df):
            current_row = audio_df.iloc[i]

            # First, check if we should start a new utterance
            if not current_utterance:
                # Initialize first utterance
                current_utterance_info['begin_sec'] = current_row['begin_sec']
            else:
                # Check time gap to previous call's end
                time_from_prev = current_row['begin_sec'] - audio_df.iloc[i-1]['end_sec']

                if time_from_prev > utterance_threshold:
                    # Finish previous utterance and calculate duration
                    utterances.append(' '.join(current_utterance))
                    current_utterance_info['end_sec'] = audio_df.iloc[i-1]['end_sec']
                    current_utterance_info['duration'] = round(current_utterance_info['end_sec'] -
                                                            current_utterance_info['begin_sec'], 6)
                    utterances_info.append(current_utterance_info)

                    # Start new utterance
                    current_utterance = []
                    current_utterance_info = {
                        'audio_id': audio_id,
                        'age_days': age_info[audio_id]['age_days'],
                        'age_weeks': age_info[audio_id]['age_weeks'],
                        'calls': [],
                        'begin_sec': current_row['begin_sec'],
                        'end_sec': None
                    }

            # Now process the current call
            if process_compound and i + 1 < len(audio_df):
                # Process compound calls
                compound_calls = [current_row['label']]
                compound_calls_info = [{
                    'label': current_row['label'],
                    'begin_sec': current_row['begin_sec'],
                    'end_sec': current_row['end_sec'],
                    'duration': current_row['duration'],
                    'ext_cut_begin_sec': current_row['ext_cut_begin_sec'],
                    'ext_cut_end_sec': current_row['ext_cut_end_sec']
                }]

                while i + 1 < len(audio_df):
                    next_row = audio_df.iloc[i + 1]
                    time_diff = next_row['begin_sec'] - audio_df.iloc[i]['end_sec']

                    if time_diff <= compound_threshold and time_diff <= utterance_threshold:
                        compound_calls.append(next_row['label'])
                        compound_calls_info.append({
                            'label': next_row['label'],
                            'begin_sec': next_row['begin_sec'],
                            'end_sec': next_row['end_sec'],
                            'duration': next_row['duration'],
                            'ext_cut_begin_sec': next_row['ext_cut_begin_sec'],
                            'ext_cut_end_sec': next_row['ext_cut_end_sec']
                        })
                        i += 1
                    else:
                        break

                if len(compound_calls) > 1:
                    call = '_'.join(compound_calls)
                    current_call_info = {
                        'compound_call': call,
                        'begin_sec': compound_calls_info[0]['begin_sec'],
                        'end_sec': compound_calls_info[-1]['end_sec'],
                        'duration': round(compound_calls_info[-1]['end_sec'] - compound_calls_info[0]['begin_sec'], 6),
                        'ext_cut_begin_sec': compound_calls_info[0]['ext_cut_begin_sec'],
                        'ext_cut_end_sec': compound_calls_info[-1]['ext_cut_end_sec'],
                        'subcalls': compound_calls_info
                    }
                else:
                    call = compound_calls[0]
                    current_call_info = {
                        'call': call,
                        'begin_sec': current_row['begin_sec'],
                        'end_sec': current_row['end_sec'],
                        'duration': current_row['duration'],
                        'ext_cut_begin_sec': current_row['ext_cut_begin_sec'],
                        'ext_cut_end_sec': current_row['ext_cut_end_sec']
                    }
            else:
                # Process single call
                call = current_row['label']
                current_call_info = {
                    'call': call,
                    'begin_sec': current_row['begin_sec'],
                    'end_sec': current_row['end_sec'],
                    'duration': current_row['duration'],
                    'ext_cut_begin_sec': current_row['ext_cut_begin_sec'],
                    'ext_cut_end_sec': current_row['ext_cut_end_sec']
                }

            # Add call to current utterance
            current_utterance.append(call)
            current_utterance_info['calls'].append(current_call_info)

            i += 1

        # Handle final utterance
        if current_utterance:
            utterances.append(' '.join(current_utterance))
            current_utterance_info['end_sec'] = audio_df.iloc[-1]['end_sec']
            current_utterance_info['duration'] = round(current_utterance_info['end_sec'] -
                                                    current_utterance_info['begin_sec'], 6)
            utterances_info.append(current_utterance_info)

    # Write outputs
    with open(output_txt, 'w') as f:
        for utterance in utterances:
            f.write(f"{utterance}\n")

    if output_json:
        with open(output_json, 'w') as f:
            json.dump(utterances_info, f, indent=2)

    if output_csv:
        # Create CSV with utterance info
        csv_data = []
        for utterance_info, utterance in zip(utterances_info, utterances):
            row = {
                'audioid': utterance_info['audio_id'],
                'age_days': utterance_info['age_days'],
                'age_weeks': utterance_info['age_weeks'],
                'begin_sec': utterance_info['begin_sec'],
                'end_sec': utterance_info['end_sec'],
                'duration': utterance_info['duration'],
                'utterance': utterance
            }
            csv_data.append(row)

        pd.DataFrame(csv_data).to_csv(output_csv, index=False)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Process call annotations to generate compound calls and utterances.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--input-csv',
        default='data/nas5_b0_f1/b0_f1_annotated.csv',
        help='Path to input CSV file'
    )

    parser.add_argument(
        '--output-txt',
        default='exp/marmoset_data/b0_f1_compound_calls.txt',
        help='Path to output text file'
    )

    parser.add_argument(
        '--compound-threshold',
        type=float,
        default=0.02,
        help='Max time gap (seconds) for compound calls'
    )

    parser.add_argument(
        '--utterance-threshold',
        type=float,
        default=1.0,
        help='Max time gap (seconds) for utterance'
    )

    parser.add_argument(
        '--process-compound',
        action='store_true',
        default=True,
        help='Enable compound call processing'
    )

    parser.add_argument(
        '--no-process-compound',
        action='store_false',
        dest='process_compound',
        help='Disable compound call processing'
    )

    parser.add_argument(
        '--no-json',
        action='store_true',
        help='Disable JSON output'
    )

    parser.add_argument(
        '--no-csv',
        action='store_true',
        help='Disable CSV output'
    )

    args = parser.parse_args()
    return args

def main(args):
    """Main function to run the script."""

    # Derive JSON and CSV paths from txt path
    base_path = args.output_txt.rsplit('.', 1)[0]
    output_json = None if args.no_json else f"{base_path}.json"
    output_csv = None if args.no_csv else f"{base_path}.csv"

    process_compound_calls(
        input_csv=args.input_csv,
        output_txt=args.output_txt,
        output_json=output_json,
        output_csv=output_csv,
        compound_threshold=args.compound_threshold,
        utterance_threshold=args.utterance_threshold,
        process_compound=args.process_compound
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)
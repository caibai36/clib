import argparse
import json
import os

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal

def create_spec_data(wav_file, save_loc, segment_duration=2.5, sample_rate=48000, nfft=512, noverlap=420):
    # Read the WAV file
    rate, data = wavfile.read(wav_file)
    num_segments = int(len(data) / (rate * segment_duration))

    spec_data = []
    for i in range(num_segments - 1): # Remove the last segment that might be incomplete
        # Extract the audio segment
        start = int(rate * segment_duration * i)
        end = int(rate * segment_duration * (i + 1))
        signal_piece = data[start:end]

        # Resample the audio segment if necessary
        if rate != sample_rate:
            signal_piece = signal.resample(signal_piece, int(len(signal_piece) * sample_rate / rate))

        # Compute the spectrogram of the audio segment
        spectrum, freqs, time, image = plt.specgram(signal_piece, NFFT=nfft, Fs=sample_rate,
                                                    window=np.hamming(nfft), noverlap=noverlap,
                                                    scale='linear', detrend='none')
        spectrum = 10 * np.log(np.abs(spectrum + 1e-6))
        if spectrum.shape != (257, 1299):
            print(f"Invalid array shape: {spectrum.shape}")
            continue
        spec_data.append(spectrum.astype(np.float16))

        plt.clf()

        # Print progress information
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{num_segments} segments")

    # Save the spectrogram data as a numpy array
    spec_data = np.array(spec_data, dtype=np.float16)
    with open(save_loc, 'wb') as f:
        np.save(f, spec_data)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Create 2500ms spectral segments to prepare the inputs of test sets. Split audio pairs into 2500ms segments and take the spectral frames for each segment (reference: 'wav_into_test_2500_raw.py' from https://marmosetbehavior.mit.edu/).\n\nThe script preprocesses audio files for a test set by splitting them into 2500ms segments, computing log-scaled linear spectrograms for each segment, and saving the stacked spectrograms as numpy arrays.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--info_json", type=str, default="data/mit_sample/info.json", help="JSON file that maps utterance ID to key-value pairs. The keys should include 'wav' and 'aud' for locations of audio files and audacity labels. {'uttid1': {'wav1': wav1_path, 'aud1': audacity_label1_path}, 'uttid2': {'wav2': wav2_path, 'aud2': audacity_label2_path}}. Each line of the audacity label file should be in the format 'begin_time_sec end_time_sec label'.")
    parser.add_argument("--test_wav_uttids", type=str, default=["Athos", "Porthos"], nargs="+", help="Utterance ID pairs for test sets. e.g., '--test_wav_uttids Cricket Enid Setta Sailor' where 'Cricket' and 'Enid' are the first pair, and 'Setta' and 'Sailor' are the second pair.")
    parser.add_argument("--test_wav_paths", type=str, default=[], nargs="+", help="Path pairs for test sets. Instead of passing a sequence of utterance IDs using --test_wav_uttids, this option directly passes paths of WAV files. The info_json file is not needed here.")
    parser.add_argument("--out_dir", type=str, default="exp/data/mit_sample", help="Output directory to store the spectra of 2500ms segments to prepare the inputs of test sets. e.g., the output file would be test_input1_uttid1 and test_input2_wavfilename (if WAV paths are provided).")
    args = parser.parse_args()
    print(args)

    # Load WAV file paths from JSON or command-line arguments
    if args.test_wav_uttids:
        with open(args.info_json) as f:
            info = json.load(f)
        wavs = [info[uttid]['wav'] for uttid in args.test_wav_uttids]
    elif args.test_wav_paths:
        wavs = args.test_wav_paths
    else:
        raise ValueError("Either --test_wav_uttids or --test_wav_paths must be provided.")

    # Create the output directory if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)

    # Process each pair of WAV files
    for i in range(0, len(wavs), 2):
        # Get the names for the output files
        name1 = args.test_wav_uttids[i] if args.test_wav_uttids else os.path.splitext(os.path.basename(wavs[i]))[0]
        name2 = args.test_wav_uttids[i + 1] if args.test_wav_uttids else os.path.splitext(os.path.basename(wavs[i + 1]))[0]

        # Create the output file paths
        save_loc1 = os.path.join(args.out_dir, f"test_input1_{name1}")
        save_loc2 = os.path.join(args.out_dir, f"test_input2_{name2}")

        # Create spectral data for each WAV file and save it
        create_spec_data(wavs[i], save_loc1)
        create_spec_data(wavs[i + 1], save_loc2)

if __name__ == "__main__":
    main()

# Implemented by bin-wu at 11:21 on 4 April 2024 from the MIT's implementation
import matplotlib as mpl
mpl.use('Agg') # Necessary when running on a server

import os
import random

import argparse
import json

import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Create training and development datasets of audios and labels stored in numpy array format (reference: 'create_data_17.py' from https://marmosetbehavior.mit.edu/). Split audio into chunks using a sliding window. Assign a target label to the chunk when the middle part of the chunk overlaps with annotated segments. Extract 257x256 specgram for each chunk as input. Note that we only keep the every nth all-noise-no-label chunk.")
parser.add_argument("--info_json", type=str, default="data/mit_sample/info.json", help="JSON file that maps utterance ID to key-value pairs. The keys should include 'wav' and 'aud' for locations of audio files and Audacity labels (e.g., {'uttid1': {'wav1': wav1_path, 'aud1': audacity_label1_path}, 'uttid2': {'wav2': wav2_path, 'aud2': audacity_label2_path}}). Each line of the Audacity label file should be in the format 'begin_time_sec end_time_sec label'.")
parser.add_argument("--train_dev", type=str, default=["Cricket", "Enid", "Setta", "Sailor"], nargs="+", help="All utterance IDs for training and development sets. Utterance IDs are a sequence of animal pairs. e.g., '--train_dev Cricket Enid Setta Sailor' where 'Cricket' and 'Enid' are the first pair, and 'Setta' and 'Sailor' are the second pair.")
parser.add_argument("--dev_pair_ind", type=int, default=[1], nargs="+", help="The indices of pairs in the train_dev array to be used as the development set. Taking value 1 means using the second pair as the development set. e.g., ['Cricket', 'Enid', 'Setta', 'Sailor'] would take ['Setta', 'Sailor'] as the development set and the remaining pairs ['Cricket', 'Enid'] as the training set.")
parser.add_argument("--out_dir", type=str, default="exp/data/mit_sample", help="Output directory to store the spectra and labels for training and development sets. Labels are stored as multihots containing two animal call types such as 'tr' and 'tr2'; Note a target one vector may contain 'tr' and 'ph2' from two animals at the same time")

args = parser.parse_args()
print(args)

with open(args.info_json) as f:
    info = json.load(f)
files = [info[uttid]['wav'] for uttid in args.train_dev]
label_files = [info[uttid]['aud'] for uttid in args.train_dev]
save_dir = args.out_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Symmetric label-to-id mapping to allow easy flipping when randomizing input order
label2id = {'cha':0,'chi':1,'ek':2,'ph':3,'ts':4,'tr':5,'trph':6,'tw':7,'noise':8,
              'tw2':9,'trph2':10,'tr2':11,'ts2':12,'ph2':13,'ek2':14,'chi2':15,'cha2':16}

# Sessions to be used for evaluation (eval_sessions=[1])
eval_sessions = args.dev_pair_ind

train_input1 = []
train_input2 = []
train_target = []
eval_input1 = []
eval_input2 = []
eval_target = []

for k in range(int(len(files) / 2)): # List of files: pair1_first, pair1_second, pair2_first, pair2_second, ...
    lines1 = []
    lines2 = []
    first = float('inf') # the chunk index of the first label of the first animal of the pair (shared by the second animal)

    for lines, label_file in [(lines1, label_files[2*k]), (lines2, label_files[2*k+1])]:
        print(f"Processing segment label file: {label_file}...")
        with open(label_file, 'r') as labels:
            segments = [line.split('\t') for line in labels if line.split('\t')[0] != "\\"] # some audacity files contain lines of "\ min_freq max_freq"

        current = 0
        start_window = current * 0.15
        end_window = start_window + 0.5
        middle_start = start_window + 0.175
        middle_end = start_window + 0.325

        for start_t, end_t, label in segments:
            # Slide a window with size 0.5s and shift 0.15s.
            # When the middle 0.15s-part of the current window overlaps with the label segment,
            # assign the label to the window; otherwise, assign "noise".
            start_t, end_t = float(start_t), float(end_t)

            # Move the window until it reaches the current segment
            while max(start_t, middle_start) > min(end_t, middle_end) and start_t > middle_end:
                lines.append("noise")
                current += 1
                start_window = current * 0.15
                end_window = start_window + 0.5
                middle_start = start_window + 0.175
                middle_end = start_window + 0.325

            # Assign the label to the window while it overlaps with the segment
            while max(start_t, middle_start) <= min(end_t, middle_end): # overlap
                if current < first:
                    first = current # Record the first window overlaps with the label segment
                lines.append(label.rstrip().lower()) # Remove the empty space or the remaining /r from window's newline (/r/n) after the label.
                current += 1
                start_window = current * 0.15
                end_window = start_window + 0.5
                middle_start = start_window + 0.175
                middle_end = start_window + 0.325

    # Pad the smaller list with noise to get to the same size
    length = max(len(lines1), len(lines2))
    lines1.extend(['noise'] * (length - len(lines1)))
    lines2.extend(['noise'] * (length - len(lines2)))

    # Create targets for windows
    # Similar to one-hot encoding, but when two animals call at the same time, two 1s would exist in a target vector
    lines = []
    for line1_label, line2_label in zip(lines1, lines2):
        init_y = np.zeros(len(label2id)) # Target vector for the current window

        init_y[label2id.get(line1_label, label2id['noise'])] = 1 if line1_label != 'noise' else 0 # Assign the label not in the label2id list as noise
        init_y[label2id.get(line2_label + '2', label2id['noise'])] = 1 if line2_label != 'noise' else 0
        if np.sum(init_y) == 0:
            init_y[label2id['noise']] = 1 # No calls exist

        lines.append(init_y)

    nfft_a=512
    rate, data = wavfile.read(files[2*k])
    rate2, data2 = wavfile.read(files[2*k+1])

    # Only look between the first and the last label
    print(f"Creating wav chunks from the chunk index of the first label: {first} to the chunk index of the last label: {length}...")
    for i in range(first, length):
        # Create the spectrogram if the current piece is not labeled as noise
        # Or pick every fifth piece that is labeled as noise
        if lines[i][label2id['noise']] != 1 or i % 5 == 0:
            # Take a 500ms piece out of the data with a step size of 150ms
            start_index1 = int(rate * i * 0.15)
            end_index1 = int(rate * (i * 0.15 + 0.5))
            signal_piece1 = data[start_index1:end_index1]

            # Resample the signal piece if the file has a rate that is not 48kHz
            if rate != 48000:
                target_length = int(len(signal_piece1) * 48000 / rate)
                signal_piece1 = signal.resample(signal_piece1, target_length)

            # Create the spectrogram and scale it logarithmically
            # Frame size 10.7ms and shift 1.926ms with 82% overlap
            # Got spectrogram images of size 257x256
            # (500-10.7)/1.926 + 1 = 255.0498
            spectrum1, freqs1, time1, image1 = plt.specgram(signal_piece1, NFFT=nfft_a, Fs=48000,
                                                            window=np.hamming(nfft_a), noverlap=420,
                                                            scale='linear', detrend='none')
            result1 = 10 * np.log(abs(spectrum1 + 0.000001))

            # Skip this piece if the shape is not as expected
            if np.shape(result1) != (257, 256):
                print("Invalid shape1:", np.shape(result1))
                continue

            # Repeat the process for the second wav file
            start_index2 = int(i * rate2 * 0.15)
            end_index2 = int(rate2 * ((i + 1) * 0.15 + 0.35))
            signal_piece2 = data2[start_index2:end_index2]

            if rate2 != 48000:
                target_length = int(len(signal_piece2) * 48000 / rate2)
                signal_piece2 = signal.resample(signal_piece2, target_length)

            spectrum2, freqs2, time2, image2 = plt.specgram(signal_piece2, NFFT=nfft_a, Fs=48000,
                                                            window=np.hamming(nfft_a), noverlap=420,
                                                            scale='linear', detrend='none')
            result2 = 10 * np.log(abs(spectrum2 + 0.000001))

            if np.shape(result2) != (257, 256):
                print("Invalid shape2:", np.shape(result2))
                continue

            # Add arrays to the correct set
            if k in eval_sessions:
                eval_target.append(lines[i])
                eval_input1.append(np.array(result1, dtype=np.float16))
                eval_input2.append(np.array(result2, dtype=np.float16))
            else:
                train_target.append(lines[i])
                train_input1.append(np.array(result1, dtype=np.float16))
                train_input2.append(np.array(result2, dtype=np.float16))

            plt.clf()
            if i % 1000 == 0:
                print(f"Processing the {i}th chunk of the {k}th pair...")


train_input1 = np.array(train_input1, dtype=np.float16)
train_input2 = np.array(train_input2, dtype=np.float16)
train_target = np.array(train_target, dtype=np.float16)
eval_input1 = np.array(eval_input1, dtype=np.float16)
eval_input2 = np.array(eval_input2, dtype=np.float16)
eval_target = np.array(eval_target, dtype=np.float16)

print('Length of training set:', len(train_input1))
print('Length of development set:', len(eval_input1))

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# Save training and development sets
# Training spectrograms for the animal1
# Training spectrograms for the animal2
# Training labels ('multi' means that target multihots contain two animal call types and calls from two animals from the same time)
# Development spectrograms for the animal1
# Development spectrograms for the animal2
# Development labels
data_sets = [
    ('train_input1', train_input1),
    ('train_input2', train_input2),
    ('train_target_multi', train_target),
    ('dev_input1', eval_input1),
    ('dev_input2', eval_input2),
    ('dev_target_multi', eval_target)
]

for filename, data in data_sets:
    file_path = os.path.join(save_dir, filename)
    with open(file_path, 'wb') as save_file:
        np.save(save_file, data)

print('Data saved successfully.')

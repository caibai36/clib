#!/usr/bin/env python3

# Implemented by bin-wu at 20:12 on 2022.05.28
# Script to segment long recordings.

import os
import argparse

import re

import numpy as np

import soundfile
import librosa

def load_wav(path, begin_sec=None, end_sec=None,sampling_rate=16000):
    """
    Load a wav file or its segment (bin-wu adapted from Tacotron).

    Parameters
    ----------
    path: the path of the audio
    begin_sec: the beginning time (in second) of the segment of the audio
    end_sec: the end time (in second) of the segment of the audio

    Returns
    -------
    the data of wav file or the data of its segment
    """
    wav = librosa.core.load(path, sr=sampling_rate)[0]
    if begin_sec is None and end_sec is None: return wav
    else:
        if begin_sec is None: begin_index = 0
        if end_sec is None: end_index = len(wav)
        assert begin_sec < end_sec, "the begin time of the segment should be less than the end time."
        begin_index = max(int(begin_sec * sampling_rate), 0)
        end_index = min(int(end_sec * sampling_rate), len(wav))

        return wav[begin_index:end_index]

def save_wav(wav, path, sampling_rate=16000) :
    """
    Save a waveform data to an audio file at path (bin-wu adapted from Tacotron).

    Parameters
    ----------
    path: the path of the audio
    """
    # Normalize to 16-bit range
    # https://simpleaudio.readthedocs.io/en/latest/tutorial.html
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    try :
        librosa.output.write_wav(path, wav.astype(np.int16), sampling_rate)
    except :
        soundfile.write(path, wav.astype(np.int16), sampling_rate)

def read_scp(path: str = ""):
    """
    Read the Kaldi script file into a dict.
    Each line of Kaldi script file has format of <uttid> <content>.
    The dict has key of <uttid> and value of <content>.
    """
    scp_dict = {}

    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            uttid, content = re.split('\s+', line, maxsplit=1)
            scp_dict[uttid] = content

    return scp_dict

def read_segments(path: str = ""):
    """
    Read the Kaldi segments file into a dict.
    The Kaldi segments file has line format of <uttid> <recid> <begin_sec> <end_sec>.
    The dict maps <uttid> to {"uttid": <uttid>, "recid": <recid>, "begin_sec": <begin_sec>, "end_sec": <end_sec>}.
    """
    segments_dict = {}

    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            uttid, recid, begin_sec, end_sec = re.split('\s+', line)
            segments_dict[uttid] = {"uttid": uttid, "recid": recid, "begin_sec": float(begin_sec), "end_sec": float(end_sec)}

    return segments_dict

parser = argparse.ArgumentParser(description="Split long recordings into segments of recordings. (NOTE: We normalize the segments into their maximum amplitudes to make them more friendly to hear).\n\nLong recordings are at the `wav.scp' file, each line with format of <recording_id> <recording_path>\nSegments of recordings are at the `segments' file, each line with format of <utterance_id> <recording_id> <begin_sec> <end_sec>.\nOutput short segmented recordings named as <utterance_id>.wav\nSee more details about 16-bit amplitude normalization at https://simpleaudio.readthedocs.io/en/latest/tutorial.html.\n", formatter_class=argparse.RawTextHelpFormatter)
wav_scp = parser.add_argument('--wav_scp', type=str, default="data/all/tmp/wav_all_test.scp", help="the Kaldi `wav.scp' file with line format of <recording_id> <recording_path>")
segments = parser.add_argument('--segments', type=str, default="data/all/tmp/segments_all_test", help="the Kaldi `segments' file with line format of <utterance_id> <recording_id> <begin_sec> <end_sec>")
output_dir = parser.add_argument('--output_dir', type=str, default="data/all/tmp/", help="the output directory to store the segmented recordings, each with name of <utterance_id>.wav")
output_wav_scp = parser.add_argument("--output_wav_scp", type=str, default="data/all/tmp/wav_seg_test.scp", help="the Kaldi `wav.scp' file of segmented recordings")
sampling_rate = parser.add_argument('--sampling_rate', type=int, default=16000, help="the sampling rate of the recordings (default 16000)")

args = parser.parse_args()

wav_scp = args.wav_scp
segments = args.segments
output_dir = args.output_dir
sampling_rate = args.sampling_rate
output_wav_scp = args.output_wav_scp

if (not os.path.exists(args.output_dir)):
    os.makedirs(args.output_dir)

uttid2wavpath = read_scp(wav_scp)
uttid2segments = read_segments(segments)

uttid2segwavpath = {}
for uttid, seg in uttid2segments.items():
    long_wav_path = uttid2wavpath[seg['recid']]
    seg_wav_path = os.path.join(output_dir, uttid + ".wav")

    wav = load_wav(long_wav_path, begin_sec=seg['begin_sec'], end_sec=seg['end_sec'], sampling_rate=sampling_rate)
    save_wav(wav, seg_wav_path, sampling_rate=sampling_rate)
    uttid2segwavpath[uttid] = os.path.abspath(seg_wav_path)

with open(output_wav_scp, 'wt', encoding='utf8') as f: # option 'w' means text model
    for uttid, path in uttid2segwavpath.items():
        f.write("{} {}\n".format(uttid, path))

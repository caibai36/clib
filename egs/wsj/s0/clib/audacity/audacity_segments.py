# Implmented by bin-wu on 2023 and March 2024
import re
from math import ceil

import torch
import torchaudio

class Segment:
    def __init__(self, begin_sec, end_sec, label, low_freq=None, high_freq=None):
        """A segment structure in audacity format

        Parameters
        ----------
        begin_sec : the begin time of a segment
        end_sec : the end time of a segment
        label : the label of the segment
        low_freq : the low frequency of a segment
        high_freq : the high frequency of a segment

        Examples
        --------
        0.108084        0.153355        tr
        \       5468.574219     10856.178711
        0.675958        1.387807        ph
        \       7743.052246     10856.178711
        1.722795        2.404735        ph
        \       7372.833496     9630.715820
        """
        self.begin_sec = begin_sec
        self.end_sec = end_sec
        self.label = label
        self.low_freq = low_freq
        self.high_freq= high_freq

    def __repr__(self):
        return str((self.begin_sec, self.end_sec, self.label, self.low_freq, self.high_freq))

def read_audacity_segments(segment_file):
    """Read the audacity segment file.

    Parameters
    ----------
    segment_file : the path the segment file of audacity

    Returns
    -------
    An segment list with each segment that has variables of begin_sec, end_sec, low_freq, and high_freq

    Examples
    --------
    # Audacity format of each line: (begin_sec end_sec label), or optionally (\ lowest_freq high_freq)
    $ cat test_audacity_segments.txt
        0.108084        0.153355        tr
        \       5468.574219     10856.178711
        0.675958        1.387807        ph
        \       7743.052246     10856.178711
        1.722795        2.404735        ph
        \       7372.833496     9630.715820
    """
    with open(segment_file, encoding='utf8') as f:
        segments = []
        for line in f:
            line = line.strip()
            first, second, third = re.split("\s+", line)

            if first != "\\":
                s = Segment(begin_sec=float(first), end_sec=float(second), label=str(third))
                segments.append(s)
            else:
                segments[-1].low_freq = float(second)
                segments[-1].high_freq = float(third)

    return segments


def sample2time(sample_index, sr=16000):
    """ Index of a sample of a waveform to the time of the waveform."""
    time = sample_index / float(sr)
    return time

def time2sample(time, sr=16000):
    """ Time of a waveform to the left closest sample index of the waveform."""
    sample_index = time * sr
    return int(sample_index)

def index2time(frame_index, window_size=0.025, window_shift=0.01, ndigits=7, center=True):
    """ Find time of a frame center according to its index (index2framecentertime).

    Parameters
    ----------
    frame_index: the index of the frame
    window_size: the frame window length of each frame in seconds (default: 0.025)
    window_shift: the frame window shift between successive frames in seconds  (default: 0.01)
    ndigits: The number of decimals to use when rounding the time (default 7)
    center (bool): True when origin is the frame center, False when origin is the frame start.

    Returns
    -------
    the time of the center of the give frame in seconds

    Example
    -------
    window_size_ms=50
    window_shift_ms=12.5
    print(index2time(frame_index=1, window_size=window_size_ms/1000, window_shift=window_shift_ms/1000)) # 0.0375
    print(time2index(0.0376, window_size=window_size_ms/1000, window_shift=window_shift_ms/1000)) # 1
    print(index2time(frame_index=1, window_size=window_size_ms/1000, window_shift=window_shift_ms/1000, center=True)) # 0.125
    print(time2index(0.0124, window_size=window_size_ms/1000, window_shift=window_shift_ms/1000, center=True)) # 1

    time2index(index2time(frame_index=3, window_size=window_size_ms/1000, window_shift=window_shift_ms/1000), window_size=window_size_ms/1000, window_shift=window_shift_ms/1000) # 3
    """
    if(center):
        time = frame_index * window_shift
    else:
        time = window_size / 2 + frame_index * window_shift
    return round(time, ndigits)

def time2index(time, window_size=0.025, window_shift=0.01, precision=0.00001, ndigits=7, center=True, verbose=False):
    """ Find the nearest frame index to the given time (time2nearestindex).

    Parameters
    ----------
    time: any given time in seconds
    window_size: the frame window length of each frame in seconds (default: 0.025)
    window_shift: the frame window shift between successive frames in seconds  (default: 0.01)
    precision: the round precision when cutting the float to interger (default: 0.00001)
    ndigits: The number of decimals to use when rounding the time (default 7)
    center (bool): True when origin is the frame center, False when origin is the frame start.
    verbose (bool): True return frame index, frame begin and end times; False return frame index

    Returns
    ------
    the index of the frame (starting from zero), alternatively the start second of frame window, and the end second of frame window


    Example
    -------
    window_size_ms=50
    window_shift_ms=12.5
    print(index2time(frame_index=1, window_size=window_size_ms/1000, window_shift=window_shift_ms/1000)) # 0.0375
    print(time2index(0.0376, window_size=window_size_ms/1000, window_shift=window_shift_ms/1000)) # 1
    print(index2time(frame_index=1, window_size=window_size_ms/1000, window_shift=window_shift_ms/1000, center=True)) # 0.125
    print(time2index(0.0124, window_size=window_size_ms/1000, window_shift=window_shift_ms/1000, center=True)) # 1

    time2index(index2time(frame_index=3, window_size=window_size_ms/1000, window_shift=window_shift_ms/1000), window_size=window_size_ms/1000, window_shift=window_shift_ms/1000) # 3

    Implemented by bin-wu in the year 2023
    """
    time=float(time)
    if(center):
        prev_index = int(time / window_shift)
    else:
        if (time < window_size / 2): return 0
        #  prev_index = int((time - window_size / 2) / window_shift)
        prev_index = int(round((time - window_size / 2), ndigits) / window_shift)

    prev_time = index2time(prev_index, window_size=window_size, window_shift=window_shift, center=center)
    next_time = index2time(prev_index + 1, window_size=window_size, window_shift=window_shift, center=center)
    average_time = (prev_time + next_time) / 2

    # 'floor' may have some precision issue, so make a precision bound.
    assert time - prev_time > -precision and next_time - time > -precision, f"{prev_index=}, {prev_time=}, {time=}, {next_time=}"

    index = prev_index if time -average_time < 0 else prev_index + 1
    begin_window_sec = round(index2time(index, window_size=window_size, window_shift=window_shift, center=center) - window_size/2, ndigits)
    end_window_sec = round(index2time(index, window_size=window_size, window_shift=window_shift, center=center) + window_size/2, ndigits)

    return (index, begin_window_sec, end_window_sec) if verbose else index

def audacitysegment2framelabel(audacity_segment_file, window_size=0.025, window_shift=0.01, num_frames=None, ndigits=7, center=True, precision=0.00001, default_label="noise", verbose=True):
    """Convert audacity segment file to frame labels (Assume that the segments have no overlap in time).

    Parameters
    ----------
    audacity_segment_file : the path the segment file of audacity or a list of Segment objects
    window_size : the window size of a frame (second)
    window_shift : the window shift of a frame (second)
    num_frames : the number of frame.
        if num_frame is None, assign it as the nearest frame to end time of the last segment
    ndigits : the number of decimals to use when rounding the time (default 7)
    center : True when origin is the frame center, False when origin is the frame start.
    precision : the round precision when cutting the float to interger (default: 0.00001)
    default_label: the default label that to be filled between the segments (the noise or silence label. default: 'noise')

    Returns
    -------
    A list of frame labels

    Examples
    --------
    frame_labels = audacitysegment2framelabel(seg_file, window_size=window_size_ms/1000, window_shift=window_shift_ms/1000, num_frames=None, ndigits=7, center=center, precision=0.00001)

    Notes
    --------
    Assume that the segments have no overlap in time.
    When segment intervals have overlaps, the later segments will overwrite the previous segments.
    The segments need not be sorted by their beginning times in the segment files.
    """
    if type(audacity_segment_file) == str:
        segments = read_audacity_segments(audacity_segment_file)
    else:
        segments = audacity_segment_file

    if not num_frames:
        end_second = max([segment.end_sec for segment in segments])
        last_frame =  time2index(end_second, window_size=window_size, window_shift=window_shift, precision=precision, ndigits=ndigits, center=center)
        num_frames = last_frame + 1

    frame_labels = [default_label] * num_frames # copy all labels as the default label (e.g., 'noise'), then assign frame labels according to the audacity segments
    for segment in segments:
        begin_frame = time2index(segment.begin_sec, window_size=window_size, window_shift=window_shift, precision=precision, ndigits=ndigits, center=center)
        end_frame = time2index(segment.end_sec, window_size=window_size, window_shift=window_shift, precision=precision, ndigits=ndigits, center=center)
        label = segment.label

        if (end_frame > (num_frames-1)) and (begin_frame <= (num_frames-1)) and verbose:
            print("Warning: The index of the end of segment {} with frame index {} is larger than (num_of_frames-1) {}. Set the index of the end of frame of the segment as the (num_of_frames-1) {}".format(segment, end_frame, (num_frames-1), num_frames-1))
            end_frame = num_frames -1
        elif (begin_frame > (num_frames-1)) and verbose:
            print("Warning: The beginning of segment {} with frame index {} is larger than (num_of_frames-1) {}. Skip the frame".format(segment, begin_frame, num_frames-1))
        else:
            pass

        if (begin_frame <= (num_frames-1)):
            assert begin_frame <= (end_frame + 1)

            # The current segment and a previous segment have overlaps, overwriting the previous one.
            if not all(frame_label == default_label for frame_label in frame_labels[begin_frame: end_frame+1]) and verbose:
                seg_file_name = audacity_segment_file if type(audacity_segment_file) == str else "segment_list"
                print(f"Warning: Segments from '{seg_file_name}' overlap. "
                f"frame_labels[{begin_frame}: {end_frame+1}]: '{frame_labels[begin_frame: end_frame+1]}' "
                f"overwritten by labels: '[{label}] * {end_frame+1-begin_frame}' from {segment.begin_sec} sec to {segment.end_sec} sec.")

            frame_labels[begin_frame: end_frame+1] = [label] * (end_frame+1-begin_frame) # copy label from the begin to the end frame including the end frame

    return frame_labels

from collections import namedtuple
Seg = namedtuple('Seg', ['begin_sec', 'end_sec', 'label', 'begin_frame', 'end_frame']) # A segment includes its end frame.
# s = Seg(begin_sec=0, end_sec=3.4, label='tr', begin_frame=0, end_frame=4)

def framelabel2audacitysegment(frame_label_list, window_size=0.025, window_shift=0.01, ndigits=7, center=True, precision=0.00001, frame_extension_mode='center'):
    """Convert a list of frame labels to a list of audacity segments

    Parameters
    ----------
    frame_label_list : a list of frame labels
    window_size : the window size of a frame (second)
    window_shift : the window shift of a frame (second)
    ndigits : the number of decimals to use when rounding the time (default 7)
    center : True when origin is the frame center, False when origin is the frame start. (default True)
    precision : the round precision when cutting the float to interger (default: 0.00001)
    frame_center_extension: extend segment's beginning time or end time to make different segments connected together. (default: "center")
        take a value from ['right', 'center', 'left', None]
        "right": [begin_frame_center_time, end_frame_center_time+window_shift]
        "center": [begin_frame_center_time-0.5*window_shift, end_frame_center_time+0.5*window_shift]
        "left": [begin_frame_center_time-window_shift, end_frame_center_time]
        None: [begin_frame_center_time, end_frame_center_time]

    Returns
    -------
    A list of audacity segments; each segment with attributes of ['begin_sec', 'end_sec', 'label', 'begin_frame', 'end_frame']
    A segment includes its end frame.

    Examples
    --------
    frame_label_list = ['tr', 'tr']
    center=True
    frame_extension_mode="center" # None # "center" # "left" # "right"
    # segs = framelabel2audacitysegment(frame_label_list, window_size=window_size_ms/1000, window_shift=window_shift_ms/1000, center=True, frame_extension_mode=frame_extension_mode)
    segs = framelabel2audacitysegment(frame_label_list, center=center, frame_extension_mode=frame_extension_mode)
    print("center:{}; frame_extension model: {}; frame_label_list: {}".format(center, frame_extension_mode, frame_label_list))
    print(segs)
    ###
    center:True; frame_extension model: center; frame_label_list: ['tr']
    [Seg(begin_sec=0.0, end_sec=0.005, label='tr', begin_frame=0, end_frame=0)]
    center:False; frame_extension model: center; frame_label_list: ['tr']
    [Seg(begin_sec=0.0075, end_sec=0.0175, label='tr', begin_frame=0, end_frame=0)]
    center:True; frame_extension model: center; frame_label_list: ['tr', 'tr']
    [Seg(begin_sec=0.0, end_sec=0.015, label='tr', begin_frame=0, end_frame=1)]
    center:False; frame_extension model: center; frame_label_list: ['tr', 'tr']
    [Seg(begin_sec=0.0075, end_sec=0.0275, label='tr', begin_frame=0, end_frame=1)]
    center:True; frame_extension model: center; frame_label_list: ['tr', 'ph']
    [Seg(begin_sec=0.0, end_sec=0.005, label='tr', begin_frame=0, end_frame=0), Seg(begin_sec=0.005, end_sec=0.015, label='ph', begin_frame=1, end_frame=1)]
    center:True; frame_extension model: center; frame_label_list: ['tr', 'ph', 'ph', 'tr', 'tr', 'tr', 'tr', 'pp', 'pp', 'noise']
    [Seg(begin_sec=0.0, end_sec=0.005, label='tr', begin_frame=0, end_frame=0), Seg(begin_sec=0.005, end_sec=0.025, label='ph', begin_frame=1, end_frame=2), Seg(begin_sec=0.025, end_sec=0.065, label='tr', begin_frame=3, end_frame=6), Seg(begin_sec=0.065, end_sec=0.085, label='pp', begin_frame=7, end_frame=8), Seg(begin_sec=0.085, end_sec=0.095, label='noise', begin_frame=9, end_frame=9)]

    center:True; frame_extension model: right; frame_label_list: ['tr']
    [Seg(begin_sec=0.0, end_sec=0.01, label='tr', begin_frame=0, end_frame=0)]
    center:False; frame_extension model: right; frame_label_list: ['tr']
    [Seg(begin_sec=0.0125, end_sec=0.0225, label='tr', begin_frame=0, end_frame=0)]
    center:True; frame_extension model: right; frame_label_list: ['tr', 'tr']
    [Seg(begin_sec=0.0, end_sec=0.02, label='tr', begin_frame=0, end_frame=1)]
    center:False; frame_extension model: right; frame_label_list: ['tr', 'tr']
    [Seg(begin_sec=0.0125, end_sec=0.0325, label='tr', begin_frame=0, end_frame=1)]
    center:True; frame_extension model: right; frame_label_list: ['tr', 'ph']
    [Seg(begin_sec=0.0, end_sec=0.01, label='tr', begin_frame=0, end_frame=0), Seg(begin_sec=0.01, end_sec=0.02, label='ph', begin_frame=1, end_frame=1)]
    center:True; frame_extension model: right; frame_label_list: ['tr', 'ph', 'ph', 'tr', 'tr', 'tr', 'tr', 'pp', 'pp', 'noise']
    [Seg(begin_sec=0.0, end_sec=0.01, label='tr', begin_frame=0, end_frame=0), Seg(begin_sec=0.01, end_sec=0.03, label='ph', begin_frame=1, end_frame=2), Seg(begin_sec=0.03, end_sec=0.07, label='tr', begin_frame=3, end_frame=6), Seg(begin_sec=0.07, end_sec=0.09, label='pp', begin_frame=7, end_frame=8), Seg(begin_sec=0.09, end_sec=0.1, label='noise', begin_frame=9, end_frame=9)]

    Implemented by bin-wu at 12:13 on 3 March. 2024.
    """
    # frame tracer
    begin_pointer = 0
    end_pointer = 0
    # frames of segments
    begin_frame = 0
    end_frame = 0

    num_frames = len(frame_label_list)
    assert num_frames, "Frame label list is empty" # not empty

    assert frame_extension_mode in [None, 'left', 'right', 'center']
    def extend_frame(begin_time, end_time, frame_extension_mode):
        if frame_extension_mode == 'left':
            begin_time = (begin_time - window_shift) if (begin_time - window_shift) > 0 else begin_time
        elif frame_extension_mode == 'right':
            end_time = end_time + window_shift
        elif frame_extension_mode == 'center':
            begin_time = (begin_time - 0.5 * window_shift) if (begin_time - 0.5 * window_shift) > 0 else begin_time
            end_time = end_time + 0.5 * window_shift
        else:
            pass
        return round(begin_time, ndigits), round(end_time, ndigits)

    segs = []
    while True:
        if begin_pointer == end_pointer: # finish a segment
            begin_frame = begin_pointer
            end_pointer += 1
        elif end_pointer == num_frames: # finish all segments
            end_frame = num_frames - 1

            begin_sec = index2time(begin_frame, window_size=window_size, window_shift=window_shift, ndigits=ndigits, center=center)
            end_sec = index2time(end_frame, window_size=window_size, window_shift=window_shift, ndigits=ndigits, center=center)
            begin_sec, end_sec = extend_frame(begin_sec, end_sec, frame_extension_mode)
            s = Seg(begin_sec=begin_sec, end_sec=end_sec, label=frame_label_list[begin_frame], begin_frame=begin_frame, end_frame=end_frame)
            segs.append(s)
            break
        elif frame_label_list[begin_pointer] == frame_label_list[end_pointer]: # inside a segment
            end_pointer += 1
        else: # boundary of a segment
            begin_pointer = end_pointer
            end_frame = end_pointer - 1

            begin_sec = index2time(begin_frame, window_size=window_size, window_shift=window_shift, ndigits=ndigits, center=center)
            end_sec = index2time(end_frame, window_size=window_size, window_shift=window_shift, ndigits=ndigits, center=center)
            begin_sec, end_sec = extend_frame(begin_sec, end_sec, frame_extension_mode)
            s = Seg(begin_sec=begin_sec, end_sec=end_sec, label=frame_label_list[begin_frame], begin_frame=begin_frame, end_frame=end_frame)
            segs.append(s)

    return segs

def overlapped_chunks(seq, chunk_size=5, chunk_shift=2, verbose=False):
    """Use a sliding window to convert a sequence (list) of objects into overlapped chunks (sub-lists)

    Parameters
    ----------
    seq : a list of object
    chunk_size : the chunk window size of a sliding window
    chunk_shift : the chunk window shift of a sliding window
    verbose : print chunk list (default False)

    Examples
    --------
    seq = "ab"
    seq = "abcdefg"
    seq = "abcdefgh"
    # seq = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
    overlapped_chunks(seq, chunk_size=5, chunk_shift=2, verbose=True)

    Outputs:
    seq: ab; chunks: ['ab']; chunk_size: 5; chunk_shift: 2
    seq: abcdefg; chunks: ['abcde', 'cdefg']; chunk_size: 5; chunk_shift: 2
    seq: abcdefgh; chunks: ['abcde', 'cdefg', 'efgh']; chunk_size: 5; chunk_shift: 2
    seq: [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]; chunks: [[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], [[5, 6], [7, 8], [9, 10], [11, 12]]]; chunk_size: 5; chunk_shift: 2

    Implmented by bin-wu at 12:48 on 16 March 2024
    """

    num_chunks = ceil(float(len(seq) - chunk_size) / chunk_shift) + 1 if (len(seq) - chunk_size) >= 0 else 1

    chunks = []
    for i in range(0, num_chunks):
        start_index = i * chunk_shift
        stop_index = min(start_index+chunk_size, len(seq)) # not included
        chunks.append(seq[start_index:stop_index])

    if verbose:
        print("seq: {}; chunks: {}; chunk_size: {}; chunk_shift: {}".format(seq, chunks, chunk_size, chunk_shift))

    return chunks

def processing(wav1, wav2=None, seg1=None, seg2=None, sampling_rate=48000, frame_size_sec=50/1000, frame_shift_sec=12.5/1000, feat_dim=80, min_freq=None, feat_type="mel", chunk_size_sec=0.5, chunk_shift_sec=0.4, verbose=True):
    """Convert long wavs and its segment files into chunks of features and labels using the sliding window.

    Parameters
    ----------
    wav1 : the first wav file
    wav2 : the second wav file
    seg1 : the first audacity segment file (each line: begin_sec end_sec label)
    seg2 : the second audacity segment file
    sampling_rate : the sampling rate of the audio
    frame_size_sec : the frame size (second)
    frame_shift_sec : the frame shift
    feat_dim : the feature dimension
    min_freq : the minimum frequency of the feature
    feat_type : the feature type (default "mel")
    chunk_size_sec : the chunk size (a chunk consists of frames)
    chunk_shift_sec : the chunk shift

    Returns
    --------
    feat_chunks1, feat_chunks2, feat_chunks12, label_chunks1, label_chunks2, label_chunks12
    Lists of chunks or merged chunks, each element with size of (chunk_size_num_frames, feat_dim) for feature-chunks or (chunk_size_num_frames) for [merged] label chunks
    or (chunk_size_num_frames*2, feat_dim) for the merged feature-chunks (feat_chunks12) with merge order from the first chunks to the second ones.
    Merged chunk of feat_chunks12 is concatenated by the first dimension feat_chunks1 and feat_chunks2 (a zero matrix when wav2 is empty).
    Merged label of label_chunks12 would keep the first segment labels same and modify the second segment labels by adding '2':
    # e.g. a 'tr' of the second segment labels becomes 'tr2'.
    The second segment labels would overwrite the first segment labels if overlaps exist.

    Examples
    --------
    sampling_rate = 44100
    frame_size_sec = 50 / 1000
    frame_shift_sec = 12.5 / 1000
    num_mels = 80
    min_freq = 3000
    train_chunk_size_sec = 500 / 1000 # 'train_chunk_size_num_frames': 41
    train_chunk_shift_sec = 150 / 1000 # 'train_chunk_shift_num_frames': 13
    wav1 = '/data/share/bin-wu/data/marmoset/vocalization/marmoset_mit/data/pair10/pair10_animal1_together.wav'
    wav2 = '/data/share/bin-wu/data/marmoset/vocalization/marmoset_mit/data/pair10/pair10_animal2_together.wav'
    seg1 = '/data/share/bin-wu/data/marmoset/vocalization/marmoset_mit/processed/audacity/audacity_labels/p10a1_toget.txt'
    seg2 = '/data/share/bin-wu/data/marmoset/vocalization/marmoset_mit/processed/audacity/audacity_labels/p10a2_toget.txt'

    # feat_chunks1, _, feat_chunks12, _, _, _ = processing(wav1=wav1, wav2=None, seg1=None, seg2=None, sampling_rate=sampling_rate, frame_size_sec=frame_size_sec, frame_shift_sec=frame_shift_sec, feat_dim=num_mels, min_freq=min_freq, feat_type="mel", chunk_size_sec=train_chunk_size_sec, chunk_shift_sec=train_chunk_shift_sec)
    # feat_chunks1, feat_chunks2, feat_chunks12, _, _, _ = processing(wav1=wav1, wav2=wav2, seg1=None, seg2=None, sampling_rate=sampling_rate, frame_size_sec=frame_size_sec, frame_shift_sec=frame_shift_sec, feat_dim=num_mels, min_freq=min_freq, feat_type="mel", chunk_size_sec=train_chunk_size_sec, chunk_shift_sec=train_chunk_shift_sec)
    # feat_chunks1, feat_chunks2, feat_chunks12, label_chunks1, _, _ = processing(wav1=wav1, wav2=wav2, seg1=seg1, seg2=None, sampling_rate=sampling_rate, frame_size_sec=frame_size_sec, frame_shift_sec=frame_shift_sec, feat_dim=num_mels, min_freq=min_freq, feat_type="mel", chunk_size_sec=train_chunk_size_sec, chunk_shift_sec=train_chunk_shift_sec)
    feat_chunks1, feat_chunks2, feat_chunks12, label_chunks1, label_chunks2, label_chunks12 = processing(wav1=wav1, wav2=wav2, seg1=seg1, seg2=seg2, sampling_rate=sampling_rate, frame_size_sec=frame_size_sec, frame_shift_sec=frame_shift_sec, feat_dim=num_mels, min_freq=min_freq, feat_type="mel", chunk_size_sec=train_chunk_size_sec, chunk_shift_sec=train_chunk_shift_sec)
    print(f"{len(feat_chunks1)=} {feat_chunks1[0].shape=}")
    print(f"{len(feat_chunks2)=} {feat_chunks2[0].shape=}")
    print(f"{len(feat_chunks12)=} {feat_chunks12[0].shape=}")
    print(f"{len(label_chunks1)=} {len(label_chunks1[0])=}")
    print(f"{len(label_chunks2)=} {len(label_chunks2[0])=}")
    print(f"{len(label_chunks12)=} {len(label_chunks12[0])=}")
    # with open('label.txt', 'w') as f:
    #     for line in label_chunks12:
    #         f.write(f"{line}\n")

    Outputs:
    len(feat_chunks1)=53859 feat_chunks1[0].shape=torch.Size([41, 80])
    len(feat_chunks2)=53859 feat_chunks2[0].shape=torch.Size([41, 80])
    len(feat_chunks12)=53859 feat_chunks12[0].shape=torch.Size([82, 80])
    len(label_chunks1)=53859 len(label_chunks1[0])=41
    len(label_chunks2)=53859 len(label_chunks2[0])=41
    len(label_chunks12)=53859 len(label_chunks12[0])=41

    Implemented by bin-wu at 16:39 on 17 March 2024
    """
    feat_chunks1 = []
    feat_chunks2 = []
    feat_chunks12 = []
    label_chunks1 = []
    label_chunks2 = []
    label_chunks12 = []

    frame_size_num_samples = int(frame_size_sec * sampling_rate)
    frame_shift_num_samples = int(frame_shift_sec * sampling_rate)
    n_fft = frame_size_num_samples

    assert feat_type == 'mel'

    if (wav1):
        audio_file = wav1
        waveform, sr = torchaudio.load(audio_file)
        assert sampling_rate == sr, f"sampling_rate of config file: '{sampling_rate}', different from sampling_rate of ({audio_file}): '{sr}'"
        transform = torchaudio.transforms.MelSpectrogram(sample_rate=sampling_rate,
                                                         n_fft=n_fft,
                                                         win_length=frame_size_num_samples,
                                                         hop_length=frame_shift_num_samples,
                                                         f_min=min_freq,
                                                         n_mels=feat_dim,
                                                         center=True)
        logmel = transform(waveform).log()[0].T  # (num_frames, feature_size)

        chunk_size_num_frames = time2index(chunk_size_sec, frame_size_sec, frame_shift_sec) + 1
        chunk_shift_num_frames = time2index(chunk_shift_sec, frame_size_sec, frame_shift_sec) + 1
        feat_chunks = overlapped_chunks(logmel, chunk_size=chunk_size_num_frames, chunk_shift=chunk_shift_num_frames)
        logmel1 = logmel
        num_frames = logmel1.shape[0]
        feat_chunks1 = feat_chunks

    if (wav2):
        assert wav1, f"When wav2 ({wav2}) exists, wav1 should exist."
        audio_file = wav2
        waveform, sr = torchaudio.load(audio_file)
        assert sampling_rate == sr, f"sampling_rate of config file: '{sampling_rate}', different from sampling_rate of ({audio_file}): '{sr}'"
        transform = torchaudio.transforms.MelSpectrogram(sample_rate=sampling_rate,
                                                         n_fft=n_fft,
                                                         win_length=frame_size_num_samples,
                                                         hop_length=frame_shift_num_samples,
                                                         f_min=min_freq,
                                                         n_mels=feat_dim,
                                                         center=True)
        logmel = transform(waveform).log()[0].T  # (num_frames, feature_size)

        chunk_size_num_frames = time2index(chunk_size_sec, frame_size_sec, frame_shift_sec) + 1
        chunk_shift_num_frames = time2index(chunk_shift_sec, frame_size_sec, frame_shift_sec) + 1
        feat_chunks = overlapped_chunks(logmel, chunk_size=chunk_size_num_frames, chunk_shift=chunk_shift_num_frames)
        logmel2 = logmel
        if (logmel1.shape[0] != logmel2.shape[0]) and verbose: print(f"Warning: logmel1 of '{wav1}' and logmel2 of '{wav2}' has different different shapes of {logmel1.shape} and {logmel2.shape}")
        num_frames = min(logmel1.shape[0], logmel2.shape[0])
        feat_chunks2 = feat_chunks

    # Merge two chunks
    if (not wav2): feat_chunks2 = [feat_chunks1[i].new_zeros(feat_chunks1[i].shape) for i in range(0, len(feat_chunks1))] # padding the second chunks as zeros when the second wav is None
    if len(feat_chunks1) != len(feat_chunks2) and verbose:
        print(f"Warning: different sizes of feature chunks for wav1: '{wav1}' has num_chunks of {len(feat_chunks1)} and wav2 '{wav2}' has num_chunks of {len(feat_chunks2)}.")
    feat_chunks12 = [torch.cat([feat_chunks1[i], feat_chunks2[i]], dim=0) for i in range(0, min(len(feat_chunks1), len(feat_chunks2)))]

    if (seg1):
        assert wav1, f"When seg1 ({seg1}) exists, wav1 should exist."
        seg_file = seg1
        frame_labels = audacitysegment2framelabel(seg_file, frame_size_sec, frame_shift_sec, num_frames=num_frames, verbose=verbose)
        label_chunks = overlapped_chunks(frame_labels, chunk_size=chunk_size_num_frames, chunk_shift=chunk_shift_num_frames)
        label_chunks1 = label_chunks

    if (seg2):
        assert wav1 and wav2 and seg1, f"When seg2 ({seg2}) exists, wav1, wav2, and seg1 should exist."
        seg_file = seg2
        frame_labels = audacitysegment2framelabel(seg_file, frame_size_sec, frame_shift_sec, num_frames=num_frames, verbose=verbose)
        label_chunks = overlapped_chunks(frame_labels, chunk_size=chunk_size_num_frames, chunk_shift=chunk_shift_num_frames)
        label_chunks2 = label_chunks

    if (seg1 and seg2):
        # Merge two labels, the second labels overwrites the first ones when overlapping.
        segments1 = read_audacity_segments(seg1)
        segments2 = read_audacity_segments(seg2)
        for i, segment in enumerate(segments2):
              segments2[i].label = segment.label + "2" # e.g. a 'tr' of the second segment labels becomes 'tr2'
        frame_labels = audacitysegment2framelabel(segments1+segments2, frame_size_sec, frame_shift_sec, num_frames=num_frames, verbose=verbose) # concatenate two segment lists
        label_chunks = overlapped_chunks(frame_labels, chunk_size=chunk_size_num_frames, chunk_shift=chunk_shift_num_frames)
        label_chunks12 = label_chunks

    return feat_chunks1, feat_chunks2, feat_chunks12, label_chunks1, label_chunks2, label_chunks12

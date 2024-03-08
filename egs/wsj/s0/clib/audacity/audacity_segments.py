import re

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

def index2time(frame_index, window_size=0.025, window_shift=0.01, ndigits=7, center=False):
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

def time2index(time, precision=0.00001, window_size=0.025, window_shift=0.01, ndigits=7, center=False, verbose=False):
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

def audacitysegment2framelabel(audacity_segment_file, window_size=0.025, window_shift=0.01, num_frames=None, ndigits=7, center=True, precision=0.00001):
    """Convert audacity segment file to frame labels (Assume that the segments have no overlap in time).

    Parameters
    ----------
    audacity_segment_file : the path the segment file of audacity
    window_size : the window size of a frame (second)
    window_shift : the window shift of a frame (second)
    num_frames : the number of frame.
        if num_frame is None, assign it as the nearest frame to end time of the last segment 
    ndigits : the number of decimals to use when rounding the time (default 7)
    center : True when origin is the frame center, False when origin is the frame start.
    precision : the round precision when cutting the float to interger (default: 0.00001)

    Returns
    -------
    A list of frame labels
    
    Examples
    --------
    frame_labels = audacitysegment2framelabel(seg_file, window_size=window_size_ms/1000, window_shift=window_shift_ms/1000, num_frames=None, ndigits=7, center=center, precision=0.00001)
    """
    segments = read_audacity_segments(seg_file)

    if not num_frames:
        last_frame =  time2index(segments[-1].end_sec, precision=precision, window_size=window_size, window_shift=window_shift, ndigits=ndigits, center=center)
        num_frames = last_frame

    frame_labels = ["noise"] * num_frames # copy all labels as 'noise', then assign frame labels according to the audacity segments    
    for segment in segments:
        begin_frame = time2index(segment.begin_sec, precision=precision, window_size=window_size, window_shift=window_shift, ndigits=ndigits, center=center)
        end_frame = time2index(segment.end_sec, precision=precision, window_size=window_size, window_shift=window_shift, ndigits=ndigits, center=center)
        label = segment.label

        if (end_frame > (num_frames-1)) and (begin_frame <= (num_frames-1)) :
            print("Warning: The index of the end of segment {} with frame index {} is larger than (num_of_frames-1) {}. Set the index of the end of frame of the segment as the (num_of_frames-1) {}".format(segment, end_frame, (num_frames-1), num_frames-1))
            end_frame = num_frames -1 
        elif (begin_frame > (num_frames-1)):
            print("Warning: The beginning of segment {} with frame index {} is larger than (num_of_frames-1) {}. Skip the frame".format(segment, begin_frame, num_frames-1))
        else:
            pass
        
        if (begin_frame <= (num_frames-1)):
            assert begin_frame <= (end_frame + 1)
            frame_labels[begin_frame: end_frame+1] = [label] * (end_frame+1-begin_frame) # copy label from the begin to the end frame including the end frame

    return frame_labels

from collections import namedtuple
Seg = namedtuple('Seg', ['begin_sec', 'end_sec', 'label', 'begin_frame', 'end_frame']) # A segment includes its end frame.
# s = Seg(begin_sec=0, end_sec=3.4, label='tr', begin_frame=0, end_frame=4)

def framelabel2audacitysegment(frame_label_list, window_size=0.025, window_shift=0.01, ndigits=7, center=True, precision=0.00001, frame_extension_mode='right'):
    """Convert a list of frame labels to a list of audacity segments

    Parameters
    ----------
    frame_label_list : a list of frame labels
    window_size : the window size of a frame (second)
    window_shift : the window shift of a frame (second)
    ndigits : the number of decimals to use when rounding the time (default 7)
    center : True when origin is the frame center, False when origin is the frame start. (default True)
    precision : the round precision when cutting the float to interger (default: 0.00001)
    frame_center_extension: extend segment's beginning time or end time to make different segments connected together. (default: "right")
        takea a value from ['right', 'center', 'left', None]
        "right": [begin_frame_center_time, end_frame_center_time+window_shift]
        "center": [begin_frame_center_time-0.5wndow_shift, end_frame_center_time+0.5window_shift]
        "left": [begin_frame_center_time-wndow_shift, end_frame_center_time]
        None: [begin_frame_center_time, end_frame_center_time]

    Returns
    -------
    A list of audacity segments; each segment with attributes of ['begin_sec', 'end_sec', 'label', 'begin_frame', 'end_frame']
    A segment includes its end frame.
    
    Examples
    --------
    frame_label_list = ['tr', 'tr']
    center=True
    frame_extension_mode="right" # None # "center" # "left" # "right"
    # segs = framelabel2audacitysegment(frame_label_list, window_size=window_size_ms/1000, window_shift=window_shift_ms/1000, center=True, frame_extension_mode=frame_extension_mode)
    segs = framelabel2audacitysegment(frame_label_list, center=center, frame_extension_mode=frame_extension_mode)
    print("center:{}; frame_extension model: {}; frame_label_list: {}".format(center, frame_extension_mode, frame_label_list))
    print(segs)
    ###
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

    Implemneted by bin-wu at 12:13 on 3 Aug. 2024.
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

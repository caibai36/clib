import os
import argparse
import numpy as np
import soundfile

parser = argparse.ArgumentParser(description=f"Get the avd segment files for each utterance (<uttid> <start_sec> <end_sec>). ")
parser.add_argument("--wav_scp", type=str, default="data/train/wav.scp", help="the wav scp file")
parser.add_argument("--segment", type=str, default="segment", help="the segment file after vad (voice activity dectection)")
parser.add_argument("--min_amplitude", type=float, default=0.1,
                    help="Getting segment file by emoving start silence and end slience whose absolute value of amplitudes are less than the given minimum amplitude")
parser.add_argument("--num_samples_gt_min", type=int, default=100,
                    help="When there are less than N samples greater than min_amp, then decrease amplitude to include more samples")
args = parser.parse_args()

def sample2time(s, sample_rate=16000):
    t = s / float(sample_rate)
    return t
8
def time2sample(t, sample_rate=16000):
    s = t * sample_rate
    return int(s)

def segment_begin_end_sec(wav, min_amp=0.1, num_samples_gt_min=100, begin_sec=None, end_sec=None, sample_rate=16000):
    """ find the start time and the end time of a segment when its amplitudes are greater than the given minimum amplitude """
    begin = 0 if begin_sec is None else time2sample(begin_sec, sample_rate)
    end = len(wav) if end_sec is None else time2sample(end_sec, sample_rate)
    segment = wav[begin:end]

    # indices = np.where(abs(segment) > min_amp)
    indices = (np.array([]),) # when the sound with too small amplitude, decrease the min_amp. In:(np.array([]),)[0].shape[0] => Out: 0
    while indices[0].shape[0] < num_samples_gt_min: # when there are less than 30 samples more than min_amp, then decrease amplitude
        print(f"Warning: when there are less than {indices[0].shape[0]} samples whose amplitudes are more than min_amp, we decrease minimum amplitude for {min_amp} to {min_amp/2}.")
        indices = np.where(abs(segment) > min_amp)
        print(f"Warning: After we decrease minimum amplitude for {min_amp} to {min_amp/2}, we have {indices[0].shape[0]} samples whose amplitudes are more than min_amp.")
        min_amp /= 2
                        
    return begin + sample2time(np.min(indices), sample_rate), begin + sample2time(np.max(indices), sample_rate)

with open(args.wav_scp, 'r') as f_wav_scp, \
     open(args.segment, 'w') as f_seg:
    for line in f_wav_scp:
        fields = line.strip().split()
        # uttid /project/nakamura-lab08/Work/bin-wu/share/tools/kaldi/tools/sph2pipe_v2.5/sph2pipe -f wav audio.sph |
        # uttid /home/dpovey/kaldi-tombstone/egs/swbd/s5c/../../../tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 1 audio.sph |
        # uttid audio.wav
        uttid = fields[0]
        wav_path = fields[-2] if "sph2pipe" in line else fields[1] 
        wav, sr = soundfile.read(wav_path)
        start_sec, end_sec = segment_begin_end_sec(wav, min_amp=args.min_amplitude, num_samples_gt_min=args.num_samples_gt_min, sample_rate=sr)
        f_seg.write(f"{uttid} {uttid} {start_sec:.3f} {end_sec:.3f}\n")

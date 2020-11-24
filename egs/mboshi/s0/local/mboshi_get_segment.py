import os
import argparse
import numpy as np
import soundfile

parser = argparse.ArgumentParser(description=f"Get the avd segment files for each utterance of mboshi (<uttid> <start_sec> <end_sec>). "
                                 f"align_vad.txt comes from the alignment file. "
                                 f"align_rmsil_vad.txt comes align_vad.txt with the start and the end silence removed.") 
parser.add_argument("--corpus", type=str, default="/project/nakamura-lab08/Work/bin-wu/share/data/mboshi/full_corpus_newsplit",
                    help="the corpus directory of mboshi, e.g., data/mboshi/full_corpus_newsplit")
parser.add_argument("--result", type=str, default="data/segment", help="the result directory will contain the align_vad.txt and align_rmsil_vad.txt")
parser.add_argument("--min_amplitude", type=float, default=0.1,
                    help="Getting the algin_rmsil_vad.txt from audio by removing the start silence and the end slience whose amplitudes are less than the given minimum amplitude")
args = parser.parse_args()

# (base) [bin-wu@ahccsclm03 s0]$  head /project/nakamura-lab08/Work/bin-wu/share/data/mboshi/ZRC_scoring/mboshi/alignment_mb/alignment_word.txt
# abiayi_2015-09-08-11-18-39_samsung-SM-T530_mdw_elicit_Dico18_1 0.216 3.446 kyema
# abiayi_2015-09-08-11-18-39_samsung-SM-T530_mdw_elicit_Dico18_1 3.536 4.376 yeekirá
word_align_file = args.corpus + "/../ZRC_scoring/mboshi/alignment_mb/alignment_word.txt"
assert os.path.exists(word_align_file), f"the word alignment file '{word_align_file}' is not found..."

if not os.path.exists(args.result):
    os.makedirs(args.result)
align_vad = os.path.join(args.result, "align_vad.txt")
align_rmsil_vad = os.path.join(args.result, "align_rmsil_vad.txt")

files = os.listdir(args.corpus + "/all") # file in format uttid.wav
uttid_set = {os.path.splitext(file)[0] for file in files if file.endswith('wav')} # all valid uttids

print("Getting the algin_vad.txt from the alignment file...")
uttid2times = {}
uttids = []
with open(word_align_file, encoding="utf8") as f_w:
    for line in f_w:
        line = line.strip()
        uttid, start_sec, end_sec, word = line.split()
        if uttid in uttid_set and uttid not in uttids:
            uttids.append(uttid)
        if uttid not in uttid2times: uttid2times[uttid] = set()
        uttid2times[uttid].add(float(start_sec))
        uttid2times[uttid].add(float(end_sec))

# get the start_sec and end_sec of the segment of the each utterance.
segment_starts = {} # key: uttid, vaule: the start time (second) of the segment
segment_ends = {} # key: uttid, vaule: the end time (second) of the segment
for uttid in uttids:
    segment_starts[uttid] = min(uttid2times[uttid])
    segment_ends[uttid] = max(uttid2times[uttid])
    
with open(align_vad, 'w') as f_align_vad:
    for uttid in uttids:
        f_align_vad.write(f"{uttid} {segment_starts[uttid]} {segment_ends[uttid]}\n")

print(f"Getting the algin_rmsil_vad.txt from audio by removing the start silence and end slience whose amplitudes are less than {args.min_amplitude}...")
def sample2time(s, sample_rate=16000):
    t = s / float(sample_rate)
    return t

def time2sample(t, sample_rate=16000):
    s = t * sample_rate
    return int(s)

def segment_begin_end_sec(wav, min_amp=0.1, begin_sec=None, end_sec=None, sample_rate=16000):
    """ find the start time and the end time of a segment when its amplitudes are greater than the given minimum amplitude """
    begin = 0 if begin_sec is None else time2sample(begin_sec, sample_rate)
    end = len(wav) if end_sec is None else time2sample(end_sec, sample_rate)
    segment = wav[begin:end]
    indices = np.where(abs(segment) > min_amp)
    return begin_sec + sample2time(np.min(indices), sample_rate), begin_sec + sample2time(np.max(indices), sample_rate)

with open(align_rmsil_vad, 'w') as f_align_rmsil_vad:
    for uttid in uttids:
        wav, sr = soundfile.read(os.path.join(args.corpus, "all", uttid + ".wav"))
        begin_sec = segment_starts[uttid]
        end_sec = segment_ends[uttid]
        start_sec, end_sec = segment_begin_end_sec(wav, min_amp=args.min_amplitude, begin_sec=begin_sec, end_sec=end_sec, sample_rate=sr)
        f_align_rmsil_vad.write(f"{uttid} {start_sec:.3f} {end_sec:.3f}\n")

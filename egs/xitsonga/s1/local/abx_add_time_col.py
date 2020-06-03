import os
import argparse

parser = argparse.ArgumentParser(description="Add time column to the abx embedding files",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--frame_length_ms", type=float, default=25, help="frame length (ms)")
parser.add_argument("--frame_shift_ms", type=float, default=10, help="frame_shift (ms)")
parser.add_argument("--file", type=str, default="tmp/001m_0007.txt",
                    help=f"an abx embedding file; we will read and write to the same file; the file name is its utterance id")
parser.add_argument("--segment_file", type=str, default=None,
                    help="the kaldi style segment file (<segment-id> <recording-id> <start-time> <end-time>)")

args = parser.parse_args()

length = args.frame_length_ms / 1000
shift = args.frame_shift_ms / 1000

uttid = os.path.splitext(os.path.basename(args.file))[0]

start_times = {}
if args.segment_file is not None:
    with open(args.segment_file) as f:
        for line in f:
            line = line.strip()
            cur_uttid, _, start_time, _ = line.split()
            start_times[cur_uttid] = float(start_time)

if args.segment_file is not None:
    t = start_times[uttid] + length / 2 # start at the center of a frame
else:
    t = length / 2

converted = []
with open(args.file) as f:
    for line in f:
        line = line.strip()
        converted.append(f"{t:.4f} {line}")
        t += shift

with open(args.file, 'w') as f:
    for line in converted:
        f.write(line + "\n")

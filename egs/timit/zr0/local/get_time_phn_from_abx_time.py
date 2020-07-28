import argparse
import os

parser = argparse.ArgumentParser(description="Get the annotated phonemes from given abx time")
parser.add_argument("--file_abx", type=str, default="eval/abx/embedding/exp/feat/mfcc39/FADG0_SI1279.txt", help="abx file with first column as time")
parser.add_argument("--file_dur_phn", type=str, default="data/test_time/test_dur_phn/FADG0_SI1279.PHN", help="file with first column start time (sec), sec col. end time (sec) and third col. the phoneme")
parser.add_argument("--result_dir", type=str, default="tmp", help="the result directory to store the $result_dir/uttid.PHN")
args = parser.parse_args()

assert os.path.splitext(os.path.basename(args.file_abx))[0] == os.path.splitext(os.path.basename(args.file_dur_phn))[0]
uttid = os.path.splitext(os.path.basename(args.file_abx))[0]

start_end_phns = []
with open(args.file_dur_phn) as f:
    for line in f:
        line = line.strip()
        start_end_phns.append(line.split())

abx_times = []
with open(args.file_abx) as f:
    for line in f:
        line = line.strip()
        abx_times.append(line.split()[0])

if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)

with open(os.path.join(args.result_dir, uttid + ".PHN"), 'w') as f:
    for time in abx_times:
        time = float(time)
        find_abx_time = False
        for start_end_phn in start_end_phns:
            start = float(start_end_phn[0])
            end = float(start_end_phn[1])
            phn = start_end_phn[2]
            if time >= start and time <= end:
                f.write(f"{time} {phn}\n")
                find_abx_time = True
                break
        if not find_abx_time:
            print(f"Warnning: abx_times {time} in the file '${args.file_abx}' not in any durations of the file ${args.file_dur_phn}, assume it is a sil")
            f.write(f"{time} sil\n")

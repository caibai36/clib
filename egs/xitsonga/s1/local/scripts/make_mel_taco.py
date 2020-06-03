import sys
import os
import json
import errno
import re
import logging
import pprint
logging.basicConfig(level=logging.INFO, stream=sys.stdout,format="[ %(asctime)s | %(filename)s | %(levelname)s ] %(message)s", datefmt="%d/%m/%Y %H:%M:%S") 

if not 'KALDI_ROOT' in os.environ:
    os.environ['KALDI_ROOT'] = '/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi'

if 'clib' in os.listdir(os.getcwd()):
    sys.path.append(os.getcwd()) # clib at the current working directory
elif "CLIB" in os.environ:
    sys.path.append(os.environ['CLIB']) # or clib at $CLIB
else:
    logging.error("Please give the path of the clib (e.g., export CLIB=$clib_path), or put the clib in the current directory")
    sys.exit()

import argparse
import kaldi_io
import subprocess
from clib.tacotron.audio import TacotronAudio

def execute_command(command):
    """ Runs a kaldi job in the foreground and waits for it to complete; raises an
        exception if its return status is nonzero.  The command is executed in
        'shell' mode so 'command' can involve things like pipes.  Often,
        'command' will start with 'run.pl' or 'queue.pl'.  The stdout and stderr
        are merged with the calling process's stdout and stderr so they will
        appear on the screen.
        See also: get_command_stdout, background_command
    """
    p = subprocess.Popen(command, shell=True)
    p.communicate()
    if p.returncode != 0:
        raise Exception("Command exited with status {0}: {1}".format(
                p.returncode, command))

parser = argparse.ArgumentParser(description="Extract the mel spectrum feature from the $data-dir/wav.scp (supporting $data_dir/segments)")
parser.add_argument("--feat_config", type=str, default="conf/feat/taco_mel_f80.json", help="the configuration of feature")
parser.add_argument("--data_dir", type=str, default="data/test3utts_mel802/", help="the data directory from kaldi data preparation, including wav.scp")
parser.add_argument("--feat_dir", type=str, default="feat/test3utts_mel802/", help="the feature directory with kaldi type compressed features")
parser.add_argument("--write_utt2num_frames", type=str, default="true", help="If true, write utt2num_frames file")
args = parser.parse_args()

with open(args.feat_config) as f_config:
    config = json.load(f_config)
logging.info(f"feature configuration:\n{pprint.pformat(config)}")
taudio = TacotronAudio(config)

segment_file = os.path.join(args.data_dir, "segments")
if not os.path.exists(segment_file):
    logging.info("segment file not found...")
    uttid_list = []
    feat_list = []
    wav_file = os.path.join(args.data_dir, "wav.scp")
    with open(wav_file) as f_wav:
        for line in f_wav:
            line = line.strip()
            if "|" in line: 
                # deal with the extended kaldi format with pipeline, e.g., "4k0c0301 sph2pipe -f wav 4k0c0301.wv1 |"
                # 4k0c0301 /project/nakamura-lab08/Work/bin-wu/share/tools/kaldi/tools/sph2pipe_v2.5/sph2pipe -f wav /project/nakamura-lab01/Share/Corpora/Speech/en/WSJ//wsj1/si_dt_20/4k0/4k0c0301.wv1 |
                uttid = re.sub("^(\S+)\s+(.*)", "\\1", line)
                uttid_list.append(uttid)
                wav_pipeline = re.sub("^(\S+)\s+(.*)", "\\2", line)
                wav_path = f"/tmp/{uttid}.wav"
                cmd = wav_pipeline + f" cat > {wav_path}"
                execute_command(cmd)
                wav = taudio.load_wav(wav_path)
                os.remove(wav_path)
            else:
                uttid, wav_path = re.split("\s+", line)
                uttid_list.append(uttid)
                wav = taudio.load_wav(wav_path)
            feat= taudio.melspectrogram(wav).T # shape (num_frames, num_mels)
            feat_list.append(feat)
else:
    # kaldi segment file format "<segment-id> <recording-id> <start-time-second> <end-time-second>\n"
    # segment-id is the key (uttid) in feats.scp
    # recording-id is the key of wav.scp
    logging.info(f"find segment file at: {segment_file}")

    recid2wav = {}
    wav_file = os.path.join(args.data_dir, "wav.scp")
    with open(wav_file) as f_wav:
        for line in f_wav:
            line = line.strip()
            if "|" in line: 
                # deal with the extended kaldi format with pipeline, e.g., "4k0c0301 sph2pipe -f wav 4k0c0301.wv1 |"
                # 4k0c0301 /project/nakamura-lab08/Work/bin-wu/share/tools/kaldi/tools/sph2pipe_v2.5/sph2pipe -f wav /project/nakamura-lab01/Share/Corpora/Speech/en/WSJ//wsj1/si_dt_20/4k0/4k0c0301.wv1 |
                recid = re.sub("^(\S+)\s+(.*)", "\\1", line)
                wav_pipeline = re.sub("^(\S+)\s+(.*)", "\\2", line)
                wav_path = f"/tmp/{recid}.wav"
                cmd = wav_pipeline + f" cat > {wav_path}"
                execute_command(cmd)
                recid2wav[recid] = wav_path
            else:
                recid, wav_path = re.split("\s+", line)
                recid2wav[recid] = wav_path

    uttid_list = []
    feat_list = []
    with open(segment_file) as f_seg:
        for line in f_seg:
            line = line.strip()
            # s0101a_100 s0101a 306.134 307.472
            uttid, recid, begin_sec, end_sec = re.split("\s+", line)
            wav_path = recid2wav[recid]
            uttid_list.append(uttid)
            wav = taudio.load_wav(wav_path, float(begin_sec), float(end_sec))
            feat= taudio.melspectrogram(wav).T # shape (num_frames, num_mels)
            feat_list.append(feat)

    # remove wav files in /tmp directory
    with open(wav_file) as f_wav:
        for line in f_wav:
            line = line.strip()
            if "|" in line:
                recid = re.sub("^(\S+)\s+(.*)", "\\1", line)
                wav_path = f"/tmp/{recid}.wav"
                if os.path.exists(wav_path):
                    os.remove(wav_path)

if not os.path.exists(args.feat_dir):
    os.makedirs(args.feat_dir)

scp_file = os.path.join(args.data_dir, "feats.scp")
ark_file = os.path.join(args.feat_dir, "raw_feats.ark")
ark_scp_output=f'ark:| copy-feats --compress=false ark:- ark,scp:{ark_file},{scp_file}'
with kaldi_io.open_or_fd(ark_scp_output,'wb') as f:
    for key, mat in zip(*(uttid_list, feat_list)):
            kaldi_io.write_mat(f, mat, key=key)

utt2num_frames_file = os.path.join(args.data_dir, "utt2num_frames")
if args.write_utt2num_frames == "true":
    with open(utt2num_frames_file, "w") as f_w:
        for key, mat in zip(*(uttid_list, feat_list)):
            f_w.write(f"{key} {len(mat)}\n")

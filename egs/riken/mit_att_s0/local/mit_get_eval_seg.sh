import os
import sys
sys.path.append(os.getcwd())

import json
import re
from clib.audacity import framelabel2audacitysegment

import argparse

parser = argparse.ArgumentParser(description="Get mit eval seg")
parser.add_argument("--chunk_json_file", type=str, default="data/mit_data/chunk_info/traincsize500cshift150_devcsize500cshift400_testcsize500cshift400/test_chunk.json", help="the test chunk json file")
parser.add_argument("--labels_file", type=str, default="exp/att/mit_data/mit0/EncRNNDecRNNAtt-enc3_bi256_ds0_drop-dec1_h512_do0.25-att_mlp-run0/mit0_two_stream_asr_mel_batchsize128_cutoff2000_labelsmoothing0.05_lr0.001_gradclip5_factor0.5_patience10_chunksizeshifttr500tr150dev500dev400test500test400_labeldownsample1/eval/beamsize10/hypo_char.txt", help="the hypothesis label file")
parser.add_argument("--pad_token", type=str, default="pad", help="the pad label")
parser.add_argument("--label_downsampling", type=int, default=1, help="the downsampling of the label sequences.")

args = parser.parse_args()

chunk_json_file = args.chunk_json_file
labels_file = args.labels_file
pad_token = args.pad_token
label_downsampling = args.label_downsampling

output_dir = os.path.join(os.path.dirname(labels_file), "seg")
if not os.path.exists(output_dir): os.makedirs(output_dir)

chunks = json.load(open(chunk_json_file, encoding='utf8'))
# Get info about the chunk_size_sec and chunk_shift_sec
chunk_size_num_frames = list(chunks.values())[0]['info']['chunk_size_num_frames']
chunk_shift_num_frames = list(chunks.values())[0]['info']['chunk_shift_num_frames']

# Pad or trim the chunk seq according to chunk_size_sec and chunk_shift_sec. Deal with label downsampling
num_padded_seq = 0
num_trimmed_seq = 0
num_total_seq = 0
num_has_calls_seq = 0
num_all_noises_seq = 0
num_ref_has_calls_seq = 0
num_ref_all_noises_seq = 0
uttid2chunkseq = {}
uttid2paddedchunkseq = {}
with open(labels_file) as f:
    for line in f:
        line = line.strip()
        uttid, content = re.split("\s+", line, maxsplit=1)
        label_list = re.split("\s+", content)
        # ['a', 'b', 'b'] with label_downsampling 3 becomes ['a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b'] 
        label_list = [label for label in label_list for i in range(label_downsampling)]

        # Collect hypothesis statistics
        if (set(label_list) == {"noise"}):
            num_all_noises_seq += 1
        else:
            num_has_calls_seq += 1

        # Collect reference statistics
        if (chunks[uttid]['info']['state'] == "has_calls"):
            num_ref_has_calls_seq += 1
        else:
            num_ref_all_noises_seq += 1

        # pad and trim the seq according to the chunk_size_sec and chunk_shift_sec
        if (len(label_list) < chunk_size_num_frames):
            padded_seq_size = chunk_size_num_frames - len(label_list)
            padded_seq = [pad_token] * padded_seq_size
            padded_seq = label_list + padded_seq
            assert(len(padded_seq) == chunk_size_num_frames)
            print("Warning: padding seq with uttid: {} of len {} to len {}\noriginal: {}\npadded: {}".format(uttid, len(label_list), len(padded_seq), content, " ".join(padded_seq)))
            num_padded_seq += 1
        elif (len(label_list) > chunk_size_num_frames):
            padded_seq = label_list[:chunk_size_num_frames]
            num_trimmed_seq += 1
            print("Warning: trimming seq with uttid: {} of len {} to len {}\noriginal: {}\npadded: {}".format(uttid, len(label_list), len(padded_seq), content, " ".join(padded_seq)))
        else:
            padded_seq = label_list

        num_total_seq += 1
        uttid2chunkseq[uttid] = label_list
        uttid2paddedchunkseq[uttid] = padded_seq

print(f"num_total_seq: {num_total_seq}\nnum_padded_seq: {num_padded_seq}\nnum_trimmed_seq: {num_trimmed_seq}\nnum_total_seq: {num_total_seq}\nnum_all_noises_seq: {num_all_noises_seq}\nnum_has_calls_seq: {num_has_calls_seq}\nnum_ref_has_calls_seq: {num_ref_has_calls_seq}\nnum_ref_all_noises_seq: {num_ref_all_noises_seq}\n")

# Merge the chunks. Split the sequence into animal one and two. Deal with sliding window overlaps.
num_chunks = len(chunks.keys())
wavid1 = list(chunks.values())[0]['info']['wavid1']
wavid2 = list(chunks.values())[0]['info']['wavid2']
frame_size_sec = list(chunks.values())[0]['info']['frame_size_sec']
frame_shift_sec = list(chunks.values())[0]['info']['frame_shift_sec']

# uttid in the format of wavid1_wavid2_chunkid
pred_label = []
pred_label1 = []
pred_label2 = []
for chunkid in range(num_chunks):
    uttid = f"{wavid1}_{wavid2}_{chunkid}"
    padded_seq = uttid2paddedchunkseq[uttid]
    pred_seq = uttid2chunkseq[uttid]
    seq = None # calls for two animals such as 'tr' and 'tr2'
    if chunkid == 1:
        seq = padded_seq # complete window for the first frame
    elif chunkid == num_chunks - 1:
        seq = pred_seq # predicted window for the last fram
    else:
        seq = padded_seq[:chunk_shift_num_frames] # Non-overlap part of the slide window

    seq1 = [] # calls for animal1
    seq2 = [] # calls for animal2
    has_call1 = False
    has_call2 = False
    for call in seq:
        if call == 'noise':
            seq1.append('noise')
            seq2.append('noise')
        elif call.endswith('2'):
            seq1.append('noise')
            seq2.append(call[:-1]) # 'tr2' => 'tr'
            has_call2 = True
        else:
            seq1.append(call)
            seq2.append('noise')
            has_call1 = True
    # if (has_call1): print(f"Has call1\n{uttid=}\n{seq=}\n{seq1=}\n{seq2=}")
    # if (has_call2): print(f"Has call2\n{uttid=}\n{seq=}\n{seq1=}\n{seq2=}")
    pred_label += seq
    pred_label1 += seq1
    pred_label2 += seq2

# Convert label sequences into segment files.
seg_file = os.path.join(output_dir, "test_" + wavid1 + wavid2 + ".txt")
with open(seg_file, 'w') as f:
    print(f"seg_file:\n{seg_file}")
    segs = framelabel2audacitysegment(pred_label, window_size=frame_size_sec, window_shift=frame_shift_sec)
    for seg in segs:
        if seg.label != "noise":
            f.write(f"{seg.begin_sec}\t{seg.end_sec}\t{seg.label}\n")

seg_file = os.path.join(output_dir, "test_" + wavid1 + ".txt")
with open(seg_file, 'w') as f:
    print(f"seg_file:\n{seg_file}")
    segs = framelabel2audacitysegment(pred_label1, window_size=frame_size_sec, window_shift=frame_shift_sec)
    for seg in segs:
        if seg.label != "noise":
            f.write(f"{seg.begin_sec}\t{seg.end_sec}\t{seg.label}\n")

seg_file = os.path.join(output_dir, "test_" + wavid2 + ".txt")
with open(seg_file, 'w') as f:
    print(f"seg_file:\n{seg_file}")
    segs = framelabel2audacitysegment(pred_label2, window_size=frame_size_sec, window_shift=frame_shift_sec)
    for seg in segs:
        if seg.label != "noise":
            f.write(f"{seg.begin_sec}\t{seg.end_sec}\t{seg.label}\n")

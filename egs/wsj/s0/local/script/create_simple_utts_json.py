from clib.kaldi import kaldi_io
import argparse
import json
import io
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Create a simple utts.json");
    parser.add_argument("--output_dir", type=str, default="data/test_small",
                        help="output directory contains the sample utts json file and the kaldi feature file")
    args = parser.parse_args()
    feat1 = np.array([[0.1, 0.1], [0.2, 0.2], [0.2, 0.2]])
    kaldi_io.write_mat(args.output_dir + "/feats1.ark", feat1)
    feat2 = np.array([[0.1, 0.1], [0.0, 0.0], [0.2, 0.2], [0.2, 0.2]])
    kaldi_io.write_mat(args.output_dir + "/feats2.ark", feat2)
    feat3 = np.array([[0.1, 0.1], [0.3, 0.3], [0.3, 0.3], [0.3, 0.3], [0.0, 0.0], [0.2, 0.2], [0.2, 0.2]])
    kaldi_io.write_mat(args.output_dir + "/feats3.ark", feat3)
    feat4 = np.array([[0.1, 0.1], [0.3, 0.3], [0.3, 0.3], [0.3, 0.3], [0.0, 0.0], [0.1, 0.1]])
    kaldi_io.write_mat(args.output_dir + "/feats4.ark", feat4)

    # id 0 for <unk>
    # id 1 for <pad>
    # id 2, 3 for <sos> and <eos>
    # id 4 for <space>
    # <unk> in vocabulary; <pad> not
    utts = {
        "spk1_u1": {
            "feat": args.output_dir + "/feats1.ark",
            "feat_dim": "2",
            "num_frames": "3",
            "num_tokens": "4",
            "text": "AB",
            "token": "<sos> A B <eos>",
            "tokenid": "2 5 6 3",
            "utt2spk": "spk1",
            "uttid": "spk1_u1",
            "vocab_size": "7"
        },
        "spk1_u2": {
            "feat": args.output_dir + "/feats2.ark",
            "feat_dim": "2",
            "num_frames": "4",
            "num_tokens": "5",
            "text": "A B",
            "token": "<sos> A <space> B <eos>",
            "tokenid": "2 5 4 6 3",
            "utt2spk": "spk1",
            "uttid": "spk1_u2",
            "vocab_size": "7"
        },
        "spk1_u3": {
            "feat": args.output_dir + "/feats3.ark",
            "feat_dim": "2",
            "num_frames": "7",
            "num_tokens": "6",
            "text": "AC B",
            "token": "<sos> A C <space> B <eos>",
            "tokenid": "2 5 7 4 6 3",
            "utt2spk": "spk1",
            "uttid": "spk1_u3",
            "vocab_size": "7"
        },
        "spk2_u4": {
            "feat": args.output_dir + "/feats4.ark",
            "feat_dim": "2",
            "num_frames": "6",
            "num_tokens": "6",
            "text": "AC A",
            "token": "<sos> A C <space> A <eos>",
            "tokenid": "2 5 7 4 5 3",
            "utt2spk": "spk2",
            "uttid": "spk2_u4",
            "vocab_size": "7"
        }
    }

    if args.output_dir:
        with open(args.output_dir + "/utts.json", 'w', encoding='utf-8') as fuo:
            json.dump(utts, fp=fuo, indent=4, sort_keys=True, ensure_ascii=False)
    else:
        json.dump(utts, fp=io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8'), indent=4, sort_keys=True, ensure_ascii=False)

if __name__ == "__main__":
   main()

# Add clib package at current directory to the binary searching path.
import sys
import os
sys.path.append(os.getcwd())

from clib.kaldi import kaldi_io
import argparse
import json
import yaml
import io
import sys
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Create a simple utts.json");
    parser.add_argument("--output_dir", type=str, default="data/test_small",
                        help="output directory contains the sample utts json file and the kaldi feature file")
    parser.add_argument("--without_config", action='store_true', help="without creating configuarion files.")

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
    token2id = {"<unk>": 0,
                "<pad>": 1,
                "<sos>": 2,
                "<eos>": 3,
                "<space>": 4,
                "A": 5,
                "B": 6,
                "C": 7}

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
            "vocab_size": "8"
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
            "vocab_size": "8"
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
            "vocab_size": "8"
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
            "vocab_size": "8"
        }
    }

    model_config_str = "---\n" + \
                       "class:\n" + \
                       "    \"<class '__main__.EncRNNDecRNNAtt'>\"\n" + \
                       "att_config:\n" + \
                       "    type: mlp\n" + \
                       "enc_fnn_sizes: [4, 9]\n" + \
                       "enc_rnn_sizes: [5, 5, 5]\n" + \
                       "enc_rnn_subsampling: [False, True, True]\n" + \
                       "enc_rnn_subsampling_type: 'concat'\n" + \
                       "dec_embedding_size: 6\n" + \
                       "dec_rnn_sizes: [8, 8]\n" + \
                       "dec_context_proj_size: 6"

    data_config_str = "train: '" + args.output_dir + "/utts.json'\n" + \
                      "dev: '" + args.output_dir + "/utts.json'\n" + \
                      "test: '" + args.output_dir + "/utts.json'\n" + \
                      "token2id: '" + args.output_dir + "/token2id.txt'"

    opts = vars(args)
    if args.output_dir:
        with open(args.output_dir + "/utts.json", 'w', encoding='utf-8') as fuo:
            json.dump(utts, fp=fuo, indent=4, sort_keys=True, ensure_ascii=False)
        with open(args.output_dir + "/token2id.txt", 'w', encoding='utf8') as ft2d:
            for key, value in token2id.items():
                print(str(key) + " " + str(value), file=ft2d)

        if not args.without_config:
            with open(args.output_dir + "/model.yaml", 'w', encoding='utf8') as f:
                print(model_config_str, file=f)
            with open(args.output_dir + "/data.yaml", 'w', encoding='utf8') as f:
                print(data_config_str, file=f)
    else:
        json.dump(utts, fp=io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8'), indent=4, sort_keys=True, ensure_ascii=False)

if __name__ == "__main__":
   main()

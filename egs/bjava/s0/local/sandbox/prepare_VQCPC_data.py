# Format of the 
# [bin-wu@ahctitan02 VectorQuantizedCPC]$(master *) head datasets/2019/english/train.json
# [
#  [
#   "english/train/unit/S015_0361841101", # path of the file, not necessary
#   0.0,
#   13.13, # duration (sec)
#   "english/train/S015/S015_0361841101" # out_path in the format of …./speaker/name_without_suffix
#   ],
#  [
#   "english/train/unit/S015_0391801644",
#   0.0,
#   [bin-wu@ahctitan02 VectorQuantizedCPC]$(master *) head datasets/2019/english/speakers.json
#   [
#    "S015",
#    "S020",
#    "S021",
#    "S023"]

import os
import argparse
import kaldi_io
import json
import numpy as np

def kdata2vdata(data_dir, data_type):
#    data_dir = train_dir
#    data_type = "train" or "dev" or "test"

    speakers = []
    uttids = []

    utt2dur = {}
    with open(os.path.join(data_dir, "utt2dur"), 'r') as f:
        for line in f:
            uttid, dur = line.strip().split()
            utt2dur[uttid] = float(dur)
            uttids.append(uttid)

    utt2spk = {}
    with open(os.path.join(data_dir, 'utt2spk'), 'r') as f:
        for line in f:
            uttid, spk = line.strip().split()
            utt2spk[uttid] = spk
            speakers.append(spk)

    for uttid, feat in kaldi_io.read_mat_scp(os.path.join(data_dir, 'feats.scp')):
        feat_dir = os.path.join(args.vdata, data_type, utt2spk[uttid]) # datasets/2019/english/train/S015/S015_0361841101
        if not os.path.exists(feat_dir):
            os.makedirs(feat_dir)
            # Kaldi feat with shape (605, 39), vq feat with shape (19, 605)
            # datasets/2019/english/train/S015/S015_0361841101.mel.npy
        np.save(os.path.join(feat_dir, uttid + ".mel.npy"), feat.T)
#        break

    info = []
    for uttid in uttids:
        info.append(["", 0, utt2dur[uttid], os.path.join(os.path.basename(args.vdata), data_type, utt2spk[uttid], uttid)])

    with open(os.path.join(args.vdata, data_type + ".json"), 'w') as f_out:
        json.dump(info, f_out, indent=4)

    return speakers

jdir = "/project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/bjava/s0"
parser = argparse.ArgumentParser(description="Kaldi data to VQCEP data")
parser.add_argument("--kdata", type=str, default=os.path.join(jdir, "data"), help="the Kaldi data directory, e.g., TIMIT/s0/data")
parser.add_argument("--vdata", type=str, default=os.path.join(jdir, "exp", "javanese_mfcc"), help="the VQCPC data directory, e.g.  datasets/2019/english, assuming the dirname is dataset name such as 'english'")

parser.add_argument("--kdata_train_dirname", type=str, default="train", help="the name of the training data of the Kaldi dataset, e.g. train for TIMIT of Kaldi")
parser.add_argument("--kdata_dev_dirname", type=str, default="dev", help="the name of the development data of the Kaldi dataset, e.g. dev")
parser.add_argument("--kdata_test_dirname", type=str, default="test", help="the name of the test data of the kaldi dataset, e.g. test")

args = parser.parse_args()

if not os.path.exists(args.vdata):
    os.makedirs(args.vdata)

train_dir = os.path.join(args.kdata, args.kdata_train_dirname)
dev_dir = os.path.join(args.kdata, args.kdata_dev_dirname)
test_dir = os.path.join(args.kdata, args.kdata_test_dirname)

train_spk = kdata2vdata(train_dir, "train")
dev_spk = kdata2vdata(dev_dir, "dev")
test_spk = kdata2vdata(test_dir, "test")

spk = train_spk + dev_spk + test_spk
spk_sorted = sorted(list(set(spk)))

with open(os.path.join(args.vdata, "speakers.json"), 'w') as f_out:
    json.dump(spk_sorted, f_out, indent=4)

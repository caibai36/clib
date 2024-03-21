from typing import Optional

import os
import sys
import argparse
import logging
logging.basicConfig(stream=sys.stdout,level=logging.INFO, format="[ %(asctime)s | %(filename)s | %(levelname)s ] %(message)s", datefmt="%d/%m/%Y %H:%M:%S")

import numpy as np
from scipy.stats import chi2
from scipy.stats import multivariate_normal
# from sklearn import metrics

import kaldi_io
if not 'KALDI_ROOT' in os.environ:
    # Default! To change run python with 'export KALDI_ROOT=/some_dir python'
    os.environ['KALDI_ROOT'] = '/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi'

parser = argparse.ArgumentParser(description=f"Train DPGMM clustering on training features and test it on test feautures",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--feat_train", type=str, default="data/test3utts/feats.scp", help="the training feat scp file")
parser.add_argument("--feat_dev", type=str, default="data/test3utts/feats.scp", help="the development feat scp file")
parser.add_argument("--feat_test", type=str, default="data/test3utts/feats.scp", help="the test feat scp file")
parser.add_argument("--feat_train2", type=str, default="data/test3utts/feats.scp", help="the second training feat scp file concatenated with the previous one")
parser.add_argument("--feat_dev2", type=str, default="data/test3utts/feats.scp", help="the second the development feat scp file concatenated with the previous one")
parser.add_argument("--feat_test2", type=str, default="data/test3utts/feats.scp", help="the second the test feat scp file concatenated with the previous one")

parser.add_argument("--result", type=str, default="exp/dpgmm2asr/test3utt/data",
                    help="dir ${result} contains the data for asr construction of DPGMM features.")
parser.add_argument("--verbose", type=bool, default=True, help="print verbose information or not")

args = parser.parse_args()

train_features = []
for uttid, feature in kaldi_io.read_mat_scp(args.feat_train):
    train_features.append(feature)
sources = np.concatenate(train_features, axis=0)

train_features = []
train_lengths = []
train_uttids = []
for uttid, feature in kaldi_io.read_mat_scp(args.feat_train):
    train_features.append(feature)
    train_lengths.append(feature.shape[0])
    train_uttids.append(uttid)
sources_train = np.concatenate(train_features, axis=0)

dev_features = []
dev_lengths = []
dev_uttids = []
for uttid, feature in kaldi_io.read_mat_scp(args.feat_dev):
    dev_features.append(feature)
    dev_lengths.append(feature.shape[0])
    dev_uttids.append(uttid)
sources_dev = np.concatenate(dev_features, axis=0)

test_features = []
test_lengths = []
test_uttids = []
for uttid, feature in kaldi_io.read_mat_scp(args.feat_test):
    test_features.append(feature)
    test_lengths.append(feature.shape[0])
    test_uttids.append(uttid)
sources_test = np.concatenate(test_features, axis=0)

train2_features = []
train2_lengths = []
train2_uttids = []
for uttid, feature in kaldi_io.read_mat_scp(args.feat_train2):
    train2_features.append(feature)
    train2_lengths.append(feature.shape[0])
    train2_uttids.append(uttid)
sources_train2 = np.concatenate(train2_features, axis=0)

dev2_features = []
dev2_lengths = []
dev2_uttids = []
for uttid, feature in kaldi_io.read_mat_scp(args.feat_dev2):
    dev2_features.append(feature)
    dev2_lengths.append(feature.shape[0])
    dev2_uttids.append(uttid)
sources_dev2 = np.concatenate(dev2_features, axis=0)

test2_features = []
test2_lengths = []
test2_uttids = []
for uttid, feature in kaldi_io.read_mat_scp(args.feat_test2):
    test2_features.append(feature)
    test2_lengths.append(feature.shape[0])
    test2_uttids.append(uttid)
sources_test2 = np.concatenate(test2_features, axis=0)

embedding_train = np.concatenate([sources_train, sources_train2], axis=1)
embedding_dev = np.concatenate([sources_dev, sources_dev2], axis=1)
embedding_test = np.concatenate([sources_test, sources_test2], axis=1)

print(embedding_train.shape)
print(embedding_dev.shape)
print(embedding_test.shape)
dir_train = os.path.join(args.result, "train")
dir_dev = os.path.join(args.result, "dev")
dir_test = os.path.join(args.result, "test")

if not os.path.exists(args.result):
    os.makedirs(args.result)
    os.makedirs(dir_train)
    os.makedirs(dir_dev)
    os.makedirs(dir_test)

print(f"shape of train fature: {embedding_train.shape}")
print(f"shape of dev feature: {embedding_dev.shape}")
print(f"shape of test feature: {embedding_test.shape}")

start_index = 0
embedding_train_list = []
for i in range(len(train_lengths)):
        embedding_train_list.append(embedding_train[start_index:start_index+train_lengths[i]])
        start_index += train_lengths[i]

scp_file = os.path.join(dir_train, "feats.scp")
ark_file = os.path.join(dir_train, "train.ark")
ark_scp_output=f'ark:| copy-feats --compress=false ark:- ark,scp:{ark_file},{scp_file}'
with kaldi_io.open_or_fd(ark_scp_output,'wb') as f:
    for key, mat in zip(*(train_uttids, embedding_train_list)):
        kaldi_io.write_mat(f, mat, key=key)

start_index = 0
embedding_dev_list = []
for i in range(len(dev_lengths)):
        embedding_dev_list.append(embedding_dev[start_index:start_index+dev_lengths[i]])
        start_index += dev_lengths[i]

scp_file = os.path.join(dir_dev, "feats.scp")
ark_file = os.path.join(dir_dev, "dev.ark")
ark_scp_output=f'ark:| copy-feats --compress=false ark:- ark,scp:{ark_file},{scp_file}'
with kaldi_io.open_or_fd(ark_scp_output,'wb') as f:
    for key, mat in zip(*(dev_uttids, embedding_dev_list)):
        kaldi_io.write_mat(f, mat, key=key)

start_index = 0
embedding_test_list = []
for i in range(len(test_lengths)):
        embedding_test_list.append(embedding_test[start_index:start_index+test_lengths[i]])
        start_index += test_lengths[i]

scp_file = os.path.join(dir_test, "feats.scp")
ark_file = os.path.join(dir_test, "test.ark")
ark_scp_output=f'ark:| copy-feats --compress=false ark:- ark,scp:{ark_file},{scp_file}'
with kaldi_io.open_or_fd(ark_scp_output,'wb') as f:
    for key, mat in zip(*(test_uttids, embedding_test_list)):
        kaldi_io.write_mat(f, mat, key=key)

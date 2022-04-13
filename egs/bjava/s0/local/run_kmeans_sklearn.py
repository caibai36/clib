import os
import argparse

import numpy as np
import sklearn
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

import kaldi_io

def index2onehot(index_vector, num_of_classes):
    """Convert an index vector to its one_of_k encoded one-hot representation.
    Args: 
        index_vector (np.ndarray): the given index vector
        num_of_classes (int): the number of classes
    Returns:
        one_hot (np.ndarray): the one-hot representation of the indexed vector.
    """
    one_hot = np.zeros((len(index_vector), num_of_classes))
    one_hot[np.arange(len(index_vector)), index_vector] = 1
    return one_hot

parser = argparse.ArgumentParser(description="Get K-means onehot features for develpment and test sets from the model by the training set.")
parser.add_argument("--feat_train", type=str, required=False, default="data/train/feats.scp", help="Kaldi scp file of training feature")
parser.add_argument("--feat_dev", type=str, required=False, default="data/dev/feats.scp", help="Kaldi scp file of develpment feature")
parser.add_argument("--feat_test", type=str, required=False, default="data/test/feats.scp", help="Kaldi scp file fo testing feature")
parser.add_argument("--output_feat_dir", type=str, required=False, default="feat/kmeans_feat", help="test post")

# Arguments for Kmeans
# km = KMeans(n_clusters=99, init="random", max_iter=300, random_state=None, algorithm="full", verbose=2).fit(X)
parser.add_argument("--K", type=int, default=99, help="the number of clusters (K)")
parser.add_argument("--seed", type=int, default=None, help="the random seed or the random state")
parser.add_argument("--algorithm", type=str, choices=["full", "auto", "elkan"], default="full", help="the type of algorithm")
parser.add_argument("--epochs", type=int, default=300, help="the maximum iteraion or epochs for k-means algorithm")
parser.add_argument("--init", type=str, default="random", choices=["k-means++", "random"], help="the way of initalizing the parameters using kmeans++ or random")
parser.add_argument("--verbose", type=int, default=2, help="Enable verbose output. If 1 then it prints the current initialization and each iteration step. If greater than 1 then it prints also the log probability and the time needed for each step.")

args = parser.parse_args()
print(args)

dir_out = args.output_feat_dir
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

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

km = KMeans(n_clusters=args.K, init=args.init, max_iter=args.epochs, random_state=args.seed, algorithm=args.algorithm, verbose=args.verbose).fit(sources_train)

embedding_train = index2onehot(km.predict(sources_train), args.K)
embedding_dev = index2onehot(km.predict(sources_dev), args.K)
embedding_test = index2onehot(km.predict(sources_test), args.K)

print("shape of kmean feature of training set {}".format(embedding_train.shape))
print("shape of kmean feature of development set {}".format(embedding_dev.shape))
print("shape of kmean feature of test set {}".format(embedding_test.shape))

start_index = 0
embedding_train_list = []
for i in range(len(train_lengths)):
        embedding_train_list.append(embedding_train[start_index:start_index+train_lengths[i]])
        start_index += train_lengths[i]

scp_file = os.path.join(dir_out, "train.scp")
ark_file = os.path.join(dir_out, "train.ark")
ark_scp_output=f'ark:| copy-feats --compress=false ark:- ark,scp:{ark_file},{scp_file}'
with kaldi_io.open_or_fd(ark_scp_output,'wb') as f:
    for key, mat in zip(*(train_uttids, embedding_train_list)):
        kaldi_io.write_mat(f, mat, key=key)

start_index = 0
embedding_dev_list = []
for i in range(len(dev_lengths)):
        embedding_dev_list.append(embedding_dev[start_index:start_index+dev_lengths[i]])
        start_index += dev_lengths[i]

scp_file = os.path.join(dir_out, "dev.scp")
ark_file = os.path.join(dir_out, "dev.ark")
ark_scp_output=f'ark:| copy-feats --compress=false ark:- ark,scp:{ark_file},{scp_file}'
with kaldi_io.open_or_fd(ark_scp_output,'wb') as f:
    for key, mat in zip(*(dev_uttids, embedding_dev_list)):
        kaldi_io.write_mat(f, mat, key=key)

start_index = 0
embedding_test_list = []
for i in range(len(test_lengths)):
        embedding_test_list.append(embedding_test[start_index:start_index+test_lengths[i]])
        start_index += test_lengths[i]

scp_file = os.path.join(dir_out, "test.scp")
ark_file = os.path.join(dir_out, "test.ark")
ark_scp_output=f'ark:| copy-feats --compress=false ark:- ark,scp:{ark_file},{scp_file}'
with kaldi_io.open_or_fd(ark_scp_output,'wb') as f:
    for key, mat in zip(*(test_uttids, embedding_test_list)):
        kaldi_io.write_mat(f, mat, key=key)

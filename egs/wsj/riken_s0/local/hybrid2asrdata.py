from typing import Optional, Dict, Callable

import os
import sys
import argparse
import pprint
import logging
logging.basicConfig(stream=sys.stdout,level=logging.INFO, format="[ %(asctime)s | %(filename)s | %(levelname)s ] %(message)s", datefmt="%d/%m/%Y %H:%M:%S")

import numpy as np
from scipy.stats import chi2
from scipy.stats import multivariate_normal
# from sklearn import metrics

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import kaldi_io
if not 'KALDI_ROOT' in os.environ:
    # Default! To change run python with 'export KALDI_ROOT=/some_dir python'
    os.environ['KALDI_ROOT'] = '/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi'

def indices2onehot_np(class_indices: np.array, num_classes: Optional[int] = None) -> np.array:
    """
    Convert a sequence of indices its onehot representation (numpy version)
    
    Example
    -------
    In [30]: %run local/dpgmm2embedding.py

    In [31]: indices2onehot_np(np.array([1, 0, 3]))
    Out[31]: 
    array([[0., 1., 0., 0.],
           [1., 0., 0., 0.],
           [0., 0., 0., 1.]])
    """
    num_classes = class_indices.max() + 1 if num_classes is None else num_classes
    onehot = np.zeros((class_indices.size, num_classes))
    onehot[np.arange(class_indices.size), class_indices] = 1
    return onehot

class ParallelDataset(Dataset):

    def __init__(self,
                 source_feats: np.ndarray,
                 target_feats: np.ndarray,
                 num_left_context: int = 0,
                 num_right_context: int = 0) -> None:
        """
        Parallel dataset of a list of source features and a list of target features.
        Target feature can be discrete labels or continuous feature vector.
        source features with left few contexts is supported.

        Arguments
        ---------
        source_feats: shape (num_feats, input_dim)
        target_feats: shape (num_feats,) or (num_feats, output_dim)
        num_left_context: number of left contexts exclude the current feature
        num_right_context: number of right contexts exclude the current feature

        Examples
        --------
        In [461]: source = np.array([1, 2, 3])
        In [462]: target = np.array([0.1, 0.2, 0.1])
        In [463]: dataset = ParallelDataset(source, target, num_left_context=2, num_right_context=2)
        In [466]: for index in range(len(dataset)):
             ...:     print(dataset[index])
        {'source': array([1, 1, 1, 2, 3]), 'target': 0.1}
        {'source': array([1, 1, 2, 3, 3]), 'target': 0.2}
        {'source': array([1, 2, 3, 3, 3]), 'target': 0.1}
        """
        super().__init__()
        # assert num_left_context > 0, "num_left_context should be greater than zero"
        # assert num_right_context > 0, "num_right_context should be greater than zero"
        source_length = len(source_feats)
        if num_left_context: source_head_fake = np.stack([source_feats[0]] * num_left_context, axis=0)
        if num_right_context: source_tail_fake = np.stack([source_feats[source_length - 1]] * num_right_context, axis=0)
        if num_left_context and num_right_context:
            self.source_feats = np.concatenate([source_head_fake, source_feats, source_tail_fake], axis=0)
        elif num_left_context and not num_right_context:
            self.source_feats = np.concatenate([source_head_fake, source_feats], axis=0)
        elif not num_left_context and num_right_context:
            self.source_feats = np.concatenate([source_feats, source_tail_fake], axis=0)
        else:
            self.source_feats = source_feats

        target_length = len(target_feats)
        if num_left_context: target_head_fake = np.stack([target_feats[0]] * num_left_context, axis=0)
        if num_right_context: target_tail_fake = np.stack([target_feats[target_length - 1]] * num_right_context, axis=0)
        if num_left_context and num_right_context:
            self.target_feats = np.concatenate([target_head_fake, target_feats, target_tail_fake], axis=0)
        elif num_left_context and not num_right_context:
            self.target_feats = np.concatenate([target_head_fake, target_feats], axis=0)
        elif not num_left_context and num_right_context:
            self.target_feats = np.concatenate([target_feats, target_tail_fake], axis=0)
        else:
            self.target_feats = target_feats

        self.num_left_context = num_left_context
        self.num_right_context = num_right_context

    def __len__(self) -> int:
        return len(self.source_feats) - self.num_left_context - self.num_right_context

    # It is OK for dataset to return array or integer
    # Dataloader will convert all numpy array to tensor
    # Dataloader will convert all integer or float to tensor.
    def __getitem__(self, index: int) -> Dict:
        return {'source': self.source_feats[index: index + self.num_left_context + self.num_right_context + 1],
                'target': self.target_feats[index + self.num_left_context]}

class ParallelCENet(nn.Module):

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int) -> None:
        """
        A neural network with cross entropy loss for parallel dataset with contexts.
        As the prediction will be posteriorgram, so we should add a softmax layer
        """
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)
        self.project = nn.Linear(hidden_dim, output_dim)
        self.loss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,
                source: torch.FloatTensor,
                target: torch.LongTensor) -> Dict:
        """
        source: shape (batch_size, seq_length, input_dim)
        target: shape (seq_length,)
        """
        o, (h, c) = self.lstm(source)
        assert torch.all(torch.eq(h[-1], o[:,-1,:])), "output and hidden dimension mismatching."
        predicted = self.softmax(self.project(h[-1]))
        loss = self.loss(predicted, target)
        return {'predicted': predicted, 'loss': loss}

parser = argparse.ArgumentParser(description=f"Train DPGMM clustering on training features and test it on test feautures",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--feat_train", type=str, default="data/test3utts/feats.scp", help="the training feat scp file")
parser.add_argument("--feat_dev", type=str, default="data/test3utts/feats.scp", help="the development feat scp file")
parser.add_argument("--feat_test", type=str, default="data/test3utts/feats.scp", help="the test feat scp file")

parser.add_argument("--dpgmm_train", type=str, default="exp/dpgmm2asr/test3utt/data/train/feats.scp", help="the training feat scp file")
parser.add_argument("--dpgmm_dev", type=str, default="exp/dpgmm2asr/test3utt/data/dev/feats.scp", help="the development feat scp file")
parser.add_argument("--dpgmm_test", type=str, default="exp/dpgmm2asr/test3utt/data/test/feats.scp", help="the test feat scp file")

parser.add_argument("--result", type=str, default="exp/hybrid2asr/test3utt/data",
                    help="dir ${result} contains the data for asr construction of DPGMM features.")
parser.add_argument("--verbose", type=bool, default=True, help="print verbose information or not")

parser.add_argument("--seed", type=int, default=2019)
parser.add_argument('--gpu', type=str, default="0", # if default is 'auto', running three times in ipython will occupy three different gpus.
                                        help="e.g., '--gpu 2' for using 'cuda:2'; '--gpu auto' for using the device with least gpu memory ")

parser.add_argument("--num_left_context", type=int, default=0)
parser.add_argument("--num_right_context", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=2)

parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--num_layers", type=int, default=3)
parser.add_argument("--learning_rate", type=float, default=0.001)

parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--print_interval", type=int, default=400)

args = parser.parse_args()
np.random.seed(seed=args.seed)

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

sources_dev_test = np.concatenate([sources_dev, sources_test], axis=0)

####
train_features = []
for uttid, feature in kaldi_io.read_mat_scp(args.dpgmm_train):
    train_features.append(feature)
targets_train = np.concatenate(train_features, axis=0)

dev_features = []
for uttid, feature in kaldi_io.read_mat_scp(args.dpgmm_dev):
    dev_features.append(feature)
targets_dev = np.concatenate(dev_features, axis=0)

test_features = []
for uttid, feature in kaldi_io.read_mat_scp(args.dpgmm_test):
    test_features.append(feature)
targets_test = np.concatenate(test_features, axis=0)

targets_dev_test = np.concatenate([targets_dev, targets_test], axis=0)

assert sources_train.shape[0] == targets_train.shape[0] and sources_dev_test.shape[0] == targets_dev_test.shape[0], "source and target dimension mismatching"
print(f"Training dataset")
print(f"shape of source training acoustic feature {sources_train.shape}")
print(f"shape of target training DPGMM posterior vectors {targets_train.shape}")
print()
print(f"Test dataset")
print(f"shape of source test/dev acoustic feature  {sources_dev_test.shape}")
print(f"shape of target test/dev DPGMM posterior vectors {targets_dev_test.shape}")
##################################
seed = args.seed
num_left_context = args.num_left_context
num_right_context = args.num_right_context
batch_size = args.batch_size

hidden_dim = args.hidden_dim
num_layers = args.num_layers
learning_rate = args.learning_rate

num_epochs = args.num_epochs
print_interval = args.print_interval

opts = vars(args)
if opts['gpu'] != 'auto':
    device = torch.device("cuda:{}".format(opts['gpu']) if torch.cuda.is_available() else "cpu")
else:
    import GPUtil # Get the device using the least GPU memory.
    device = torch.device("cuda:{}".format(GPUtil.getAvailable(order='memory')[0]) if torch.cuda.is_available() and \
                          GPUtil.getAvailable(order='memory') else "cpu")
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
print("python " + ' '.join([x for x in sys.argv])) # save current script command
print("Getting Options...")
print("\n" + pprint.pformat(opts))
print("Device: '{}'".format(device))

train_source_data = sources_train
train_target_data = np.argmax(targets_train, axis=1) # cluster posteriorgram to cluster label

test_source_data = sources_dev_test
test_target_data = np.argmax(targets_dev_test, axis=1)

assert(train_source_data.shape[1] == test_source_data.shape[1])
input_dim = train_source_data.shape[1]
assert train_target_data.min() >= 0, "the class of index should be greater or equal to zero"
output_dim = int(train_target_data.max()) + 1 # We index the classes from 0

train_dataset = ParallelDataset(train_source_data, train_target_data, num_left_context=num_left_context, num_right_context=num_right_context)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size)

test_dataset = ParallelDataset(test_source_data, test_target_data, num_left_context=num_left_context, num_right_context=num_right_context)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)

model = ParallelCENet(input_dim, hidden_dim, output_dim, num_layers)
model.to(device)
print(model)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

step = 0
running_loss = 0
for epoch in range(num_epochs):
    for batch in train_dataloader:
        source = batch['source'].to(device).float()
        target = batch['target'].to(device).long()

        output = model(source, target)

        optimizer.zero_grad()
        output['loss'].backward()
        optimizer.step()

        running_loss += output['loss'].item()
        step += 1

        if (step % print_interval == 0):
            print(f"Epoch: {epoch} batch: {step} loss: {running_loss / print_interval:.4f}")
            running_loss = 0

training_outputs = []
test_outputs = []
with torch.no_grad():
    for batch in test_dataloader:
        source = batch['source'].to(device).float()
        target = batch['target'].to(device).long()

        output = model(source, target)
        test_outputs.append(output['predicted'])
        
    for batch in train_dataloader:
        source = batch['source'].to(device).float()
        target = batch['target'].to(device).long()

        output = model(source, target)
        training_outputs.append(output['predicted'])

z_posterior_train, z_posterior_dev_test = torch.cat(training_outputs, dim=0).cpu().numpy(), torch.cat(test_outputs, dim=0).cpu().numpy()
print("Shape of posteriorgrams of training data and test+dev data")
print(z_posterior_train.shape)
print(z_posterior_dev_test.shape)
print("First instance posteriorgram of training data and test+dev data")
print(z_posterior_train[0])
print(z_posterior_dev_test[0])

###################################
# Split the dev and test set
assert sum(dev_lengths) + sum(test_lengths) == len(z_posterior_dev_test)
embedding_train = z_posterior_train
embedding_dev = z_posterior_dev_test[:sum(dev_lengths)]
embedding_test = z_posterior_dev_test[sum(dev_lengths):]

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

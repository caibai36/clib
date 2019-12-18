# Add clib package at current directory to the binary searching path.
import sys
import os
import re
from pprint import pprint
sys.path.append(os.getcwd())

import json
import pprint
import argparse
import torch
from clib.kaldi.kaldi_data import KaldiDataLoader, KaldiDataset 

import torch
from torch import nn
from torch.nn import functional as F
# created by local/script/create_simple_utts_json.py
json_file = 'data/test_small/utts.json'

parser = argparse.ArgumentParser()
parser.add_argument('--json_file', type=str, default=json_file, help="the test utterance json file")
# We follow the index convention of torchtext by setting padding id as 1.
parser.add_argument('--padding_tokenid', type=int, default=1, help="the id of padding token")

# Temporarily disable passing parameters to interactive shell (for emacs to run python (or rlipython)).
# remove [] when passing parameters to python scripts (eg. python test_ars.py --padding_tokenid=-5)
args = parser.parse_args([])

with open(args.json_file, encoding='utf8') as f:
    # utts_json is a dictionary mapping utt_id to fields of each utterance
    utts_json = json.load(f)
    # Each utterance instance is a list of fields includes 'feat', 'tokenid' and etc.
    utts_instances = list(utts_json.values())

    dataset = KaldiDataset(utts_instances)
    dataloader = KaldiDataLoader(dataset=dataset, batch_size=3, padding_tokenid=args.padding_tokenid)

    batches = []
    for batch in dataloader:
        batches.append(batch)

def get_rnn(name):
    """ Get the RNN module by its name string.
    We can write string more convenient in the configuration file.
    We can also manage the already registered rnns or add the new custom rnns.
    The name can be "LSTM", 'lstm' and etc.
    """
    registered_rnn = {'lstm': nn.LSTM,
                      'gru': nn.GRU,
                      'rnn': nn.RNN,
                      'lstmcell': nn.LSTMCell,
                      'grucell': nn.GRUCell,
                      'rnncell': nn.RNNCell}

    avaliable_rnn = list(registered_rnn.keys())

    if name.lower() in registered_rnn:
        return registered_rnn[name.lower()]
    else:
        raise NotImplementedError("The rnn module '{}' is not implemented\nAvaliable rnn modules include {}".format(name, avaliable_rnn))

def get_act(name):
    """ Get the activation module by name string.
    The name be 'ReLU', 'leaky_relu' and etc.
    """
    if (getattr(nn, name, None)):
        return getattr(nn, name)
    else:
        # [key, str(value)] format:  ['ReLU', "<class 'torch.nn.modules.activation.ReLU'>"]
        avaliable_act_module =  [key for key, value in torch.nn.modules.activation.__dict__.items() if "torch.nn.modules.activation." in str(value)]
        raise NotImplementedError("The activation Module '{}' is not implemented\nAvaliable activation modules include {}".format(name, avaliable_act_module))

def get_func(name):
    """ Get the function by name string.
    The name be 'relu', 'leaky_relu' and etc.
    """
    if(not name):
        return lambda x:x
    elif (getattr(F, name, None)):
        return getattr(F, name)
    elif (getattr(torch, name, None)):
        return getattr(torch, name)
    else:
        # [key, str(value)] format ['relu', '<function relu at 0x7f6fb8121a70>'],
        available_func =  [key for key, value in torch.nn.functional.__dict__.items() if ("built-in " in str(value) or "function " in str(value)) and str(key)[0] != '_' and str(key)[-1] != '_']
        raise NotImplementedError("The function '{}' is not implemented.\nAvaliable functions include {}".format(name, available_func))

test_in = next(iter(dataloader))['feat']
in_size = test_in.shape[-1]

enc_fnn_sizes = [4, 9]
enc_fnn_act = 'ReLU'
enc_fnn_dropout=0.25

enc_fnn_layers = nn.ModuleList()
num_enc_fnn_layers = len(enc_fnn_sizes)
enc_fnn_act = enc_fnn_act if isinstance(enc_fnn_act, list) else [enc_fnn_act] * num_enc_fnn_layers # make copy for each layer.
enc_fnn_dropout = enc_fnn_dropout if isinstance(enc_fnn_dropout, list) else [enc_fnn_dropout] * num_enc_fnn_layers

pre_size = in_size
for i in range(num_enc_fnn_layers):
    enc_fnn_layers.append(nn.Linear(pre_size, enc_fnn_sizes[i]))
    enc_fnn_layers.append(get_act(enc_fnn_act[i])())
    enc_fnn_layers.append(nn.Dropout(p=enc_fnn_dropout[i]))
    pre_size = enc_fnn_sizes[i]

print(enc_fnn_layers)
print(test_in.shape)

# # nn.ModuleList is list without forward method, but nn.Sequential has.
# # f(*args) where args is a list
# test_out = nn.Sequential(*enc_fnn_layers)(test_in)
# print(test_out.shape)
test_out=test_in
for layer in enc_fnn_layers:
    test_out = layer(test_out)
print(test_out.shape)


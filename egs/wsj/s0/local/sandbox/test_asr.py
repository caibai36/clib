# Add clib package at current directory to the binary searching path.
from typing import Union

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
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
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

def length2mask(sequence_lengths: torch.tensor, max_length: Union[int, None] = None) -> torch.tensor:
    """
    Given a variable of shape ``(batch_size,)`` that represents the sequence lengths of each batch
    element, this function returns a ``(batch_size, max_length)`` mask variable.  For example, if
    our input was ``[2, 2, 3]``, with a ``max_length`` of 4, we'd return
    ``[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]``.

    Examples:
        sequence_lengths: [2, 2, 3], max_length: 4: -> mask: [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]

        In [451]: lengths = torch.tensor([2, 2, 3])
        In [452]: length2mask(lengths, 4)
        Out[452]:
        tensor([[1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 0]])

        In [453]: length2mask(lengths, 2)
        Out[453]:
        tensor([[1, 1],
                [1, 1],
                [1, 1]])

        In [276]: length2mask(lengths)
        Out[276]:
        tensor([[1, 1, 0],
                [1, 1, 0],
                [1, 1, 1]])
    """
    if max_length is None:
        max_length = torch.max(sequence_lengths)
    ones_seqs = sequence_lengths.new_ones(len(sequence_lengths), max_length)
    cumsum_ones = ones_seqs.cumsum(dim=-1)

    return (cumsum_ones <= sequence_lengths.unsqueeze(-1)).long()

input = next(iter(dataloader))['feat']
input_lengths = next(iter(dataloader))['num_frames'] # or None
# input = batches[1]['feat']
# input_lengths = batches[1]['num_frames']
training = False

input_padding_value = 0.0
in_size = input.shape[-1]

enc_fnn_sizes = [4, 9]
enc_fnn_act = 'ReLU'
enc_fnn_dropout = 0.25

enc_rnn_sizes = [5, 5, 5]
enc_rnn_config = {'type': 'lstm', 'bi': True}
enc_rnn_dropout = 0.25

enc_rnn_subsampling = [False, True, True]
enc_rnn_subsampling_type = 'pair_concat' # or 'pair_take_second'

enc_fnn_layers = nn.ModuleList()
# make copy of the configuration for each layer.
num_enc_fnn_layers = len(enc_fnn_sizes)
if not isinstance(enc_fnn_act, list): enc_fnn_act = [enc_fnn_act] * num_enc_fnn_layers
if not isinstance(enc_fnn_dropout, list): enc_fnn_dropout = [enc_fnn_dropout] * num_enc_fnn_layers
assert num_enc_fnn_layers == len(enc_fnn_act) == len(enc_fnn_dropout), "Number of list mismatches the lengths of specified configuration lists."

if (input_lengths is None):
    cur_batch_size = input.shape[0]
    max_seq_length = input.shape[1]
    input_lengths = [max_seq_length] * cur_batch_size

pre_size = in_size
for i in range(num_enc_fnn_layers):
    enc_fnn_layers.append(nn.Linear(pre_size, enc_fnn_sizes[i]))
    enc_fnn_layers.append(get_act(enc_fnn_act[i])())
    enc_fnn_layers.append(nn.Dropout(p=enc_fnn_dropout[i]))
    pre_size = enc_fnn_sizes[i]

print(enc_fnn_layers)

print("Before passing to the feedforward network")
print(input.shape)

# # nn.ModuleList is list without forward method, but nn.Sequential has.
# # f(*args) where args is a list
# output = nn.Sequential(*enc_fnn_layers)(input)
# print(output.shape)
output=input
for layer in enc_fnn_layers:
    output = layer(output)

print("After passing to the feedforward network")
print(output.shape)
print()

enc_rnn_layers = nn.ModuleList()
# make copy of the configuration for each layer.
num_enc_rnn_layers = len(enc_rnn_sizes)
if not isinstance(enc_rnn_config, list): enc_rnn_config = [enc_rnn_config] * num_enc_rnn_layers
if not isinstance(enc_rnn_dropout, list): enc_rnn_dropout = [enc_rnn_dropout] * num_enc_rnn_layers
if not isinstance(enc_rnn_subsampling, list): enc_rnn_subsampling = [enc_rnn_subsampling] * num_enc_rnn_layers
if not isinstance(enc_rnn_subsampling_type, list): enc_rnn_subsampling_type = [enc_rnn_subsampling_type] * num_enc_rnn_layers
assert num_enc_rnn_layers == len(enc_rnn_config) == len(enc_rnn_dropout) == len(enc_rnn_subsampling) == len(enc_rnn_subsampling_type), \
    "Number of rnn layers mismatches the lengths of specificed configuration lists"

input = output # pipeline
pre_size = input.shape[-1]
for i in range(num_enc_rnn_layers):
    rnn_layer = get_rnn(enc_rnn_config[i]['type'])
    enc_rnn_layers.append(rnn_layer(pre_size, enc_rnn_sizes[i], batch_first=True, bidirectional=enc_rnn_config[i]['bi']))
    pre_size = enc_rnn_sizes[i] * (2 if enc_rnn_config[i]['bi'] else 1)
    if (enc_rnn_subsampling[i] and enc_rnn_subsampling_type[i] == 'pair_concat'): pre_size = pre_size * 2

print(enc_rnn_layers)

print("Before passing to the rnn")
print(input.shape)
print("lengths: {}".format(input_lengths))
print()

output = input
output_lengths = input_lengths

too_short_for_subsampling = False # set the flag to true when speech feature is too short for subsampling.
for i in range(len(enc_rnn_layers)):
    layer = enc_rnn_layers[i]
    packed_sequence = pack(output, output_lengths, batch_first=True)
    output, _ = layer(packed_sequence) # LSTM returns '(output, [hn ,cn])'
    output, _ = unpack(output, batch_first=True, padding_value=input_padding_value) # unpack returns (data, length)
    # dropout of lstm module behaves randomly even with same torch seed, so we'll append dropout layer.
    output = F.dropout(output, p=enc_rnn_dropout[i], training=training)

    # Deal with the problem that the length is too short for subsampling
    if (output.shape[1] == 1 and enc_rnn_subsampling[i]):
        too_short_for_subsampling = True
    if (too_short_for_subsampling and enc_rnn_subsampling[i] and enc_rnn_subsampling_type[i] == 'pair_concat'):
        output = torch.cat([output] * 2, dim=-1) # Double the dimension by copying the batch for the layer i outputed features.

    # Subsampling by taking the second frame or concatenating frames for every two frames as a unit.
    if (enc_rnn_subsampling[i] and not too_short_for_subsampling):
        if (enc_rnn_subsampling_type[i] == 'pair_take_second'):
            output = output[:, (2 - 1)::2] # Sample the second frame every two frames
            output_lengths = torch.LongTensor([length // 2 for length in output_lengths])
        elif (enc_rnn_subsampling_type[i] == 'pair_concat'): # apply concatenation for each outputed features
            output = output[:, :output.shape[1] // 2 * 2].contiguous() # Drop the last frame if the length is odd.
            # without contiguous => RuntimeError: view size is not compatible...(at least one dimension spans across two contiguous subspaces).
            output = output.view(output.shape[0], output.shape[1] // 2, output.shape[2] * 2)
            output_lengths = torch.LongTensor([length // 2 for length in output_lengths])
        else:
            raise NotImplementedError("The subsampling type {} is not implemented yet.\n".format(enc_rnn_subsampling_type[i]) +
                                      "The type 'pair_take_second' only preserves the last frame every two frames.\n" +
                                      "The type 'pair_concat' concatenates the frames every two frames.\n")

    print("After layer '{}' applying the subsampling '{}' with type '{}': shape is {}, lengths is {} ".format(
        i, enc_rnn_subsampling[i], enc_rnn_subsampling_type[i], output.shape, output_lengths))

    # Print the warning if the lenghth is too short for subsampling.
    if (too_short_for_subsampling and enc_rnn_subsampling[i] and enc_rnn_subsampling_type[i] == 'pair_take_second'):
        print("Warning: Input speech too short (seq_len = 1) for 'pair_take_second' subsampling. Subsampling shuts down for the layer {} outputed features for current batch.".format(i), file=sys.stderr)
    if (too_short_for_subsampling and enc_rnn_subsampling[i] and enc_rnn_subsampling_type[i] == 'pair_concat'):
        print("Warning: Input speech too short (seq_len = 1) for 'pair_concat' subsampling. Double the dimension by copying the batch for the layer {} outputed features.".format(i), file=sys.stderr)

    # print("mask of lengths is\n{}".format(length2mask(output_lengths)))

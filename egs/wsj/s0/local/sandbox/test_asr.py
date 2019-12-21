from typing import List, Dict, Tuple, Union

import sys
import os
import re
from pprint import pprint
# Add clib package at current directory to the binary searching path.
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

class SpeechEncoder(nn.Module):
    """
    The speech encoder accepts the feature (batch_size x max_seq_length x in_size),
    passes the feature to several layers of feedforward neural network (fnn)
    and then to several layers of RNN (rnn) with subsampling
    (by concatenating every pair of frames with type 'pair_concat'
    or by taking the first frame every frame pair with the type 'pair_take_first').

    ps: We can pass the parameters by copying the same configuration to each layer
        or by specifying a list of configurations for each layer.
        We do padding at the end of sequence whenever subsampling needs more frames to concatenate.
    """

    def __init__(self,
                 enc_input_size: int,
                 enc_fnn_sizes: List[int] = [4, 9],
                 enc_fnn_act: Union[str, List[str]] = 'ReLU',
                 enc_fnn_dropout: Union[float, List[float]] = 0.25,
                 enc_rnn_sizes: List[int] = [5, 5, 5],
                 enc_rnn_config: Union[Dict, List[Dict]] = {'type': 'lstm', 'bi': True},
                 enc_rnn_dropout: Union[float, List[float]] = 0.25,
                 enc_rnn_subsampling: Union[bool, List[bool]] = [False, True, True],
                 enc_rnn_subsampling_type: Union[str, List[str]] = 'pair_concat', # 'pair_concat' or 'pair_take_first'
                 enc_input_padding_value: float = 0.0
    ) -> None:
        super().__init__()

        # make copy of the configuration for each layer.
        num_enc_fnn_layers = len(enc_fnn_sizes)
        if not isinstance(enc_fnn_act, list): enc_fnn_act = [enc_fnn_act] * num_enc_fnn_layers
        if not isinstance(enc_fnn_dropout, list): enc_fnn_dropout = [enc_fnn_dropout] * num_enc_fnn_layers

        num_enc_rnn_layers = len(enc_rnn_sizes)
        if not isinstance(enc_rnn_config, list): enc_rnn_config = [enc_rnn_config] * num_enc_rnn_layers
        if not isinstance(enc_rnn_dropout, list): enc_rnn_dropout = [enc_rnn_dropout] * num_enc_rnn_layers
        if not isinstance(enc_rnn_subsampling, list): enc_rnn_subsampling = [enc_rnn_subsampling] * num_enc_rnn_layers
        if not isinstance(enc_rnn_subsampling_type, list): enc_rnn_subsampling_type = [enc_rnn_subsampling_type] * num_enc_rnn_layers

        assert num_enc_fnn_layers == len(enc_fnn_act) == len(enc_fnn_dropout), "Number of list does not match the lengths of specified configuration lists."
        assert num_enc_rnn_layers == len(enc_rnn_config) == len(enc_rnn_dropout) == len(enc_rnn_subsampling) == len(enc_rnn_subsampling_type), \
            "Number of rnn layers does not matches the lengths of specificed configuration lists."
        for t in enc_rnn_subsampling_type:
            assert t in {'pair_concat', 'pair_take_first'}, \
                "The subsampling type '{}' is not implemented yet.\n".format(t) + \
                "Only support the type 'pair_concat' and 'pair_take_first':\n" + \
                "the type 'pair_take_first' preserves the first frame every two frames;\n" + \
                "the type 'pair_concat' concatenates the frame pair every two frames.\n"

        self.enc_input_size = enc_input_size
        self.enc_fnn_sizes = enc_fnn_sizes
        self.enc_fnn_act = enc_fnn_act
        self.enc_fnn_dropout = enc_fnn_dropout
        self.enc_rnn_sizes = enc_rnn_sizes
        self.enc_rnn_config = enc_rnn_config
        self.enc_rnn_dropout = enc_rnn_dropout
        self.enc_rnn_subsampling = enc_rnn_subsampling
        self.enc_rnn_subsampling_type = enc_rnn_subsampling_type
        self.enc_input_padding_value = enc_input_padding_value

        pre_size = self.enc_input_size
        self.enc_fnn_layers = nn.ModuleList()
        for i in range(num_enc_fnn_layers):
            self.enc_fnn_layers.append(nn.Linear(pre_size, enc_fnn_sizes[i]))
            self.enc_fnn_layers.append(get_act(enc_fnn_act[i])())
            self.enc_fnn_layers.append(nn.Dropout(p=enc_fnn_dropout[i]))
            pre_size = enc_fnn_sizes[i]

        self.enc_rnn_layers = nn.ModuleList()
        for i in range(num_enc_rnn_layers):
            rnn_layer = get_rnn(enc_rnn_config[i]['type'])
            self.enc_rnn_layers.append(rnn_layer(pre_size, enc_rnn_sizes[i], batch_first=True, bidirectional=enc_rnn_config[i]['bi']))
            pre_size = enc_rnn_sizes[i] * (2 if enc_rnn_config[i]['bi'] else 1) # for bidirectional rnn
            if (enc_rnn_subsampling[i] and enc_rnn_subsampling_type[i] == 'pair_concat'): pre_size = pre_size * 2 # for pair_concat subsampling

        self.enc_final_size = pre_size

    def get_config(self):
        return { 'class': str(self.__class__),
                 'enc_input_size': self.enc_input_size,
                 'enc_fnn_sizes': self.enc_fnn_sizes,
                 'enc_fnn_act': self.enc_fnn_act,
                 'enc_fnn_dropout': self.enc_fnn_dropout,
                 'enc_rnn_sizes': self.enc_rnn_sizes,
                 'enc_rnn_config': self.enc_rnn_config,
                 'enc_rnn_dropout': self.enc_rnn_dropout,
                 'enc_rnn_subsampling': self.enc_rnn_subsampling,
                 'enc_rnn_subsampling_type': self.enc_rnn_subsampling_type,
                 'enc_fnn_layers': self.enc_fnn_layers,
                 'enc_rnn_layers': self.enc_rnn_layers,
                 'enc_final_size': self.enc_final_size,
                 'enc_input_padding_value': self.enc_input_padding_value}

    def encode(self,
               input: torch.Tensor,
               input_lengths: Union[List[int], None] = None)->(torch.FloatTensor, torch.Tensor):
        """ Encode the feature (batch_size x max_seq_length x in_size),
        and output the context vector (batch_size x max_seq_length' x context_size)
        and its mask (batch_size x max_seq_length').

        Note: the dimension of context vector is influenced by 'bidirectional' and 'subsampling (pair_concat)' options of RNN.
              the max_seq_length' influenced by 'subsampling' options of RNN.
        """
        print("Shape of the input: {}".format(input.shape))

        if (input_lengths is None):
            cur_batch_size, max_seq_length, cur_input_size = input.shape
            input_lengths = [max_seq_length] * cur_batch_size

        output = input
        for layer in self.enc_fnn_layers:
            output = layer(output)

        output_lengths = input_lengths
        for i in range(len(self.enc_rnn_layers)):
            layer = self.enc_rnn_layers[i]
            packed_sequence = pack(output, output_lengths, batch_first=True)
            output, _ = layer(packed_sequence) # LSTM returns '(output, [hn ,cn])'
            output, _ = unpack(output, batch_first=True, padding_value=self.enc_input_padding_value) # unpack returns (data, length)
            # dropout of lstm module behaves randomly even with same torch seed, so we'll append dropout layer.
            output = F.dropout(output, p=self.enc_rnn_dropout[i], training=self.training)

            # Subsampling by taking the first frame or concatenating frames for every two frames.
            if (self.enc_rnn_subsampling[i]):

                # Padding the max_seq_length be a multiple of 2 (even number) for subsampling.
                # ps: That padding frames with a multiple of 8 (with 3 times of subsampling) before inputting to the rnn
                #     equals to that padding 3 times in the middle of layers with a multiple of 2,
                #     because of the pack and unpack operation only takes feature with effective lengths to rnn.
                if (output.shape[1] % 2 != 0): # odd length
                    extended_part = torch.ones(output.shape[0], 1, output.shape[2], device = output.device) * self.enc_input_padding_value
                    output = torch.cat([output, extended_part], dim=1) # pad to be even length

                if (self.enc_rnn_subsampling_type[i] == 'pair_take_first'):
                    output = output[:, ::2]
                    output_lengths = torch.LongTensor([(length + (2 - 1)) // 2 for length in output_lengths])
                elif (self.enc_rnn_subsampling_type[i] == 'pair_concat'):
                    output = output.view(output.shape[0], output.shape[1] // 2, output.shape[2] * 2)
                    output_lengths = torch.LongTensor([(length + (2 - 1)) // 2 for length in output_lengths])
                else:
                    raise NotImplementedError("The subsampling type {} is not implemented yet.\n".format(self.enc_rnn_subsampling_type[i]) +
                                              "Only support the type 'pair_concat' and 'pair_take_first':\n" +
                                              "The type 'pair_take_first' takes the first frame every two frames.\n" +
                                              "The type 'pair_concat' concatenates the frame pair every two frames.\n")

            print("After layer '{}' applying the subsampling '{}' with type '{}': shape is {}, lengths is {} ".format(
                i, self.enc_rnn_subsampling[i], self.enc_rnn_subsampling_type[i], output.shape, output_lengths))
            print("mask of lengths is\n{}".format(length2mask(output_lengths)))

        context, context_mask = output, length2mask(output_lengths)
        return context, context_mask

input = next(iter(dataloader))['feat']
input_lengths = next(iter(dataloader))['num_frames'] # or None
# input_lengths = torch.LongTensor([7, 2, 1])
# input = batches[1]['feat']
# input_lengths = batches[1]['num_frames']
in_size = input.shape[-1]

speech_encoder = SpeechEncoder(in_size,
                               enc_fnn_sizes = [4, 9],
                               enc_fnn_act = 'ReLU',
                               enc_fnn_dropout = 0.25,
                               enc_rnn_sizes = [5, 5, 5],
                               enc_rnn_config = {'type': 'lstm', 'bi': True},
                               enc_rnn_dropout = 0.25,
                               enc_rnn_subsampling = [False, True, True],
                               enc_rnn_subsampling_type = 'pair_concat')

speech_encoder.get_config()
context, context_mask = speech_encoder.encode(input, input_lengths)
# print(context.shape, context_mask)

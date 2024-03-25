# Implemented by bin-wu at 10:20 on 12 April 2020
# Heavily inspired by andros's code

# # Training
# train_asr()
# # Evaluation
# recog_asr()
# # Compute WER
# /project/nakamura-lab08/Work/bin-wu/share/tools/kaldi/src/bin/compute-wer --mode=present ark,t:exp/tmp/test_small_att/eval/ref_char.txt ark,t:exp/tmp/test_small_att/eval/hypo_char.txt
# /project/nakamura-lab08/Work/bin-wu/share/tools/kaldi/src/bin/compute-wer --mode=present ark,t:exp/tmp/test_small_att/eval/ref_word.txt ark,t:exp/tmp/test_small_att/eval/hypo_word.txt

from typing import List, Dict, Tuple, Union, Any

import sys
import os
import shutil
import glob
import time
import logging

import argparse
import math
import re # parser class name
import json # for data files
import pprint
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack, pad_sequence

# pip install *
import yaml # for config files
import tabulate
import tqdm

# Add clib package at current directory to path.
sys.path.append(os.getcwd())
from clib.kaldi.kaldi_data import KaldiDataLoader, KaldiDataset

def get_rnn(name):
    """ Get the RNN module by its name string.
    We can write module name directly in the configuration file.
    We can also manage the already registered rnns or add the new custom rnns.
    The name can be "LSTM", 'lstm' and etc.

    Example
    -------
    In [1]: lstm = get_rnn('lstm')(2, 5)
    In [2]: result, _ = lstm(torch.Tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]))
    In [3]: result.shape
    Out[3]: torch.Size([2, 3, 5])
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

    Example
    -------
    In [1]: relu = get_act('relu')()
    In [2]: relu(torch.Tensor([-1, 2]))
    Out[2]: tensor([0., 2.])
    """
    registered_act = {"relu": torch.nn.ReLU,
                      "relu6": torch.nn.ReLU6,
                      "elu": torch.nn.ELU,
                      "prelu": torch.nn.PReLU,
                      "leaky_relu": torch.nn.LeakyReLU,
                      "threshold": torch.nn.Threshold,
                      "hardtanh": torch.nn.Hardtanh,
                      "sigmoid": torch.nn.Sigmoid,
                      "tanh": torch.nn.Tanh,
                      "log_sigmoid": torch.nn.LogSigmoid,
                      "softplus": torch.nn.Softplus,
                      "softshrink": torch.nn.Softshrink,
                      "softsign": torch.nn.Softsign,
                      "tanhshrink": torch.nn.Tanhshrink}

    avaliable_act = list(registered_act.keys())

    if name.lower() in registered_act:
        return registered_act[name.lower()]
    else:
        raise NotImplementedError("The act module '{}' is not implemented\nAvaliable act modules include {}".format(name, avaliable_act))

def get_att(name):
    """ Get attention module by name string.
    The name can be 'dot_product', 'mlp' and etc.

    Example
    -------
    In [347]: query = torch.Tensor([[3, 4], [3, 5]])
    In [348]: context = torch.Tensor([[[3, 4], [4, 3]], [[3, 5], [3, 4]]])
    In [349]: mask = torch.ByteTensor([[1, 0],[1, 1]])
    In [350]: input = {'query': query, 'context': context, 'mask': mask}
    In [351]: attention = get_att('mlp')(context.shape[-1], query.shape[-1])
    In [353]: attention(input)
    Out[353]:{'p_context': tensor([[0.4973, 0.5027], [0.5185, 0.4815]], grad_fn=<SoftmaxBackward>),
       'expected_context': tensor([[3.5027, 3.4973], [3.0000, 4.5185]], grad_fn=<SqueezeBackward1>)}
    """
    registered_att = {'dot_product': DotProductAttention,
                      'mlp': MLPAttention}

    avaliable_att = list(registered_att.keys())

    if name.lower() in registered_att:
        return registered_att[name.lower()]
    else:
        raise NotImplementedError("The att module '{}' is not implemented\nAvaliable att modules include {}".format(name, avaliable_att))

def get_optim(name):
    """ Get optimizer by name string.
    The name can be 'adam', 'sgd' and etc.

    Example
    -------
    In [350]: model=nn.Linear(2, 3)
    In [351]: optimizer = get_optim('adam')(model.parameters(), lr=0.005)
    """
    registered_optim = {"adam": torch.optim.Adam,
                        "sgd": torch.optim.SGD,
#                        "adamw": torch.optim.AdamW,
                        "sparse_adam": torch.optim.SparseAdam,
                        "adagrad": torch.optim.Adagrad,
                        "adadelta": torch.optim.Adadelta,
                        "rmsprop": torch.optim.RMSprop,
                        "adamax": torch.optim.Adamax,
                        "averaged_sgd": torch.optim.ASGD}

    avaliable_optim = list(registered_optim.keys())

    if name.lower() in registered_optim:
        return registered_optim[name.lower()]
    else:
        #raise NotImplementedError("The optim module '{}' is not implemented\n".format(name) +
        #                          "Avaliable optim modules include {}".format(avaliable_optim))
        raise NotImplementedError(f"The optim module '{name}' is not implemented\n"
                                  f"Avaliable optim modules include {avaliable_optim}")

def length2mask(sequence_lengths: torch.Tensor, max_length: Union[int, None] = None) -> torch.Tensor:
    """
    Given a variable of shape ``(batch_size,)`` that represents the sequence lengths of each batch
    element, this function returns a ``(batch_size, max_length)`` mask variable.  For example, if
    our input was ``[2, 2, 3]``, with a ``max_length`` of 4, we'd return
    ``[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]]``.

    Examples
    --------
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
        max_length = int(torch.max(sequence_lengths).item())
    ones_seqs = sequence_lengths.new_ones(len(sequence_lengths), max_length)
    cumsum_ones = ones_seqs.cumsum(dim=-1)

    return (cumsum_ones <= sequence_lengths.unsqueeze(-1)).long()

def mask2length(mask: torch.Tensor) -> torch.LongTensor:
    """
    Compute sequence lengths for the batch from a binary mask.

    Parameters
    ----------
    mask: a binary mask of shape [batch_size, sequence_length]

    Returns
    -------
    the lengths of the sequences in the batch of shape [batch_size]

    Example
    -------
    In [458]: mask
    Out[458]:
    tensor([[1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 0]])

    In [459]: mask2length(mask)
    Out[459]: tensor([2, 2, 3])
    """
    return mask.long().sum(-1)

class CrossEntropyLossLabelSmoothing(nn.Module):
    """ Cross entropy loss function support label smoothing and weight of classes.

    https://arxiv.org/abs/1512.00567 Section 7. Model Regularization via Label Smoothing
    smoothed_loss = (1 - label_smoothing) * H(label_prob, model_prob) + label_smoothing * H(label_prob, uniform_prob)

    Cross entropy between two true distribution 'prob1' and model distribution 'prob2'
    (https://en.wikipedia.org/wiki/Cross_entropy)
    H(prob1, prob2) = -sum(prob1 *log(prob2))

    Paramters
    ---------
    label_smoothing: ratio smoothed by the uniform distribution
    weight: weight of the each type of class; shape (num_classes) (e.g., setting the weight zero when target is the padding label)
    reduction: 'mean' or 'sum' or 'none'; take the 'mean' or 'sum' of loss over batch, or return loss per batch if 'none'.

    Paramters of forward function
    -----------------------------
    source: shape (batch_size, num_classes) (or (batch_size * seq_length, num_classes))
    target: shape (batch_size) or (batch_size * seq_length)

    Returns of forward function
    ---------------------------
    loss: shape (batch_size) or (batch_size * seq_length) if reduction is 'none'
    or shape () if reduction is 'mean' or 'sum'

    Example
    -------
    Input:
    source = torch.Tensor([[0.9, 0.2, 0.3], [0.1, 0.9, 0.3], [0.9, 0.2, 0.3]])
    target = torch.LongTensor([1, 2, 1])

    label_smoothing = 0.8
    weight = torch.Tensor([0.1, 0.5, 0.4])
    reduction = 'none'

    ce_loss = CrossEntropyLossLabelSmoothing(label_smoothing=label_smoothing, weight=weight, reduction=reduction)
    print(ce_loss(source, target))

    Output:
    tensor([0.6011, 0.4742, 0.6011])
    """
    def __init__(self,
                 label_smoothing: float = 0.,
                 weight: Union[torch.Tensor, None] = None,
                 reduction: str = 'mean') -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        self.weight = weight
        self.reduction = reduction

        assert reduction == 'sum' or reduction == 'mean' or reduction == 'none', \
            "unknown return eduction '{}', reduction should be 'none', 'sum' or 'mean'".format(reduction)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_model_prob = torch.nn.functional.log_softmax(source, dim=-1) # batch_size x num_classes
        cross_entropy_label_and_model = -log_model_prob.gather(dim=1, index=target.unsqueeze(1)).squeeze(1) # [batch_size]; ce per batch

        if(self.label_smoothing > 0):
            num_classes = source.shape[-1]
            # sometimes '1/(num_classes-2)' to exclude <sos> and <eos>
            uniform_prob = torch.ones_like(source) * (1 / num_classes) # [batch_size * num_classes]
            cross_entropy_uniform_and_model = -(log_model_prob * uniform_prob).sum(dim=-1) # [batch_size]; cross entropy per batch
            cross_entropy_mixed = (1 - self.label_smoothing) * cross_entropy_label_and_model + \
                                  self.label_smoothing * cross_entropy_uniform_and_model
        else:
            cross_entropy_mixed = cross_entropy_label_and_model # shape of (batch_size)

        if self.weight is not None:
            cross_entropy_mixed = cross_entropy_mixed * self.weight.index_select(dim=0, index=target) # shape of (batch_size)

        if (self.reduction == 'none'):
            return cross_entropy_mixed
        elif (self.reduction == 'sum'):
            return cross_entropy_mixed.sum(dim=-1)
        else:
            return cross_entropy_mixed.mean(dim=-1)

def test_cross_entropy_label_smooth():
    seed = 2020
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: '{}'".format(device))

    source = torch.Tensor([[0.9, 0.2, 0.3], [0.1, 0.9, 0.3], [0.9, 0.2, 0.3]])
    target = torch.LongTensor([1, 2, 1])

    label_smoothing = 0.8
    weight = torch.Tensor([0.1, 0.5, 0.4])
    reduction = 'none'

    source, target, weight = source.to(device), target.to(device), weight.to(device)
    ce_loss = CrossEntropyLossLabelSmoothing(label_smoothing=label_smoothing, weight=weight, reduction=reduction)
    ce_loss.to(device)

    A = ce_loss(source, target)
    B = torch.Tensor([0.6011, 0.4742, 0.6011]).to(device)
    print(A)
    print(B)
    assert torch.all(torch.lt(torch.abs(torch.add(A, -B)), 1e-4)) # A == B

class PyramidRNNEncoder(nn.Module):
    """ The RNN encoder with support of subsampling (for input with long length such as speech feature).
    https://arxiv.org/abs/1508.01211 "LAS" section 3.1 formula (5)

    The PyramidRNNEncoder accepts the feature (batch_size x max_seq_length x in_size),
    passes the feature to several layers of feedforward neural network (fnn)
    and then to several layers of RNN (rnn) with subsampling
    (by concatenating every pair of frames with type 'concat'
    or by taking the first frame every frame pair with the type 'drop').

    ps: We can pass the parameters by copying the same configuration to each layer
        or by specifying a list of configurations for each layer.
        We do padding at the end of sequence whenever subsampling needs more frames to concatenate.
    """

    def __init__(self,
                 enc_input_size: int,
                 enc_fnn_sizes: List[int] = [512],
                 enc_fnn_act: str = 'relu',
                 enc_fnn_dropout: Union[float, List[float]] = 0.25,
                 enc_rnn_sizes: List[int] = [256, 256, 256],
                 enc_rnn_config: Dict = {'type': 'lstm', 'bi': True},
                 enc_rnn_dropout: Union[float, List[float]] = 0.25,
                 enc_rnn_subsampling: Union[bool, List[bool]] = [False, True, True],
                 enc_rnn_subsampling_type: str = 'concat', # 'concat' or 'drop'
                 enc_input_padding_value: float = 0.0
    ) -> None:
        super().__init__()

        # make copy of the configuration for each layer.
        num_enc_fnn_layers = len(enc_fnn_sizes)
        if not isinstance(enc_fnn_dropout, list): enc_fnn_dropout = [enc_fnn_dropout] * num_enc_fnn_layers

        num_enc_rnn_layers = len(enc_rnn_sizes)
        if not isinstance(enc_rnn_dropout, list): enc_rnn_dropout = [enc_rnn_dropout] * num_enc_rnn_layers
        if not isinstance(enc_rnn_subsampling, list): enc_rnn_subsampling = [enc_rnn_subsampling] * num_enc_rnn_layers

        assert num_enc_fnn_layers == len(enc_fnn_dropout), \
            "Number of fnn layers does not match the lengths of specified configuration lists."
        assert num_enc_rnn_layers == len(enc_rnn_dropout) == len(enc_rnn_subsampling), \
            "Number of rnn layers does not matches the lengths of specificed configuration lists."
        assert enc_rnn_subsampling_type in {'concat', 'drop'}, \
            "The subsampling type '{}' is not implemented yet.\n".format(t) + \
            "Only support the type 'concat' and 'drop':\n" + \
            "the type 'drop' preserves the first frame every two frames;\n" + \
            "the type 'concat' concatenates the frame pair every two frames.\n"

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

        self.num_enc_fnn_layers = num_enc_fnn_layers
        self.num_enc_rnn_layers = num_enc_rnn_layers

        pre_size = self.enc_input_size
        self.enc_fnn_layers = nn.ModuleList()
        for i in range(num_enc_fnn_layers):
            self.enc_fnn_layers.append(nn.Linear(pre_size, enc_fnn_sizes[i]))
            self.enc_fnn_layers.append(get_act(enc_fnn_act)())
            self.enc_fnn_layers.append(nn.Dropout(p=enc_fnn_dropout[i]))
            pre_size = enc_fnn_sizes[i]

        self.enc_rnn_layers = nn.ModuleList()
        for i in range(num_enc_rnn_layers):
            rnn_layer = get_rnn(enc_rnn_config['type'])
            self.enc_rnn_layers.append(rnn_layer(pre_size, enc_rnn_sizes[i], batch_first=True, bidirectional=enc_rnn_config['bi']))
            pre_size = enc_rnn_sizes[i] * (2 if enc_rnn_config['bi'] else 1) # for bidirectional rnn
            if (enc_rnn_subsampling[i] and enc_rnn_subsampling_type == 'concat'): pre_size = pre_size * 2 # for concat subsampling

        self.output_size = pre_size

    def get_context_size(self) -> int:
        return self.output_size

    def encode(self,
               input: torch.Tensor,
               input_lengths: Union[torch.Tensor, None] = None,
               verbose: bool = False) -> (torch.Tensor, torch.Tensor):
        """ Encode the feature (batch_size x max_seq_length x in_size), optionally with its lengths (batch_size),
        and output the context vector (batch_size x max_seq_length' x context_size)
        with its mask (batch_size x max_seq_length').

        ps: the dimension of context vector is influenced by 'bidirectional' and 'subsampling (concat)' options of RNN.
            the max_seq_length' influenced by 'subsampling' options of RNN.
        """
        if (input_lengths is None):
            cur_batch_size, max_seq_length, cur_input_size = input.shape
            input_lengths = [max_seq_length] * cur_batch_size

        output = input
        for layer in self.enc_fnn_layers:
            output = layer(output)

        output_lengths = input_lengths
        for i in range(self.num_enc_rnn_layers):
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
                    extended_part = output.new_ones(output.shape[0], 1, output.shape[2]) * self.enc_input_padding_value
                    output = torch.cat([output, extended_part], dim=-2) # pad to be even length

                if (self.enc_rnn_subsampling_type == 'drop'):
                    output = output[:, ::2]
                    output_lengths = torch.LongTensor([(length + (2 - 1)) // 2 for length in output_lengths]).to(output.device)
                elif (self.enc_rnn_subsampling_type == 'concat'):
                    output = output.contiguous().view(output.shape[0], output.shape[1] // 2, output.shape[2] * 2)
                    output_lengths = torch.LongTensor([(length + (2 - 1)) // 2 for length in output_lengths]).to(output.device)
                else:
                    raise NotImplementedError("The subsampling type {} is not implemented yet.\n".format(self.enc_rnn_subsampling_type) +
                                              "Only support the type 'concat' and 'drop':\n" +
                                              "The type 'drop' takes the first frame every two frames.\n" +
                                              "The type 'concat' concatenates the frame pair every two frames.\n")
            if verbose:
                print("After layer '{}' applying the subsampling '{}' with type '{}': shape is {}, lengths is {} ".format(
                    i, self.enc_rnn_subsampling[i], self.enc_rnn_subsampling_type, output.shape, output_lengths))
                print("mask of lengths is\n{}".format(length2mask(output_lengths)))

        context, context_mask = output, length2mask(output_lengths)
        return context, context_mask

def test_encoder():
    seed = 2020
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    json_file = 'conf/data/test_small/utts.json'
    padding_tokenid = 1 # follow torchtext

    with open(json_file, encoding='utf8') as f:
        utts_json = json.load(f)
        utts_instances = list(utts_json.values())

        dataset = KaldiDataset(utts_instances)
        dataloader = KaldiDataLoader(dataset=dataset, batch_size=3, padding_tokenid=padding_tokenid)

    batches = []
    for batch in dataloader:
        batches.append(batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: '{}'".format(device))

    input = next(iter(dataloader))['feat']
    input_lengths = next(iter(dataloader))['num_frames'] # or None
    # input_lengths = torch.LongTensor([7, 2, 1])
    # input = batches[1]['feat']
    # input_lengths = batches[1]['num_frames']
    enc_input_size = input.shape[-1]

    speech_encoder = PyramidRNNEncoder(enc_input_size,
                                       enc_fnn_sizes = [4, 9],
                                       enc_fnn_act = 'ReLU',
                                       enc_fnn_dropout = 0.25,
                                       enc_rnn_sizes = [5, 5, 5],
                                       enc_rnn_config = {'type': 'lstm', 'bi': True},
                                       enc_rnn_dropout = 0.25,
                                       enc_rnn_subsampling = [False, True, True],
                                       enc_rnn_subsampling_type = 'concat')

    speech_encoder.to(device)
    input, input_lengths = input.to(device), input_lengths.to(device)
    context, context_mask = speech_encoder.encode(input, input_lengths, verbose=True)
    # print(context.shape, context_mask)

class DotProductAttention(nn.Module):
    """  Attention by dot product.
    https://arxiv.org/abs/1508.04025 "Effective MNT" section 3.1 formula (8) (dot version)

    DotProductAttention is a module that takes in a dict with key of 'query' and 'context'
    (alternative key of 'mask' and 'need_expected_context'),
    and returns a output dict with key ('p_context' and 'expected_context').

    It takes 'query' (batch_size x query_size) and 'context' (batch_size x context_length x context_size),
    returns the proportion of attention ('p_context': batch_size x context_length) the query pays to different parts of context
    and the expected context vector ('expected_context': batch_size x context_size)
    by taking weighted average over the context by the proportion of attention.

    Example
    -------
    Input:
    query = torch.Tensor([[3, 4], [3, 5]])
    context = torch.Tensor([[[3, 4], [4, 3]], [[3, 5], [3, 4]]])
    mask = torch.ByteTensor([[1, 1],[1, 0]])
    input = {'query': query, 'context': context, 'mask': mask}

    attention = DotProductAttention()
    output = attention(input)

    Output:
    {'p_context': tensor([[0.7311, 0.2689], [0.9933, 0.0067]]),
    'expected_context': tensor([[3.2689, 3.7311], [3.0000, 4.9933]])}
    """

    def __init__(self,
                 context_size: int = -1,
                 query_size: int = -1,
                 normalize: bool = False) -> None:
        super().__init__()
        self.normalize = normalize
        self.att_vector_size = context_size

    def compute_expected_context(self, p_context: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """ compute the expected context by taking the weighted (p_context) average.

        p_context: batch_size x context_length
        context: batch_size x context_length x context_size
        expected_context: batch_size x context_size
        """
        return torch.bmm(p_context.unsqueeze(-2), context).squeeze(-2)

    def forward(self, input: Dict) -> Dict:
        query = input['query'] # batch_size x query_size
        context = input['context'] # batch_size x context_length x context_size
        assert query.shape[-1] == context.shape[-1], \
            "The query_size ({}) and context_size ({}) need to be same for the DotProductAttention.".format(
                query.shape[-1], context.shape[-1])
        mask = input.get('mask', None)
        need_expected_context = input.get('need_expected_context', True)

        # score = dot_product(context,query) formula (8) of "Effective MNT".
        score = torch.bmm(context, query.unsqueeze(-1)).squeeze(-1) # batch_size x context_length
        if self.normalize: score = score / math.sqrt(query_size)
        if mask is not None: score.masked_fill_(mask==0, -1e9)
        p_context = F.softmax(score, dim=-1)
        expected_context = self.compute_expected_context(p_context, context) if need_expected_context else None
        return {'p_context': p_context,
                'expected_context': expected_context}

class MLPAttention(nn.Module):
    """  Attention by multilayer perception (mlp).
    https://arxiv.org/abs/1508.04025 "Effective MNT" section 3.1 formula (8) (concat version)

    score = V*tanh(W[context,query])

    MLPAttention will concatenate the query and context and pass them through two-layer mlp to get the probability (attention) over context.
    It is a module that takes in a dict with key of 'query' and 'context' (alternative key of 'mask' and 'need_expected_context'),
    and returns a output dict with key ('p_context' and 'expected_context').

    It takes 'query' (batch_size x query_size) and 'context' (batch_size x context_length x context_size),
    returns the proportion of attention ('p_context': batch_size x context_length) the query pays to different parts of context
    and the expected context vector ('expected_context': batch_size x context_size)
    by taking weighted average over the context by the proportion of attention.

    Example
    -------
    Input:
    query = torch.Tensor([[3, 4], [3, 5]])
    context = torch.Tensor([[[3, 4], [4, 3]], [[3, 5], [3, 4]]])
    mask = torch.ByteTensor([[1, 1],[1, 0]])
    input = {'query': query, 'context': context, 'mask': mask}

    attention = MLPAttention(context.shape[-1], query.shape[-1])
    output = attention(input)

    Output:
    {'p_context': tensor([[0.4997, 0.5003], [0.4951, 0.5049]], grad_fn=<SoftmaxBackward>),
    'expected_context': tensor([[3.5003, 3.4997], [3.0000, 4.4951]], grad_fn=<SqueezeBackward1>)}
    """

    def __init__(self,
                 context_size: int,
                 query_size: int,
                 att_hidden_size: int = 256,
                 att_act: str = 'tanh',
                 normalize: bool = True) -> None:
        super().__init__()
        self.concat2proj = nn.Linear(query_size+context_size, att_hidden_size) # W in formula (8) of "Effective MNT"
        self.att_act = get_act(att_act)()
        self.proj2score = nn.utils.weight_norm(nn.Linear(att_hidden_size, 1)) if normalize \
                          else nn.Linear(att_hidden_size, 1) # V in formula (8) of "Effective MNT"
        self.att_vector_size = context_size

    def compute_expected_context(self, p_context: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """ compute the expected context by taking the weighted (p_context) average.

        p_context: batch_size x context_length
        context: batch_size x context_length x context_size
        expected_contex: batch_size x context_size
        """
        return torch.bmm(p_context.unsqueeze(-2), context).squeeze(-2)

    def forward(self, input: Dict) -> Dict:
        query = input['query'] # batch_size x query_size
        batch_size, query_size = query.shape
        context = input['context'] # batch_size x context_length x context_size
        batch_size, context_length, context_size = context.shape
        mask = input.get('mask', None)
        need_expected_context = input.get('need_expected_context', True)

        # score = V*tanh(W[context,query]) formula (8) of "Effective MNT".
        concat = torch.cat([context, query.unsqueeze(-2).expand(batch_size, context_length, query_size)], dim=-1) # batch_size x context_length x (context_size + query_size)
        score = self.proj2score(self.att_act(self.concat2proj(concat))).squeeze(-1) # batch_size x context_length

        if mask is not None: score.masked_fill_(mask==0, -1e9)
        p_context = F.softmax(score, dim=-1)
        expected_context = self.compute_expected_context(p_context, context) if need_expected_context else None # batch_size x context_size
        return {'p_context': p_context,
                'expected_context': expected_context}

def test_attention() :
    seed = 2020
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: '{}'".format(device))

    query = torch.Tensor([[3, 4], [3, 5]])
#    query = torch.Tensor([[[3, 4], [4, 3]], [[3, 5], [3, 4]]])
    context = torch.Tensor([[[3, 4], [4, 3]], [[3, 5], [3, 4]]])
    mask = torch.ByteTensor([[1, 1],[1, 0]])
    input = {'query': query.to(device), 'context': context.to(device), 'mask': mask.to(device)}

    attention = DotProductAttention(context.shape[-1], query.shape[-1])
    attention.to(device)
    output = attention(input)
    print(output)

    attention = MLPAttention(context.shape[-1], query.shape[-1])
    attention.to(device)
    output = attention(input)
    print(output)

class LuongDecoder(nn.Module):
    """ Implementation of the decoder of "Effective NMT" by Luong.
    https://arxiv.org/abs/1508.04025 "Effective MNT"
    section 3 formula (5) to create attentional vector
    section 3.3 the input feeding approach

    At time step t of decoder, the message flows as follows
    [attentional_vector[t-1], input] -> hidden[t] ->
    [expected_context_vector[t], hidden[t]] -> attentional_vector[t]

    Input feeding: concatenate the input with the attentional vector from
    last time step to combine the alignment information in the past.

    attentional vector: we get hidden state at the top the stacked LSTM layers,
    then concatenate the hidden state and expected context vector for linearly
    projecting (context_proj_*) to the attentional vector.
    see "Effective NMT" section 3 formula (5)
        attentional_vector = tanh(W[context_vector, hidden])

    Example
    -------
    In:
    input_embedding = torch.Tensor([[0.3, 0.4], [0.3, 0.5]])
    context = torch.Tensor([[[3, 4], [4, 3]], [[3, 5], [3, 4]]])
    context_mask = torch.ByteTensor([[1, 1],[1, 0]])

    luong_decoder = LuongDecoder(att_config={"type": "mlp"},
                                 context_size=context.shape[-1],
                                 input_size=input_embedding.shape[-1],
                                 rnn_sizes=[512, 512],
                                 rnn_config={"type": "lstmcell"},
                                 rnn_dropout=0.25,
                                 context_proj_size=3, # the size of attentional vector
                                 context_proj_act='tanh')

    luong_decoder.set_context(context, context_mask)
    output, att_out = luong_decoder(input_embedding)
    # output, att_out = luong_decoder(input_embedding, dec_mask=torch.Tensor([0, 1])) # mask the first instance in two batches
    print("output of Luong decoder: {}".format(output))
    print("output of attention layer: {}".format(att_out))

    Out:
    output of Luong decoder: tensor([[0.0268, 0.0782, 0.0374], [0.0285, 0.1341, 0.0169]], grad_fn=<TanhBackward>)
    output of attention layer: {'p_context': tensor([[0.4982, 0.5018], [0.4990, 0.5010]], grad_fn=<SoftmaxBackward>),
                         'expected_context': tensor([[3.5018, 3.4982], [3.0000, 4.4990]], grad_fn=<SqueezeBackward1>)}
    """

    def __init__(self,
                 att_config: Dict, # configuration of attention
                 context_size: int,
                 input_size: int,
                 rnn_sizes: List[int] = [512, 512],
                 rnn_config: Dict = {"type": "lstmcell"},
                 rnn_dropout: Union[List[float], float] = 0.25,
                 context_proj_size: int = 256, # the size of attentional vector
                 context_proj_act: str = 'tanh',
                 context_proj_dropout: int = 0.25) -> None:
        super().__init__()

        # Copy the configuration for each layer
        num_rnn_layers = len(rnn_sizes)
        if not isinstance(rnn_dropout, list): rnn_dropout = [rnn_dropout] * num_rnn_layers # sometimes dropout not at the top layer
        assert num_rnn_layers == len(rnn_dropout), "The number of rnn layers does not match length of rnn_dropout list."

        self.att_config = att_config
        self.context_size = context_size
        self.input_size = input_size
        self.rnn_sizes = rnn_sizes
        self.rnn_config = rnn_config
        self.rnn_dropout = rnn_dropout
        self.context_proj_size = context_proj_size
        self.context_proj_act = context_proj_act
        self.context_proj_dropout = context_proj_dropout

        self.num_rnn_layers = num_rnn_layers

        # Initialize attentional vector of previous time step
        self.attentional_vector_pre = None

        # Initialize stacked rnn layers with their hidden states and cell states
        self.rnn_layers = nn.ModuleList()
        self.rnn_hidden_cell_states = []
        pre_size = input_size + context_proj_size # input feeding
        for i in range(num_rnn_layers):
            self.rnn_layers.append(get_rnn(rnn_config['type'])(pre_size, rnn_sizes[i]))
            self.rnn_hidden_cell_states.append(None) # initialize (hidden state, cell state) of each layer as Nones.
            pre_size = rnn_sizes[i]

        # Get expected context vector from attention
        self.attention_layer = get_att(att_config['type'])(context_size, pre_size)

        # Combine hidden state and context vector to be attentional vector.
        self.context_proj_layer = nn.Linear(pre_size + context_size, context_proj_size)

        self.output_size = context_proj_size

    def set_context(self, context: torch.Tensor, context_mask: Union[torch.Tensor, None] = None) -> None:
        self.context = context
        self.context_mask = context_mask

    def get_context_and_its_mask(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.context, self.context_mask

    def reset(self) -> None:
        """ Reset the the luong decoder
        by setting the attentional vector of the previous time step to be None,
        which means forgetting all the formation
        accumulated in the history (by RNN) before the current time step
        and forgetting the attention information of the previous time step.
        """
        self.attentional_vector_pre = None
        for i in range(self.num_rnn_layers):
            self.rnn_hidden_cell_states[i] = None

    def decode(self, input: torch.Tensor, dec_mask: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, Dict]:
        """
        input: batch_size x input_size
        dec_mask: batch_size
        # target batch 3 with length 2, 1, 3 => mask = [[1, 1, 0], [1, 0, 0], [1, 1, 1]]
        # Each time step corresponds to each column of the mask.
        # In time step 2, the second column [1, 0, 1] as the dec_mask
        # dec_mask with shape [batch_size]
        # target * dec_mask.unsqueeze(-1).expand_as(target) will mask out
        # the feature of the second element of batch at time step 2, while the element with length 1
        """
        batch_size, input_size = input.shape
        if self.attentional_vector_pre is None:
            self.attentional_vector_pre = input.new_zeros(batch_size, self.context_proj_size)

        # Input feeding: initialize the input of LSTM with previous attentional vector information
        output = torch.cat([input, self.attentional_vector_pre], dim=-1)
        for i in range(self.num_rnn_layers):
            output, cell = self.rnn_layers[i](output, self.rnn_hidden_cell_states[i]) # LSTM cell return (h, c)
            self.rnn_hidden_cell_states[i] = (output, cell) # store the hidden state and cell state of current layer for next time step.
            if dec_mask is not None: output = output * dec_mask.unsqueeze(-1).expand_as(output)

            output = F.dropout(output, p=self.rnn_dropout[i], training=self.training)

        # Get the context vector from the hidden state at the top of rnn layers.
        att_out = self.attention_layer({'query': output, 'context': self.context, 'mask': self.context_mask})

        # Get the attentional vector of current time step by linearly projection from hidden state and context vector
        act_func = get_act(self.context_proj_act)()
        output = act_func(self.context_proj_layer(torch.cat([output, att_out['expected_context']], dim = -1)))

        if dec_mask is not None: output = output * dec_mask.unsqueeze(-1).expand_as(output)

        self.attentional_vector_pre = output # attentional vector before dropout might be more stable

        output = F.dropout(output, p=self.context_proj_dropout, training=self.training)

        return output, att_out

def test_luong_decoder():
    seed = 2020
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: '{}'".format(device))

    input_embedding = torch.Tensor([[0.3, 0.4], [0.3, 0.5]])
    context = torch.Tensor([[[3, 4], [4, 3]], [[3, 5], [3, 4]]])
    context_mask = torch.ByteTensor([[1, 1], [1, 0]])

    luong_decoder = LuongDecoder(att_config={"type": "mlp"},
                                 context_size=context.shape[-1],
                                 input_size=input_embedding.shape[-1],
                                 rnn_sizes=[512, 512],
                                 rnn_config={"type": "lstmcell"},
                                 rnn_dropout=0.25,
                                 context_proj_size=3, # the size of attentional vector
                                 context_proj_act='tanh')
    input_embedding, context, context_mask = input_embedding.to(device), context.to(device), context_mask.to(device)
    luong_decoder.to(device)
    luong_decoder.set_context(context, context_mask)
    output, att_out = luong_decoder.decode(input_embedding)
    # output, att_out = luong_decoder(input_embedding, dec_mask=torch.Tensor([0, 1])) # mask the first instance in two batches
    print("output of Luong decoder: {}".format(output))
    print("output of attention layer: {}".format(att_out))

class EncRNNDecRNNAtt(nn.Module):
    """ Sequence-to-sequence module with RNN encoder and RNN decoder with attention mechanism.

    The default encoder is pyramid RNN Encoder (https://arxiv.org/abs/1508.01211 "LAS" section 3.1 formula (5)),
    which passes the input feature through forward feedback neural network ('enc_fnn_*') and RNN ('enc_rnn_*').
    Between the RNN layers we use the subsampling ('enc_rnn_subsampling_*')
by concatenating both frames or taking the first frame every two frames.

    The default decoder is Luong Decoder (https://arxiv.org/abs/1508.04025 "Effective MNT"
    section 3 formula (5) to create attentional vector; section 3.3 the input feeding approach)
    which passes the embedding of the input ('dec_embedding_*')
    along with previous attentional vector into a stacked RNN layers ('dec_rnn_*'),
    linearly projects the top RNN hidden state and expected context vector to the current attentional vector ('dec_context_proj_*'),
    and feed the attentional vector to next time step.

    The default attention is the multilayer perception attention by concatenation
    (https://arxiv.org/abs/1508.04025 "Effective MNT" section 3.1 formula (8) (concat version))
    which concatenates the hidden state from decoder and each part of context from encoder
    and passes them to two layers neural network to get the alignment score.
    The scores are then normalized to get the probability of different part of context and get the expected context vector.

    We tie the embedding weight by default
    (https://arxiv.org/abs/1608.05859 "Using the Output Embedding to Improve Language Models" introduction),
    which shares the weights between (dec_onehot => dec_embedding) and (output_embedding(attentional vector) => pre_softmax)

    Information flow:
    encode:
    enc_input->encoder->context

    decode with one time step:
    dec_input->dec_embedding->dec_hidden
    (context,dec_hidden)->context_vector
    (context_vector,dec_hidden)->attentional_vector
    attentional_vector->pre_softmax
    """
    def __init__(self,
                 enc_input_size: int,
                 dec_input_size: int,
                 dec_output_size: int,
                 enc_fnn_sizes: List[int] = [512],
                 enc_fnn_act: str = 'relu',
                 enc_fnn_dropout: Union[float, List[float]] = 0.25,
                 enc_rnn_sizes: List[int] = [256, 256, 256],
                 enc_rnn_config: Dict = {'type': 'lstm', 'bi': True},
                 enc_rnn_dropout: Union[float, List[float]] = 0.25,
                 enc_rnn_subsampling: Union[bool, List[bool]] = [False, True, True],
                 enc_rnn_subsampling_type: str = 'concat', # 'concat' or 'drop'
                 dec_embedding_size: int = 256,
                 dec_embedding_dropout: float = 0.25,
                 # share weights between (input_onehot => input_embedding) and (output_embedding => pre_softmax)
                 dec_embedding_weights_tied: bool = True,
                 dec_rnn_sizes: List[int] = [512, 512],
                 dec_rnn_config: Dict = {"type": "lstmcell"},
                 dec_rnn_dropout: Union[List[float], float] = 0.25,
                 dec_context_proj_size: int = 256, # the size of attentional vector
                 dec_context_proj_act: str = 'tanh',
                 dec_context_proj_dropout: int = 0.25,
                 enc_config: Dict = {'type': 'pyramid_rnn_encoder'},
                 dec_config: Dict = {'type': 'luong_decoder'},
                 att_config: Dict = {'type': 'mlp'}, # configuration of attention
    ) -> None:
        super().__init__()

        self.enc_input_size = enc_input_size
        self.dec_input_size = dec_input_size
        self.dec_output_size = dec_output_size
        self.enc_fnn_sizes = enc_fnn_sizes
        self.enc_fnn_act = enc_fnn_act
        self.enc_fnn_dropout = enc_fnn_dropout
        self.enc_rnn_sizes = enc_rnn_sizes
        self.enc_rnn_config = enc_rnn_config
        self.enc_rnn_dropout = enc_rnn_dropout
        self.enc_rnn_subsampling = enc_rnn_subsampling
        self.enc_rnn_subsampling_type = enc_rnn_subsampling_type
        self.dec_embedding_size = dec_embedding_size
        self.dec_embedding_dropout = dec_embedding_dropout
        self.dec_embedding_weights_tied = dec_embedding_weights_tied
        self.dec_rnn_sizes = dec_rnn_sizes
        self.dec_rnn_config = dec_rnn_config
        self.dec_rnn_dropout = dec_rnn_dropout
        self.dec_context_proj_size = dec_context_proj_size
        self.dec_context_proj_act = dec_context_proj_act
        self.dec_context_proj_dropout = dec_context_proj_dropout
        self.enc_config = enc_config
        self.dec_config = dec_config
        self.att_config = att_config

        assert enc_config['type'] == 'pyramid_rnn_encoder', \
            "The encoder type '{}' is not implemented. Supported types include 'pyramid_rnn_encoder'.".format(enc_config['type'])
        assert dec_config['type'] == 'luong_decoder', \
            "The decoder type '{}' is not implemented. Supported types include 'luong_encoder'.".format(dec_config['type'])

        # Encoder
        self.encoder = PyramidRNNEncoder(enc_input_size=enc_input_size,
                                         enc_fnn_sizes=enc_fnn_sizes,
                                         enc_fnn_act=enc_fnn_act,
                                         enc_fnn_dropout=enc_fnn_dropout,
                                         enc_rnn_sizes=enc_rnn_sizes,
                                         enc_rnn_config=enc_rnn_config,
                                         enc_rnn_dropout=enc_rnn_dropout,
                                         enc_rnn_subsampling=enc_rnn_subsampling,
                                         enc_rnn_subsampling_type=enc_rnn_subsampling_type,
                                         enc_input_padding_value=0.0)
        self.enc_context_size = self.encoder.get_context_size()

        # Embedder
        self.dec_embedder = nn.Embedding(dec_input_size, dec_embedding_size, padding_idx=None)

        # Decoder
        self.decoder = LuongDecoder(att_config=att_config,
                                    context_size=self.enc_context_size,
                                    input_size=dec_embedding_size,
                                    rnn_sizes=dec_rnn_sizes,
                                    rnn_config=dec_rnn_config,
                                    rnn_dropout=dec_rnn_dropout,
                                    context_proj_size=dec_context_proj_size,
                                    context_proj_act=dec_context_proj_act,
                                    context_proj_dropout=dec_context_proj_dropout)

        # Presoftmax
        self.dec_presoftmax = nn.Linear(dec_context_proj_size, dec_output_size) # decoder.output_size == dec_context_proj_size

        # Typing weight
        if (dec_embedding_weights_tied):
            assert (dec_input_size, dec_embedding_size) == (dec_output_size, dec_context_proj_size), \
                f"When tying embedding weights: the shape of embedder weights: " + \
                f"(dec_input_size = {dec_input_size}, dec_embedding_size = {dec_embedding_size})\n" + \
                f"should be same as the shape of presoftmax weights: " + \
                f"(dec_output_size = {dec_output_size}, dec_context_proj_size = {dec_context_proj_size})"
            # tie weights between dec_embedder(input_onehot => input_embedding) and presoftmax(output_embedding => pre_softmax)
            self.dec_presoftmax.weight = self.dec_embedder.weight

    def get_config(self) -> Dict:
        return {'class': str(self.__class__),
                'enc_input_size': self.enc_input_size,
                'dec_input_size': self.dec_input_size,
                'dec_output_size': self.dec_output_size,
                'enc_fnn_sizes': self.enc_fnn_sizes,
                'enc_fnn_act': self.enc_fnn_act,
                'enc_fnn_dropout': self.enc_fnn_dropout,
                'enc_rnn_sizes': self.enc_rnn_sizes,
                'enc_rnn_config': self.enc_rnn_config,
                'enc_rnn_dropout': self.enc_rnn_dropout,
                'enc_rnn_subsampling': self.enc_rnn_subsampling,
                'enc_rnn_subsampling_type': self.enc_rnn_subsampling_type,
                'dec_embedding_size': self.dec_embedding_size,
                'dec_embedding_dropout': self.dec_embedding_dropout,
                'dec_embedding_weights_tied': self.dec_embedding_weights_tied,
                'dec_rnn_sizes': self.dec_rnn_sizes,
                'dec_rnn_config': self.dec_rnn_config,
                'dec_rnn_dropout': self.dec_rnn_dropout,
                'dec_context_proj_size': self.dec_context_proj_size,
                'dec_context_proj_act': self.dec_context_proj_act,
                'dec_context_proj_dropout': self.dec_context_proj_dropout,
                'enc_config': self.enc_config,
                'dec_config': self.dec_config,
                'att_config': self.att_config}

    def encode(self,
               enc_input: torch.Tensor,
               enc_input_lengths: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Paramters
        ---------
        enc_input: input feature with shape (batch_size x max_seq_length x in_size),
        enc_input_lengths: lengths of input with shape (batch_size) or None

        Returns
        -------
        the context vector (batch_size x max_context_length x context_size)
        the mask of context vector (batch_size x max_context_length).

        Note:
        We set the context for decoder when calling the encode function.
        """
        context, context_mask = self.encoder.encode(enc_input, enc_input_lengths)
        self.decoder.set_context(context, context_mask)
        return context, context_mask

    def decode(self,
               dec_input: torch.Tensor,
               dec_mask: Union[torch.Tensor, None] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Paramters
        ---------
        dec_input: a sequence of input tokenids at current time step with shape [batch_size]
        dec_mask: mask the embedding at the current step with shape [batch_size] or None
            # target batch 3 with length 2, 1, 3 => mask = [[1, 1, 0], [1, 0, 0], [1, 1, 1]]
            # Each time step dec_mask corresponds to each column of the mask.
            # For example: in time step 2, the second column [1, 0, 1] as the dec_mask

        Returns
        -------
        the dec_output(or presoftmax) with shape (batch_size x dec_output_size)
        the att_output of key 'p_context' with its value (batch_size x context_length)
            and key 'expected_context' with its value (batch_size x context_size).

        Note:
        Before calling self.decode, make sure the context is already set by calling self.encode.
        """
        assert dec_input.dim() == 1, "Input of decoder should with a sequence of tokenids with size [batch_size]"
        dec_input_embedding = self.dec_embedder(dec_input)
        dec_input_embedding = F.dropout(dec_input_embedding, p=self.dec_embedding_dropout, training=self.training)
        dec_output, att_output = self.decoder.decode(dec_input_embedding, dec_mask)
        return self.dec_presoftmax(dec_output), att_output

    def reset(self):
        """ Reset the decoder state.
        e.g. the luong decoder sets the attentional vector of the previous time step to be None,
        which means forgetting all the formation accumulated
        in the history (by RNN) before the current time step
        and forgetting the attention information of the previous time step.
        """
        self.decoder.reset()

def test_EncRNNDecRNNAtt():
    seed = 2020
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    json_file = 'conf/data/test_small/utts.json'
    padding_tokenid = 1 # follow torchtext

    with open(json_file, encoding='utf8') as f:
        utts_json = json.load(f)
        utts_instances = list(utts_json.values())

        dataset = KaldiDataset(utts_instances)
        dataloader = KaldiDataLoader(dataset=dataset, batch_size=3, padding_tokenid=padding_tokenid)

    batches = []
    for batch in dataloader:
        batches.append(batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: '{}'".format(device))

    batch = next(iter(dataloader))
    # batch = batches[1]
    enc_input = batch['feat']
    enc_input_lengths = batch['num_frames'] # or None
    enc_input_size = enc_input.shape[-1]
    dec_input = batch['tokenid']
    dec_input_size = batch['vocab_size']
    dec_output_size = batch['vocab_size']

    model = EncRNNDecRNNAtt(enc_input_size,
                            dec_input_size,
                            dec_output_size,
                            enc_fnn_sizes=[4, 9],
                            enc_fnn_act='relu',
                            enc_fnn_dropout=0.25,
                            enc_rnn_sizes=[5, 5, 5],
                            enc_rnn_config={'type': 'lstm', 'bi': True},
                            enc_rnn_dropout=0.25,
                            enc_rnn_subsampling=[False, True, True],
                            enc_rnn_subsampling_type='concat',
                            dec_embedding_size=6,
                            dec_embedding_dropout=0.25,
                            dec_embedding_weights_tied=True,
                            dec_rnn_sizes=[8, 8],
                            dec_rnn_config={"type": "lstmcell"},
                            dec_rnn_dropout=0.25,
                            dec_context_proj_size=6,
                            dec_context_proj_act='tanh',
                            dec_context_proj_dropout=0.25,
                            enc_config={'type': 'pyramid_rnn_encoder'},
                            dec_config={'type': 'luong_decoder'},
                            att_config={'type': 'mlp'})

    enc_input, enc_input_lengths, dec_input = enc_input.to(device), enc_input_lengths.to(device), dec_input.to(device)
    model = model.to(device)
    context, context_mask = model.encode(enc_input, enc_input_lengths)
    dec_output, att_output = model.decode(dec_input[:,2]) # at time step 2

    print(f"enc_input: {enc_input}")
    print(f"enc_input_lengths: {enc_input_lengths}")
    print(f"dec_input at time step 2: {dec_input[:,2]}")
    print(f"context: {context}")
    print(f"context_mask: {context_mask}")
    print(f"dec_output: {dec_output}")
    print(f"att_output: {att_output}")

def greedy_search_torch(model: nn.Module,
                       source: torch.Tensor,
                       source_lengths: torch.Tensor,
                       sos_id: int,
                       eos_id: int,
                       max_dec_length: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.Tensor, torch.Tensor]:
    """ Generate the hypothesis from source by greedy search (beam search with beam_size 1)

    Parameters
    ----------
    model: an attention sequence2sequence model
    source: shape of [batch_size, source_max_length, source_size]
    source_length: shape of [batch_size]
    sos_id: id of the start of sequence token
    eos_id: id of the end of sequence token
    max_dec_length: the maximum length of the hypothesis (a sequence of tokens)
        the decoder can generate, even if eos token does not occur.

    Returns
    -------
    hypothesis: shape [batch_size, dec_length]; each hypothesis is a sequence of tokenid
        (which has no sos_id, but with eos_id if its length is less than max_dec_length)
    lengths of hypothesis: shape [batch_size]; length without sos_id but with eos_id
    attentions of hypothesis: shape [batch_size, dec_length, context_size]
    presoftmax of hypothesis: shape [batch_size, dec_length, dec_output_size]
    """
    model.reset()
    model.train(False)
    model.encode(source, source_lengths) # set the context for decoding at the same time

    batch_size = source.shape[0]

    hypo_list = [] # list of different time steps
    hypo_att_list = []
    hypo_presoftmax = []
    hypo_lengths = source.new_full([batch_size], -1).long()
    cur_tokenids = source.new_full([batch_size], sos_id).long()
    for time_step in range(max_dec_length):
        presoftmax, dec_att = model.decode(cur_tokenids)
        next_tokenids = presoftmax.argmax(-1) # [batch_size]
        hypo_list.append(next_tokenids)
        hypo_att_list.append(dec_att['p_context'])
        hypo_presoftmax.append(presoftmax)

        for i in range(batch_size):
            if next_tokenids[i] == eos_id and hypo_lengths[i] == -1:
                hypo_lengths[i] = time_step + 1
        if all(hypo_lengths != -1): break
        cur_tokenids = next_tokenids

    hypo = torch.stack(hypo_list, dim=1) # [batch_size, dec_length]
    hypo_att = torch.stack(hypo_att_list, dim=1) # [batch_size, dec_length, context_size]
    hypo_presoftmax = torch.stack(hypo_presoftmax, dim=1) # [batch_size, dec_length, dec_output_size]
    return hypo, hypo_lengths, hypo_att, hypo_presoftmax

def greedy_search(model: nn.Module,
                  source: torch.Tensor,
                  source_lengths: torch.Tensor,
                  sos_id: int,
                  eos_id: int,
                  max_dec_length: int) -> Tuple[List[torch.LongTensor], torch.LongTensor, List[torch.Tensor]]:
    """ Generate the hypothesis from source by greedy search (beam search with beam_size 1)

    Parameters
    ----------
    model: an attention sequence2sequence model
    source: shape of [batch_size, source_max_length, source_size]
    source_length: shape of [batch_size]
    sos_id: id of the start of sequence token
    eos_id: id of the end of sequence token
    max_dec_length: the maximum length of the hypothesis (a sequence of tokens)
        the decoder can generate, even if eos token does not occur.

    Returns
    -------
    cropped hypothesis: a list of [hypo_lengths[i]] tensors with the length batch_size.
        each element in the batch is a sequence of tokenids excluding eos_id.
    cropped lengths of hypothesis: shape [batch_size]; excluding sos_id and eos_id
    cropped attentions of hypothesis: a list of [hypo_lengths[i], context_length[i]] tensors
        with the length batch_size
    cropped presoftmax of hypothesis: a list of [hypo_lengths[i], dec_output_size] tensors

    Example
    -------
    Input:
    token2id, first_batch, model = get_token2id_firstbatch_model()
    source, source_lengths = first_batch['feat'], first_batch['num_frames']
    sos_id, eos_id = int(token2id['<sos>']), int(token2id['<eos>']) # 2, 3
    max_dec_length = 5

    model.to(device)
    source, source_lengths = source.to(device), source_lengths.to(device)
    hypo, hypo_lengths, hypo_att, hypo_presoftmax = greedy_search_torch(model, source, source_lengths, sos_id, eos_id, max_dec_length)
    cropped_hypo, cropped_hypo_lengths, cropped_hypo_att, cropped_hypo_presoftmax = greedy_search(model, source, source_lengths, sos_id, eos_id, max_dec_length)

    Output:
    ---hypo---
    tensor([[5, 4, 3, 3],
            [5, 4, 5, 4],
            [5, 6, 3, 3]])
    ---hypo_lengths---
    tensor([ 3, -1,  3])
    ---hypo_att---
    tensor([[[0.0187, 0.9813],
             [0.0210, 0.9790],
             [0.0193, 0.9807],
             [0.0201, 0.9799]],

            [[0.0057, 0.9943],
             [0.0056, 0.9944],
             [0.0050, 0.9950],
             [0.0056, 0.9944]],

            [[1.0000, 0.0000],
             [1.0000, 0.0000],
             [1.0000, 0.0000],
             [1.0000, 0.0000]]], grad_fn=<StackBackward>)
    ---hypo_presoftmax---
    tensor([[[-2.0391e+00, -2.4686e+00, -3.3696e+00, -1.2013e+00, -1.9056e+00, 4.6328e+00,  1.2401e-01, -9.1314e-01],
             [-4.5584e+00, -5.5560e+00, -1.8747e+00,  2.1034e-03,  1.0197e+00, -3.7233e+00,  9.6586e-01,  2.8960e-02],
             [-3.4743e+00, -4.5990e+00, -2.4292e+00,  6.7183e-01, -6.9239e-02, -2.2819e+00,  5.3374e-01,  9.2140e-03],
             [-3.8338e+00, -5.1679e+00, -1.9896e+00,  8.6487e-01,  5.4353e-01, -3.8287e+00,  5.9950e-01,  2.5497e-01]],

            [[-1.0977e+00, -1.8111e+00, -3.2346e+00, -9.9084e-01, -2.3206e+00, 5.5821e+00, -3.4452e-01, -7.9397e-01],
             [-3.1162e+00, -4.4986e+00, -1.2099e+00, -6.0075e-02,  6.6851e-01, -2.0799e+00,  2.1094e-01,  2.7038e-01],
             [-1.5080e+00, -2.7002e+00, -2.3081e+00, -2.9946e-01, -1.3555e+00, 2.6545e+00, -4.2277e-01, -1.3397e-01],
             [-3.0643e+00, -4.4616e+00, -1.1970e+00, -2.8974e-02,  6.4926e-01, -2.0641e+00,  1.8507e-01,  2.8324e-01]],

            [[-2.2006e+00, -2.2896e+00, -3.6796e+00, -1.0538e+00, -1.8577e+00, 4.2987e+00,  5.3117e-01, -1.2819e+00],
             [-4.5086e+00, -4.8001e+00, -2.4802e+00, -1.3172e-01,  9.3378e-01, -3.6198e+00,  1.4054e+00, -6.8509e-01],
             [-2.6262e+00, -3.4670e+00, -2.7019e+00,  1.9906e+00, -3.1856e-01, -3.5389e+00,  6.1016e-01, -2.3925e-01],
             [-3.5176e+00, -4.2128e+00, -2.5136e+00,  1.0150e+00,  3.2375e-01, -3.6914e+00,  9.3068e-01, -3.6401e-01]]], grad_fn=<StackBackward>)

    ---cropped_hypo---
    [tensor([5, 4]), tensor([5, 4, 5, 4]), tensor([5, 6])]
    ---cropped_hypo_lengths---
    tensor([2, 4, 2])
    ---cropped_hypo_att---
    [tensor([[0.0187, 0.9813],
            [0.0210, 0.9790]], grad_fn=<SliceBackward>),
     tensor([[0.0057, 0.9943],
            [0.0056, 0.9944],
            [0.0050, 0.9950],
            [0.0056, 0.9944]], grad_fn=<AliasBackward>),
     tensor([[1.],
            [1.]], grad_fn=<SliceBackward>)]
    ---cropped_hypo_presoftmax---
    [tensor([[-2.0391e+00, -2.4686e+00, -3.3696e+00, -1.2013e+00, -1.9056e+00, 4.6328e+00,  1.2401e-01, -9.1314e-01],
             [-4.5584e+00, -5.5560e+00, -1.8747e+00,  2.1034e-03,  1.0197e+00, -3.7233e+00,  9.6586e-01,  2.8960e-02]], grad_fn=<SliceBackward>),
     tensor([[-1.0977, -1.8111, -3.2346, -0.9908, -2.3206,  5.5821, -0.3445, -0.7940],
             [-3.1162, -4.4986, -1.2099, -0.0601,  0.6685, -2.0799,  0.2109,  0.2704],
             [-1.5080, -2.7002, -2.3081, -0.2995, -1.3555,  2.6545, -0.4228, -0.1340],
             [-3.0643, -4.4616, -1.1970, -0.0290,  0.6493, -2.0641,  0.1851,  0.2832]], grad_fn=<SliceBackward>),
     tensor([[-2.2006, -2.2896, -3.6796, -1.0538, -1.8577,  4.2987,  0.5312, -1.2819],
             [-4.5086, -4.8001, -2.4802, -0.1317,  0.9338, -3.6198,  1.4054, -0.6851]], grad_fn=<SliceBackward>)]
    """
    batch_size = source.shape[0]
    # shape: [batch_size, dec_length], [batch_size], [batch_size, dec_length, context_size]
    hypo, hypo_lengths, hypo_att, hypo_presoftmax = greedy_search_torch(model, source, source_lengths, sos_id, eos_id, max_dec_length)

    context_lengths = mask2length(model.decoder.context_mask)
    cropped_hypo_lengths = crop_hypothesis_lengths(hypo_lengths, max_dec_length) # remove eos_id
    cropped_hypo = [hypo[i][0:cropped_hypo_lengths[i]] for i in range(batch_size)]
    cropped_hypo_att = [hypo_att[i][0:cropped_hypo_lengths[i], 0:context_lengths[i]] for i in range(batch_size)]
    cropped_hypo_presoftmax = [hypo_presoftmax[i][0:cropped_hypo_lengths[i], :] for i in range(batch_size)]

    return cropped_hypo, cropped_hypo_lengths, cropped_hypo_att, cropped_hypo_presoftmax

def crop_hypothesis_lengths(hypo_lengths, max_dec_length):
    """ Remove the eos_id from lengths of the hypothesis

    Parameters
    ----------
    hypo_lengths: shape [batch_size]
    max_dec_length: the maximum length of the decoder hypothesis

    Returns
    -------
    the lengths of hypothesis with eos_id cropped
    """
    cropped_hypo_lengths = torch.ones_like(hypo_lengths)

    batch_size = hypo_lengths.shape[0]
    for i in range(batch_size):
        if hypo_lengths[i] == -1: # reach the max length of decoder without eos_id
            cropped_hypo_lengths[i] = max_dec_length
        else:
            cropped_hypo_lengths[i] = hypo_lengths[i] - 1 # remove eos_id

    return cropped_hypo_lengths

def get_token2id_firstbatch_model():
    """ for testing """
    seed = 2020
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    token2id_file = "conf/data/test_small/token2id.txt"
    token2id = {}
    with open(token2id_file, encoding='utf8') as f:
        for line in f:
            token, tokenid = line.strip().split()
            token2id[token] = tokenid
    padding_id = token2id['<pad>'] # 1

    json_file = 'conf/data/test_small/utts.json'
    utts_json = json.load(open(json_file, encoding='utf8'))
    dataloader = KaldiDataLoader(dataset=KaldiDataset(list(utts_json.values())), batch_size=3, padding_tokenid=1)
    first_batch = next(iter(dataloader))
    feat_dim, vocab_size = first_batch['feat_dim'], first_batch['vocab_size'] # global config

    pretrained_model = "conf/data/test_small/pretrained_model/model_e2000.mdl"
    model = load_pretrained_model_with_config(pretrained_model)
    return token2id, first_batch, model

def test_greedy_search():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: '{}'".format(device))
    token2id, first_batch, model = get_token2id_firstbatch_model()
    source, source_lengths = first_batch['feat'], first_batch['num_frames']
    sos_id, eos_id = int(token2id['<sos>']), int(token2id['<eos>']) # 2, 3
    max_dec_length = 4

    model.to(device)
    source, source_lengths = source.to(device), source_lengths.to(device)
    hypo, hypo_lengths, hypo_att, hypo_presoftmax = greedy_search_torch(model, source, source_lengths, sos_id, eos_id, max_dec_length)
    print(f"---hypo---\n{hypo}\n---hypo_lengths---\n{hypo_lengths}\n---hypo_att---\n{hypo_att}\n---hypo_presoftmax---\n{hypo_presoftmax}")
    print()
    cropped_hypo, cropped_hypo_lengths, cropped_hypo_att, cropped_hypo_presoftmax = greedy_search(model, source, source_lengths, sos_id, eos_id, max_dec_length)
    print(f"---cropped_hypo---\n{cropped_hypo}\n---cropped_hypo_lengths---\n{cropped_hypo_lengths}")
    print(f"---cropped_hypo_att---\n{cropped_hypo_att}\n---cropped_hypo_presoftmax---\n{cropped_hypo_presoftmax}")

class Node:
    def __init__(self,
                 state: List[int],
                 tokenid_path: List[torch.LongTensor],
                 log_prob: torch.Tensor,
                 score: torch.Tensor,
                 coeff_length_penalty: float = 1.0,
                 active: bool = True) -> None:
        """ The node of the search tree of the beam search.

        Parameters
        ----------
        state: the path of the input indices of different time steps
        tokenid_path: the path of tokenids of output to current node; a list of LongTensors with shape [1]
        log_prob: log probability of the path;  tensors with shape [1] (e.g. torch.Tensor([0.7]))
        score: the score of the path; tensor with the shape [1], usually same as log_prob
        coeff_length_penalty: large coefficient of length penalty will increase more penalty for long sentences.
            Google NMT: https://arxiv.org/pdf/1609.08144.pdf
            formula (14): length_penalty(seq)=(5+len(seq))^coeff/(5+1)^coeff
        active: current node hit the tokenid of end of sequence (<sos>) or not

        Example
        -------
        init_node = Node(state=[4], tokenid_path=[torch.LongTensor([5])], log_prob = torch.FloatTensor([-11]), score = torch.FloatTensor([-11]))
        expanded_nodes = init_node.expand(2, torch.Tensor([0, 0.25, 0.05, 0.3, 0.4]).log(), eos_id=3, expand_size=3)
        print(init_node)
        print("---")
        for node in expanded_nodes: print(node)
        # output:
        # [score: -11.000, tokenid_path:[5], state: [4], active: True]
        # ---
        # [score: -11.916, tokenid_path:[5, 4], state: [4, 2], active: True]
        # [score: -12.204, tokenid_path:[5, 3], state: [4, 2], active: False]
        # [score: -12.386, tokenid_path:[5, 1], state: [4, 2], active: True]
        """
        self.state = state
        self.tokenid_path = tokenid_path
        self.log_prob = log_prob
        self.score = score
        self.coeff_length_penalty = coeff_length_penalty
        self.active = active

    def length_penalty(self, length:int, alpha:float=1.0, const:float=5) -> float:
        """
        Generating the long sequence will add more penalty.
        Google NMT: https://arxiv.org/pdf/1609.08144.pdf
        formula (14): lp(Y)=(5+|Y|)^alpha / (5+1)^alpha
        alpha increases => length_penalty increases
        """
        return ((const + length)**alpha) / ((const + 1)**alpha)

    def expand(self, state_t:int, log_prob_t:torch.Tensor, eos_id:int, expand_size:int=5) -> List['Node']:
        """
        Parameters
        ----------
        state_t: a index of current input (input comes from active nodes of all trees of the previous time step).
        log_prob_t: log probability for expansion of the given node. shape [vocab_size]
        expand_size: the number of the nodes one node of the previous level allows to expand to the current level.
            Usually it is equals to beam_size.
            Alternatively a number smaller than `beam_size` may give better results,
            as it can introduce more diversity into the search.
            See [Beam Search Strategies for Neural Machine Translation.
            Freitag and Al-Onaizan, 2017](https://arxiv.org/abs/1702.01806).

        Returns
        -------
        a list of candidate nodes expanded from the given node
        """
        if expand_size >= len(log_prob_t):
            expand_size = len(log_prob_t) # expand all possible active nodes
        topk_log_prob_t, topk_log_prob_t_indices = log_prob_t.topk(expand_size, dim=0) # shape [expand_size], [expand_size]
        log_seq_prob = self.log_prob + topk_log_prob_t # shape [expand_size]
        scores = log_seq_prob

        scores = scores / self.length_penalty(len(self.tokenid_path), alpha=self.coeff_length_penalty) # shape [expand_size]

        expanded_nodes = []
        for i in range(expand_size):
            active = False if topk_log_prob_t_indices[i].item() == eos_id else True
            expand_node = Node(self.state + [state_t], # e.g. [1, 3] + [4] = [1, 3, 4]
                               self.tokenid_path + [topk_log_prob_t_indices[i].unsqueeze(-1)], # [tensor([1]),tensor([0])]+[tensor([2])]=[tensor([1]),tensor([0]),tensor([2])]
                               log_seq_prob[i].unsqueeze(-1), # e.g. torch.Tensor([0.4])
                               scores[i].unsqueeze(-1),
                               coeff_length_penalty=self.coeff_length_penalty,
                               active=active)
            expanded_nodes.append(expand_node)

        return expanded_nodes

    def __repr__(self):
        return "[score: {:.3f}, tokenid_path:{}, state: {}, active: {}]".format(self.score.item(), [x.item() for x in self.tokenid_path], self.state, self.active)

class Level:
    def __init__(self,
                 beam_size: int  = 5,
                 nbest: int = 1) -> None:
        """ A level with a stack of nodes of one search tree at a certain time step for beam search

        Paramters
        ---------
        beam_size: the number of nodes at previous level allows to expand to the current level
        nbest: the number of best sequences needed to be searched.
            The nbest should be smaller or equals to the beam size
        stack: a set of nodes at current level.
            A node is active if its tokenid_path hasn't ended with <sos>, else the node is finished.
            If the number of finished nodes is more than or equals to nbest, we think this level is finished.

        Example
        -------
        print("---init---")
        batch_size = 2
        empty_nodes = [Node(state=[], tokenid_path=[], log_prob = torch.FloatTensor([0]), score = torch.FloatTensor([0])) for _ in range(batch_size)]
        batch_level = [Level(beam_size=2, nbest=2) for _ in range(batch_size)]
        for b in range(batch_size): batch_level[b].add_node(empty_nodes[b])
        for b in range(batch_size): print(f"Tree {b}, level -1: {batch_level[b]}")
        print("---time step 0---")
        if not batch_level[0].is_finished(): batch_level[0].step([0], torch.Tensor([[0.4, 0.35, 0.25]]).log(), eos_id=0, expand_size=2) # find one sequence with <eos>;tree 0 finish if nbest=1
        if not batch_level[1].is_finished(): batch_level[1].step([1], torch.Tensor([[0.2, 0.5, 0.3]]).log(), eos_id=0, expand_size=2)
        for b in range(batch_size): print(f"Tree {b}, level 0: {batch_level[b]}")
        print("---time step 1---")
        if not batch_level[0].is_finished(): batch_level[0].step([0], torch.Tensor([[0.25, 0.35, 0.4]]).log(), eos_id=0, expand_size=2)
        if not batch_level[1].is_finished(): batch_level[1].step([1, 2], torch.Tensor([[0.1, 0.8, 0.1],[0.9, 0.05, 0.05]]).log(), eos_id=0, expand_size=2)
        for b in range(batch_size): print(f"Tree {b}, level 1: {batch_level[b]}")
        # ---init---
        # Tree 0, level -1: [score: 0.000, tokenid_path:[], state: [], active: True]
        # Tree 1, level -1: [score: 0.000, tokenid_path:[], state: [], active: True]
        # ---time step 0---
        # Tree 0, level 0: [score: -1.100, tokenid_path:[0], state: [0], active: False],[score: -1.260, tokenid_path:[1], state: [0], active: True]
        # Tree 1, level 0: [score: -0.832, tokenid_path:[1], state: [1], active: True],[score: -1.445, tokenid_path:[2], state: [1], active: True]
        # ---time step 1---
        # Tree 0, level 1: [score: -1.100, tokenid_path:[0], state: [0], active: False],[score: -1.966, tokenid_path:[1, 2], state: [0, 0], active: True]
        # Tree 1, level 1: [score: -0.916, tokenid_path:[1, 1], state: [1, 1], active: True],[score: -1.309, tokenid_path:[2, 0], state: [1, 2], active: False]
        """
        assert beam_size >= nbest, "beam_size should be more or equals to nbest"
        self.beam_size = beam_size
        self.nbest = nbest
        self.stack = []

    def add_node(self, node: Node) -> None:
        self.stack.append(node)

    def get_active_nodes(self) -> List[Node]:
        """ Get a list active node at the current level. """
        return [node for node in self.stack if node.active]

    def get_finished_nodes(self) -> List[Node]:
        """ Get a list of finished nodes at the current level. """
        return [node for node in self.stack if not node.active]

    def is_finished(self) -> bool:
        """ Check whether the nbest nodes finished """
        return all([not node.active for node in self.stack[0: self.nbest]])

    def step(self, state_t:List[int], log_prob_t:torch.Tensor, eos_id:int, expand_size=5) -> None:
        """
        One step of a search tree from the previous level to the current level

        Paramters
        ---------
        state_t: indices of current input (previous active nodes of all search trees) of the search tree.
            len(state_t) = num_active_nodes_prev_level_of_search_tree
        log_prob_t: log probability vectors of current output for the search tree
            shape [num_active_nodes_prev_level_of_search_tree, dec_output_size]
        eos_id: the tokenid of the <eos> (end of sequence)
        expand_size: the number of the nodes one node of the previous level allows to expand to the current level.
        """
        nodes_next_level = []

        # Collect the expanded nodes from active nodes of the previous level
        active_nodes = self.get_active_nodes()
        num_active_nodes = len(state_t)
        assert(len(active_nodes) == num_active_nodes)
        for i in range(num_active_nodes):
            nodes_next_level.extend(active_nodes[i].expand(state_t[i], log_prob_t[i], eos_id, expand_size))

        # Collect the finished nodes of the previous level
        nodes_next_level.extend(self.get_finished_nodes())

        # prune the stack to beam_size by preserving the nodes with highest score
        self.stack = nodes_next_level
        self.sort()

    def sort(self):
        """ sort the nodes at current level and keep the nodes with highest score """
        # take all possible nodes if beam_size is very large
        beam_size = self.beam_size if self.beam_size <= len(self.stack) else len(self.stack)
        self.stack = sorted(self.stack, key = lambda node: node.score.item(), reverse=True)[0:beam_size]

    def __repr__(self):
        return ",".join(map(str, self.stack))

    @staticmethod
    def split_indices(batch_num_active_nodes: List[int]) -> Tuple[List, List]:
        """ Split indices of active nodes of all trees of the previous time step (the current input indices) for each tree

        Example
        -------
        batch_num_active_nodes: [1, 1, 1]
        starts: [0, 1, 2]
        ends:   [1, 2, 3]
        batch_num_active_nodes: [2, 0, 4]
        starts: [0, 2, 2]
        ends:   [2, 2, 6]
        """
        starts, ends = [] , []
        start = 0
        for num_active_nodes in batch_num_active_nodes:
            end = start + num_active_nodes
            starts.append(start)
            ends.append(end)
            start = end
        return starts, ends

    @staticmethod
    def get_encoder_indices(batch_num_active_nodes: List[int]) -> List[int]:
        """ Get the batch index of context for each active node.

        Examples
        --------
        batch_num_active_nodes of [1, 1, 1] output [0, 1, 2]
        batch_num_active_nodes of [2, 1, 4] output [0, 0, 1, 2, 2, 2, 2]
        batch_num_active_nodes of [2, 0, 4] output [0, 0, 2, 2, 2, 2]
        batch_num_active_nodes of [0, 1, 0] output [1]
        """
        result = []
        batch_size = len(batch_num_active_nodes)
        for i in range(batch_size):
            result.extend([i]*batch_num_active_nodes[i])
        return result

def beam_search_torch(model: nn.Module,
                      source: torch.Tensor,
                      source_lengths: torch.Tensor,
                      sos_id: int,
                      eos_id: int,
                      max_dec_length: int,
                      beam_size: int = 5,
                      expand_size: int = 5,
                      coeff_length_penalty: float = 1,
                      nbest: int = 1) -> Tuple[torch.LongTensor, torch.LongTensor, torch.Tensor]:
    """ Generate the hypothesis from source by beam search.
    The beam search is a greedy algorithm to explore the search tree by expanding the most promising `beam_size' nodes
    at each level of the tree (each time step) to get the most possible sequence.
    This is the batch version of the beam search. The searching strategy applies to a batch of search trees independently.
    However, at each time step, we combine the all active nodes (the nodes without hitting the <sos> token)
    of the previous time step as the current input (or an active batch) to the model for efficiency.
    Note that the decoder (model state) is indexed by the active batch, the encoder (context) is indexed by the batch.

    Parameters
    ----------
    model: an attention sequence2sequence model
    source: shape of [batch_size, source_max_length, source_size]
    source_length: shape of [batch_size]
    sos_id: id of the start of sequence token, to create the start input
    eos_id: id of the end of sequence token, to judge a node is active or not
    max_dec_length: the maximum length of the hypothesis (a sequence of tokens)
        the decoder can generate, even if eos token does not occur.
    beam_size:   the number of the nodes all nodes of the previous level allows to expand to the current level
    expand_size: the number of the nodes one node of the previous level allows to expand to the current level
        Usually expand_size equals beam_size.
        Alternatively a number smaller than `beam_size` may give better results,
        as it can introduce more diversity into the search.
        See [Beam Search Strategies for Neural Machine Translation.
        Freitag and Al-Onaizan, 2017](https://arxiv.org/abs/1702.01806).
    coeff_length_penalty: large coefficient of length penalty will increase more penalty for long sentences.
        Google NMT: https://arxiv.org/pdf/1609.08144.pdf
        formula (14): length_penalty(seq)=(5+len(seq))^coeff/(5+1)^coeff
    nbest: get nbest sequences by the beam search

    Returns
    -------
    hypothesis: shape [batch_size * nbest, dec_length]
        each hypothesis is a sequence of tokenid, ordered as the first nbest chunk,
        the second nbest chunk, ... the batch_size-th nbest chunk
        (which has no sos_id, but with eos_id if its length <  max_dec_length)
    lengths of hypothesis: shape [batch_size * nbest]
        length without sos_id but with eos_id
    attentions of hypothesis: shape [batch_size * nbest, dec_length, context_size]
    presoftmax of hypothesis: shape [batch_size * nbest, dec_length, dec_output_size]

    References
    ----------
    Wiki beam search: https://en.wikipedia.org/wiki/Beam_search
    Basic beam search example: https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/
    """
    model.reset()
    model.train(False)
    model.encode(source, source_lengths) # set the context for decoding at the same time
    context, context_mask = model.decoder.get_context_and_its_mask()

    batch_size = source.shape[0]

    cur_tokenids = source.new_full([batch_size], sos_id).long() # current input [active_batch_size]

    # Initialize a batch of search trees.
    trees = []
    for b in range(batch_size):
        empty_node = Node(state=[], tokenid_path=[], log_prob = torch.FloatTensor([0]).to(source.device),
                          score = torch.FloatTensor([0]).to(source.device), coeff_length_penalty=coeff_length_penalty, active=True)
        level = Level(beam_size, nbest)
        level.add_node(empty_node)
        trees.append(level) # a tree represented by its level at each time step
    batch_num_active_nodes = [1 for _ in range(batch_size)]

    # Explore the nbest sequences of search trees.
    att_list = []
    presoftmax_list = []
    for time_step in range(max_dec_length):
        presoftmax, dec_att = model.decode(cur_tokenids) # shape [active_batch_size, dec_output_size], [active_batch_size, context_length]
        log_prob = F.log_softmax(presoftmax, dim=-1) # shape [active_batch_size, dec_output_size]
        att_list.append(dec_att['p_context'])
        presoftmax_list.append(presoftmax)

        # Expand previous active nodes independently for each tree in the batch.
        starts, ends = Level.split_indices(batch_num_active_nodes) # previous global active indices for each tree: [2,0,4]=>starts:[0,2,2];ends:[2,2,6]
        active_nodes_all_trees = []
        for b in range(batch_size):
            if trees[b].is_finished(): continue # batch_num_active_nodes[b] = 0 by default even skipped
            state_t = list(range(starts[b], ends[b])) # length: [num_nodes_to_expand_curr_tree]
            log_prob_t = log_prob[starts[b]: ends[b]] # shape: [num_nodes_to_expand_curr_tree, dec_output_size]
            trees[b].step(state_t, log_prob_t, eos_id, expand_size=expand_size)
            active_nodes = trees[b].get_active_nodes() # active nodes current level (current time step)
            batch_num_active_nodes[b] = len(active_nodes) if not trees[b].is_finished() else 0
            if not trees[b].is_finished(): active_nodes_all_trees.extend(active_nodes)

        # print(f"------time step {time_step}------")
        # print("\n".join(map(str, trees)))
        if all([tree.is_finished() for tree in trees]): break

        # Collect the active nodes of all trees at current level for the future expansion
        cur_tokenids = torch.cat([node.tokenid_path[-1] for node in active_nodes_all_trees]) # shape [active_batch_size]
        # Update the state of decoder (e.g. attentional_vector_pre for LuongDecoder) for active nodes
        if model.decoder.__class__.__name__ == 'LuongDecoder':
            input_indices = source.new([node.state[-1] for node in active_nodes_all_trees]).long() # shape [active_batch_size] input indices for active nodes
            model.decoder.attentional_vector_pre = torch.index_select(model.decoder.attentional_vector_pre, dim=0, index=input_indices)
            for i in range(len(model.decoder.rnn_hidden_cell_states)):
                hidden, cell = model.decoder.rnn_hidden_cell_states[i]
                hidden = torch.index_select(hidden, dim=0, index=input_indices)
                cell = torch.index_select(cell, dim=0, index=input_indices)
                model.decoder.rnn_hidden_cell_states[i] = (hidden, cell)
        # Update the state of encoder (the context) for active nodes
        context_indices = source.new(Level.get_encoder_indices(batch_num_active_nodes)).long() # get the batch index of context for each active node: [2,0,4]=>[0,0,2,2,2,2]
        model.decoder.set_context(context.index_select(dim=0, index=context_indices), \
                                  context_mask.index_select(dim=0, index=context_indices) if context_mask is not None else None)

    # Generate the hypothesis from the last level of the tree
    hypo_list = [] # list of different time steps
    hypo_length_list = []
    hypo_att_list = []
    hypo_presoftmax_list = []
    for b in range(batch_size):
        for n in range(nbest):
            node = trees[b].stack[n] # iterate over nbest all active nodes at the last level of the tree
            hypo_list.append(torch.cat(node.tokenid_path))
            hypo_length_list.append(len(node.tokenid_path) if node.tokenid_path[-1].item() == eos_id else -1) # -1 means not finished

            node_att_list = [att_list[t][node.state[t]] for t in range(len(node.state))]
            node_att = torch.stack(node_att_list) # shape [dec_length, context_size]
            hypo_att_list.append(node_att)

            node_presoftmax_list = [presoftmax_list[t][node.state[t]] for t in range(len(node.state))]
            node_presoftmax = torch.stack(node_presoftmax_list) # shape [dec_length, dec_output_size]
            hypo_presoftmax_list.append(node_presoftmax)

    hypo = pad_sequence(hypo_list, batch_first=True, padding_value=eos_id)
    hypo_lengths = source.new(hypo_length_list).long()
    hypo_att = pad_sequence(hypo_att_list, batch_first=True, padding_value=0)
    hypo_presoftmax = pad_sequence(hypo_presoftmax_list, batch_first=True, padding_value=0)

    # recover the state of the model
    model.decoder.set_context(context, context_mask)
    model.reset()
    return hypo, hypo_lengths, hypo_att, hypo_presoftmax

def beam_search(model: nn.Module,
                source: torch.Tensor,
                source_lengths: torch.Tensor,
                sos_id: int,
                eos_id: int,
                max_dec_length: int,
                beam_size: int = 5,
                expand_size: int = 5,
                coeff_length_penalty: float = 1,
                nbest: int = 1) -> Tuple[List[torch.LongTensor], torch.LongTensor, List[torch.Tensor], List[torch.Tensor]]:
    """ Generate the hypothesis from source by beam search.
    The beam search is a greedy algorithm to explore the search tree by expanding the most promising `beam_size' nodes
    at each level of the tree (each time step) to get the most possible sequence.
    This is the batch version of the beam search. The searching strategy applies to a batch of search trees independently.
    However, at each time step, we combine the all active nodes (the nodes without hitting the <sos> token)
    of the previous time step as the current input (or an active batch) to the model for efficiency.
    Note that the decoder (model state) is indexed by the active batch, the encoder (context) is indexed by the batch.

    Parameters
    ----------
    model: an attention sequence2sequence model
    source: shape of [batch_size, source_max_length, source_size]
    source_length: shape of [batch_size]
    sos_id: id of the start of sequence token, to create the start input
    eos_id: id of the end of sequence token, to judge a node is active or not
    max_dec_length: the maximum length of the hypothesis (a sequence of tokens)
        the decoder can generate, even if eos token does not occur.
    beam_size:   the number of the nodes all nodes of the previous level allows to expand to the current level
    expand_size: the number of the nodes one node of the previous level allows to expand to the current level
        Usually expand_size equals beam_size.
        Alternatively a number smaller than `beam_size` may give better results,
        as it can introduce more diversity into the search.
        See [Beam Search Strategies for Neural Machine Translation.
            Freitag and Al-Onaizan, 2017](https://arxiv.org/abs/1702.01806).
    coeff_length_penalty: large coefficient of length penalty will increase more penalty for long sentences.
        Google NMT: https://arxiv.org/pdf/1609.08144.pdf
        formula (14): length_penalty(seq)=(5+len(seq))^coeff/(5+1)^coeff
    nbest: get nbest sequences by the beam search

    Returns
    -------
    cropped hypothesis: a list of [hypo_lengths[i]] tensors with the length batch_size*nbest
        each element in the batch is a sequence of tokenids excluding eos_id.
        ordered as the first nbest chunk, the second nbest chunk, ... the batch_size-th nbest chunk
    cropped lengths of hypothesis: shape [batch_size]; excluding sos_id and eos_id
    cropped attentions of hypothesis: a list of [hypo_lengths[i], context_length[i]] tensors
        with the length batch_size*nbest
    cropped presoftmax of hypothesis: a list of [hypo_lengths[i], dec_output_size] tensors
        with the lenght batch_size*nbest (hypo can not back propagate, but hypo presoftmax can)

    References
    ----------
    Wiki beam search: https://en.wikipedia.org/wiki/Beam_search
    Basic beam search example: https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/

    Example
    -------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: '{}'".format(device))
    token2id, first_batch, model = get_token2id_firstbatch_model()
    source, source_lengths = first_batch['feat'], first_batch['num_frames']
    sos_id, eos_id = int(token2id['<sos>']), int(token2id['<eos>']) # 2, 3
    max_dec_length = 4

    model.to(device)
    source, source_lengths = source.to(device), source_lengths.to(device)
    hypo, hypo_lengths, hypo_att, hypo_presoftmax = beam_search_torch(model, source, source_lengths, sos_id, eos_id, max_dec_length, beam_size=2, expand_size=2, nbest=1)
    cropped_hypo, cropped_hypo_lengths, cropped_hypo_att, cropped_hypo_presoftmax = beam_search(model, source, source_lengths, sos_id, eos_id, max_dec_length, beam_size=2, expand_size=2, nbest=1)

    # # Each level of the search tree
    # ------time step 0------
    # [score: -0.026, tokenid_path:[5], state: [0], active: True],[score: -5.436, tokenid_path:[6], state: [0], active: True]
    # [score: -0.010, tokenid_path:[5], state: [1], active: True],[score: -7.122, tokenid_path:[6], state: [1], active: True]
    # [score: -0.044, tokenid_path:[5], state: [2], active: True],[score: -4.565, tokenid_path:[6], state: [2], active: True]
    # ------time step 1------
    # [score: -1.033, tokenid_path:[5, 4], state: [0, 0], active: True],[score: -1.087, tokenid_path:[5, 6], state: [0, 0], active: True]
    # [score: -1.117, tokenid_path:[5, 4], state: [1, 2], active: True],[score: -1.516, tokenid_path:[5, 7], state: [1, 2], active: True]
    # [score: -0.727, tokenid_path:[5, 6], state: [2, 4], active: True],[score: -1.198, tokenid_path:[5, 4], state: [2, 4], active: True]
    # ------time step 2------
    # [score: -1.316, tokenid_path:[5, 6, 3], state: [0, 0, 1], active: False],[score: -1.822, tokenid_path:[5, 4, 3], state: [0, 0, 0], active: False]
    # [score: -1.118, tokenid_path:[5, 4, 5], state: [1, 2, 2], active: True],[score: -2.382, tokenid_path:[5, 7, 4], state: [1, 2, 3], active: True]
    # [score: -0.962, tokenid_path:[5, 6, 3], state: [2, 4, 4], active: False],[score: -1.764, tokenid_path:[5, 4, 6], state: [2, 4, 5], active: True]
    # ------time step 3------
    # [score: -1.316, tokenid_path:[5, 6, 3], state: [0, 0, 1], active: False],[score: -1.822, tokenid_path:[5, 4, 3], state: [0, 0, 0], active: False]
    # [score: -1.823, tokenid_path:[5, 4, 5, 4], state: [1, 2, 2, 0], active: True],[score: -2.097, tokenid_path:[5, 4, 5, 7], state: [1, 2, 2, 0], active: True]
    # [score: -0.962, tokenid_path:[5, 6, 3], state: [2, 4, 4], active: False],[score: -1.764, tokenid_path:[5, 4, 6], state: [2, 4, 5], active: True]
    # ---hypo---
    # tensor([[5, 6, 3, 3],
    #         [5, 4, 5, 4],
    #         [5, 6, 3, 3]], device='cuda:0')
    # ---hypo_lengths---
    # tensor([ 3, -1,  3], device='cuda:0')
    # ---hypo_att---
    # tensor([[[0.0187, 0.9813],
    #          [0.0210, 0.9790],
    #          [0.0212, 0.9788],
    #          [0.0000, 0.0000]],

    #         [[0.0057, 0.9943],
    #          [0.0056, 0.9944],
    #          [0.0050, 0.9950],
    #          [0.0056, 0.9944]],

    #         [[1.0000, 0.0000],
    #          [1.0000, 0.0000],
    #          [1.0000, 0.0000],
    #          [0.0000, 0.0000]]], device='cuda:0', grad_fn=<CopySlices>)
    # ---hypo_presoftmax---
    # tensor([[[-2.0391e+00, -2.4686e+00, -3.3696e+00, -1.2013e+00, -1.9056e+00, 4.6328e+00,  1.2401e-01, -9.1314e-01],
    #          [-4.5584e+00, -5.5560e+00, -1.8747e+00,  2.1036e-03,  1.0197e+00, -3.7233e+00,  9.6586e-01,  2.8960e-02],
    #          [-2.8887e+00, -4.3951e+00, -2.3741e+00,  1.8658e+00, -2.3473e-01, -3.5218e+00,  2.5737e-01,  3.2377e-01],
    #          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, 0.0000e+00,  0.0000e+00,  0.0000e+00]],

    #         [[-1.0977e+00, -1.8111e+00, -3.2346e+00, -9.9084e-01, -2.3206e+00, 5.5821e+00, -3.4452e-01, -7.9397e-01],
    #          [-3.1162e+00, -4.4986e+00, -1.2099e+00, -6.0075e-02,  6.6851e-01, -2.0799e+00,  2.1094e-01,  2.7038e-01],
    #          [-1.5080e+00, -2.7002e+00, -2.3081e+00, -2.9946e-01, -1.3555e+00, 2.6545e+00, -4.2277e-01, -1.3397e-01],
    #          [-3.0643e+00, -4.4616e+00, -1.1970e+00, -2.8974e-02,  6.4926e-01, -2.0641e+00,  1.8507e-01,  2.8324e-01]],

    #         [[-2.2006e+00, -2.2896e+00, -3.6796e+00, -1.0538e+00, -1.8577e+00, 4.2987e+00,  5.3117e-01, -1.2819e+00],
    #          [-4.5086e+00, -4.8001e+00, -2.4802e+00, -1.3172e-01,  9.3378e-01, -3.6198e+00,  1.4054e+00, -6.8509e-01],
    #          [-2.6262e+00, -3.4670e+00, -2.7019e+00,  1.9906e+00, -3.1856e-01, -3.5389e+00,  6.1016e-01, -2.3925e-01],
    #          [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00, 0.0000e+00,  0.0000e+00,  0.0000e+00]]],
    #          device='cuda:0', grad_fn=<CopySlices>)

    # ---cropped_hypo---
    # [tensor([5, 6], device='cuda:0'),
    #  tensor([5, 4, 5, 4], device='cuda:0'),
    #  tensor([5, 6], device='cuda:0')]
    # ---cropped_hypo_lengths---
    # tensor([2, 4, 2], device='cuda:0')
    # ---cropped_hypo_att---
    # [tensor([[0.0187, 0.9813],
    #          [0.0210, 0.9790]], device='cuda:0', grad_fn=<SliceBackward>),
    #  tensor([[0.0057, 0.9943],
    #          [0.0056, 0.9944],
    #          [0.0050, 0.9950],
    #          [0.0056, 0.9944]], device='cuda:0', grad_fn=<AliasBackward>),
    #  tensor([[1.],
    #          [1.]], device='cuda:0', grad_fn=<SliceBackward>)]
    # ---cropped_hypo_presoftmax---
    # [tensor([[-2.0391e+00, -2.4686e+00, -3.3696e+00, -1.2013e+00, -1.9056e+00, 4.6328e+00,  1.2401e-01, -9.1314e-01],
    #          [-4.5584e+00, -5.5560e+00, -1.8747e+00,  2.1036e-03,  1.0197e+00, -3.7233e+00,  9.6586e-01,  2.8960e-02]],
    #          device='cuda:0', grad_fn=<SliceBackward>),
    #  tensor([[-1.0977, -1.8111, -3.2346, -0.9908, -2.3206,  5.5821, -0.3445, -0.7940],
    #          [-3.1162, -4.4986, -1.2099, -0.0601,  0.6685, -2.0799,  0.2109,  0.2704],
    #          [-1.5080, -2.7002, -2.3081, -0.2995, -1.3555,  2.6545, -0.4228, -0.1340],
    #          [-3.0643, -4.4616, -1.1970, -0.0290,  0.6493, -2.0641,  0.1851,  0.2832]],
    #          device='cuda:0', grad_fn=<SliceBackward>),
    #  tensor([[-2.2006, -2.2896, -3.6796, -1.0538, -1.8577,  4.2987,  0.5312, -1.2819],
    #          [-4.5086, -4.8001, -2.4802, -0.1317,  0.9338, -3.6198,  1.4054, -0.6851]],
    #          device='cuda:0', grad_fn=<SliceBackward>)]
    """
    batch_size = source.shape[0]
    # [batch_size*nbest, dec_length], [batch_size*best], [batch_size*nbest, dec_length, context_size] [batch_size*nbest, dec_length, dec_output_size]
    hypo, hypo_lengths, hypo_att, hypo_presoftmax = beam_search_torch(model, source, source_lengths, sos_id, eos_id, max_dec_length,
                                                                      beam_size, expand_size, coeff_length_penalty, nbest)

    batch_size = len(source)
    index = []
    for i in range(batch_size): # [0, 1, 2] => [0, 0, 1, 1, 2, 2] if nbest = 2
        index += [i] * nbest
    context_lengths = mask2length(model.decoder.context_mask) # [batch_size]
    context_lengths = context_lengths.index_select(dim=0, index=source.new(index).long()) # [batch_size * nbest]; [3, 4, 2] => [3, 3, 4, 4, 2, 2]
    cropped_hypo_lengths = crop_hypothesis_lengths(hypo_lengths, max_dec_length) # remove eos_id
    cropped_hypo = [hypo[i][0:cropped_hypo_lengths[i]] for i in range(batch_size * nbest)]
    cropped_hypo_att = [hypo_att[i][0:cropped_hypo_lengths[i], 0:context_lengths[i]] for i in range(batch_size * nbest)]
    cropped_hypo_presoftmax = [hypo_presoftmax[i][0:cropped_hypo_lengths[i], :] for i in range(batch_size * nbest)]

    return cropped_hypo, cropped_hypo_lengths, cropped_hypo_att, cropped_hypo_presoftmax

def test_beam_search():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: '{}'".format(device))
    token2id, first_batch, model = get_token2id_firstbatch_model()
    source, source_lengths = first_batch['feat'], first_batch['num_frames']
    sos_id, eos_id = int(token2id['<sos>']), int(token2id['<eos>']) # 2, 3
    max_dec_length = 4

    model.to(device)
    source, source_lengths = source.to(device), source_lengths.to(device)
    hypo, hypo_lengths, hypo_att, hypo_presoftmax = beam_search_torch(model, source, source_lengths, sos_id, eos_id, max_dec_length, beam_size=2, expand_size=2, nbest=1)
    print(f"---hypo---\n{hypo}\n---hypo_lengths---\n{hypo_lengths}\n---hypo_att---\n{hypo_att}\n---hypo_presoftmax---\n{hypo_presoftmax}")
    print()
    cropped_hypo, cropped_hypo_lengths, cropped_hypo_att, cropped_hypo_presoftmax = beam_search(model, source, source_lengths, sos_id, eos_id, max_dec_length, beam_size=2, expand_size=2, nbest=1)
    print(f"---cropped_hypo---\n{cropped_hypo}\n---cropped_hypo_lengths---\n{cropped_hypo_lengths}")
    print(f"---cropped_hypo_att---\n{cropped_hypo_att}\n---cropped_hypo_presoftmax---\n{cropped_hypo_presoftmax}")

def init_logger(file_name="", stream="stdout"):
    """ Initialize a logger to terminal and file at the same time. """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[ %(asctime)s | %(filename)s | %(levelname)s ] %(message)s", "%d/%m/%Y %H:%M:%S")

    logger.handlers = [] # Clear existing stream and file handlers
    if stream == "stdout":
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if file_name:
        file_handler = logging.FileHandler(file_name, 'w') # overwrite the log file if exists
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def load_model_config(model_config: Dict) -> object:
     """ Get a model object from model configuration file.

     The configuration contains model class name and object parameters,
     e.g., {'class': "<class 'seq2seq.asr.EncRNNDecRNNAtt'>", 'dec_embedding_size: 6}
     """
     import importlib
     full_class_name = model_config.pop('class', None) # get model_config['class'] and delete 'class' item
     module_name, class_name = re.findall("<class '([0-9a-zA-Z_\.]+)\.([0-9a-zA-Z_]+)'>", full_class_name)[0]
     class_obj = getattr(importlib.import_module(module_name), class_name)
     return class_obj(**model_config) # get a model object

def save_model_config(model_config: Dict, path: str) -> None:
    assert ('class' in model_config), "The model configuration should contain the class name"
    json.dump(model_config, open(path, 'w'), indent=4)

def save_model_state_dict(model_state_dict: Dict, path: str) -> None:
    model_state_dict_at_cpu = {k: v.cpu() for k, v in list(model_state_dict.items())}
    torch.save(model_state_dict_at_cpu, path)

def save_options(options: Dict, path: str) -> None:
    json.dump(options, open(path, 'w'), indent=4)

def save_model_with_config(model: nn.Module, model_path: str) -> None:
    """ Given the model and the path to the model, save the model ($dir/model_name.mdl)
    along with its configuration ($dir/model_name.conf) at the same time. """
    assert model_path.endswith(".mdl"), "model '{}' should end with '.mdl'".format(model_path)
    config_path = os.path.splitext(model_path)[0] + ".conf"
    save_model_config(model.get_config(), config_path)
    save_model_state_dict(model.state_dict(), model_path)

def load_pretrained_model_with_config(model_path: str) -> nn.Module:
    """ Given the path to the model, load the model ($dir/model_name.mdl)
    along with its configuration ($dir/model_name.conf) at the same time. """
    assert model_path.endswith(".mdl"), "model '{}' should end with '.mdl'".format(model_path)
    config_path = os.path.splitext(model_path)[0] + ".conf"
    model_config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    pretrained_model = load_model_config(model_config)
    pretrained_model.load_state_dict(torch.load(model_path))
    return pretrained_model

def train_asr():
    exp_dir="exp/tmp"
    model_name = "test_small_att"
    result_dir = os.path.join(exp_dir, model_name, "train")

    data_config_default = "conf/data/test_small/data.yaml"
    model_config_default = "conf/data/test_small/model.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2020, help="seed")
    parser.add_argument('--gpu', type=str, default="0", # if 'auto', running three times in ipython will occupy three different gpus.
                        help="e.g., '--gpu 2' for using 'cuda:2'; '--gpu auto' for using the device with least gpu memory ")

    parser.add_argument('--data_config', type=str, default=data_config_default,
                        help="configuration for dataset (e.g., train, dev and test jsons; \
                        see: conf/data/test_small/data.yaml or conf/data/test_small/create_simple_utts_json.py)")
    parser.add_argument('--cutoff', type=int, default=-1, help="cut off the utterances with the frames more than x.")
    parser.add_argument('--const_token', type=json.loads, default=dict(unk='<unk>', pad='<pad>', sos='<sos>', eos='<eos>', spc='<space>'),
                        help="constant token dict used in text, default as '{\"unk\":\"<unk>\", \"pad\":\"<pad>\", \"sos\":\"<sos>\", \"eos\":\"<eos>\", \"spc\": \"<space>\"}'")
    parser.add_argument('--batch_size', type=int, default=3, help="batch size for the dataloader")

    parser.add_argument('--model_config', type=str, default=model_config_default,
                        help="configuration for model; see: conf/data/test_small/model.yaml")
    parser.add_argument('--pretrained_model', default="",
                        help="the path to pretrained model (model.mdl) with its configuration (model.conf) at same directory")
    parser.add_argument('--label_smoothing', type=float, default=0, help="label smoothing for loss function")
    parser.add_argument('--optim', type=str, default='Adam', help="optimizer")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate for optimizer")
    parser.add_argument('--reducelr', type=json.loads, default={'factor':0.5, 'patience':3},
                        help="None or a dict with keys of 'factor' and 'patience'. \
                        If performance keeps bad more than 'patience' epochs, \
                        reduce the lr by lr = lr * 'factor'")

    parser.add_argument('--num_epochs', type=int, default=10, help="number of epochs")
    parser.add_argument('--grad_clip', type=float, default=20, help="gradient clipping to prevent exploding gradient (NaN).")
    parser.add_argument('--save_interval', type=int, default=1, help='save the model every x epoch')

    parser.add_argument('--result', type=str, default=result_dir, help="result directory")
    parser.add_argument('--overwrite', action='store_true', help='overwrite the result')
    parser.add_argument('--exit', action='store_true', help="immediately exit training or continue with additional epochs")

    args = parser.parse_args()

    ###########################
    opts = vars(args)

    np.random.seed(opts['seed'])
    torch.manual_seed(opts['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opts['seed'])

    if opts['gpu'] != 'auto':
        device = torch.device("cuda:{}".format(opts['gpu']) if torch.cuda.is_available() else "cpu")
    else:
        import GPUtil # Get the device using the least GPU memory.
        device = torch.device("cuda:{}".format(GPUtil.getAvailable(order='memory')[0]) if torch.cuda.is_available() and \
                              GPUtil.getAvailable(order='memory') else "cpu")

    overwrite_warning = ""
    if not os.path.exists(opts['result']):
        os.makedirs(opts['result'])
    else:
        assert os.path.dirname(opts['pretrained_model']) != opts['result'], \
            "the pretrained_model '{}' at the existing result directory '{}' to be deleted for overwriting!".format(
                opts['pretrained_model'], opts['result'])
        overwrite_or_not = 'yes' if opts['overwrite'] else None
        while overwrite_or_not not in {'yes', 'no', 'n', 'y'}:
            overwrite_or_not = input("Overwriting the result directory ('{}') [y/n]?".format(opts['result']).lower().strip())
        if overwrite_or_not in {'yes', 'y'}:
            for x in glob.glob(os.path.join(opts['result'], "*")):
                if os.path.isdir(x): shutil.rmtree(x)
                if os.path.isfile(x): os.remove(x)
            overwrite_warning = "!!!Overwriting the result directory: '{}'".format(opts['result'])
        else: sys.exit(0) # overwrite_or_not in {'no', 'n'}

    save_options(opts, os.path.join(opts['result'], f"options.json"))

    logger = init_logger(os.path.join(opts['result'], "report.log"))
    if overwrite_warning: logger.warning(overwrite_warning)
    logger.info("python " + ' '.join([x for x in sys.argv])) # save current script command
    logger.info("Getting Options...")
    logger.info("\n" + pprint.pformat(opts))
    logger.info("Device: '{}'".format(device))

    ###########################
    logger.info("Loading Dataset...")
    data_config = yaml.load(open(opts['data_config']), Loader=yaml.FullLoader) # contains token2id map file and (train, dev, test) utterance json file
    logger.info("\n" + pprint.pformat(data_config))

    token2id, id2token = {}, {}
    with open(data_config['token2id'], encoding='utf8') as ft2d:
        for line in ft2d:
            token, token_id = line.split()
            token2id[token] = int(token_id)
            id2token[int(token_id)] = token
    assert len(token2id) == len(id2token), \
        "token and id in token2id file '{}' should be one-to-one correspondence".format(data_config['token2id'])
    assert opts['const_token']['pad'] in token2id, \
        "Required token {} by option const_token, for padding the token sequence, not found in '{}' file".format(opts['padding_token'], data_config['token2id'])
    padding_tokenid=token2id[opts['const_token']['pad']] # global config.

    dataloader = {}
    for dset in {'train', 'dev', 'test'}:
        instances = json.load(open(data_config[dset], encoding='utf8')).values() # the json file mapping utterance id to instance (e.g., {'02c': {'uttid': '02c', 'num_frames': 20}, ...})

        save_json_path = os.path.join(opts['result'], "excluded_utts_" + dset + ".json") # json file (e.g., '$result_dir/excluded_utts_train.json') to save the excluded long utterances
        if (opts['cutoff'] > 0):
            instances, _ = KaldiDataset.cutoff_long_instances(instances, cutoff=opts['cutoff'], dataset=dset, save_excluded_utts_to=save_json_path, logger=logger) # cutoff the long utterances

        dataset = KaldiDataset(instances, field_to_sort='num_frames') # Every batch has instances with similar lengths, thus less padded elements; required by pad_packed_sequence (pytorch < 1.3)
        shuffle_batch = True if dset == 'train' else False # shuffle the batch when training, with each batch has instances with similar lengths.
        dataloader[dset] = KaldiDataLoader(dataset=dataset, batch_size=opts['batch_size'], shuffle_batch=shuffle_batch, padding_tokenid=padding_tokenid)

    ###########################
    logger.info("Loading Model...")
    first_batch = next(iter(dataloader['train']))
    feat_dim, vocab_size = first_batch['feat_dim'], first_batch['vocab_size'] # global config

    if opts['pretrained_model']:
        model = load_pretrained_model_with_config(opts['pretrained_model']).to(device)
        logger.info("Loading the pretrained model '{}'".format(opts['pretrained_model']))
    else:
        model_config = yaml.load(open(opts['model_config']), Loader=yaml.FullLoader) # yaml can load json or yaml
        model_config['enc_input_size'] = feat_dim
        model_config['dec_input_size'] = vocab_size
        model_config['dec_output_size'] = vocab_size
        model = load_model_config(model_config).to(device)

    ###########################
    logger.info("Making Loss Function...")
    # Set the padding token with weight zero, other types one.
    classes_weight = torch.ones(vocab_size).detach().to(device)
    classes_weight[padding_tokenid] = 0
    loss_func = CrossEntropyLossLabelSmoothing(label_smoothing=opts['label_smoothing'], weight=classes_weight, reduction='none') # loss per batch
    loss_func.to(device)

    ###########################
    logger.info("Setting Optimizer...")
    optimizer = get_optim(opts['optim'])(model.parameters(), lr=opts['lr'])
    if opts['reducelr'] is not None:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=opts['reducelr']['factor'],
                                      patience=opts['reducelr']['patience'], min_lr=5e-5, verbose=True)

    ###########################
    logger.info("Start Training...")

    def run_batch(feat, feat_len, text, text_len, train_batch, model=model, loss_func=loss_func, optimizer=optimizer):
        """ Run one batch.

        Parameters
        ----------
        feat (batch_size x max_seq_length x enc_input_size)
        text (batch_size x max_text_length)
        feat_len, text_len (batch_size)
        train_batch (bool): training when True, evluating when False.
                            when train_batch is False, we will not update the parameters,
                            and stop some functions such as dropout.

        Returns
        -------
        average token loss of the current batch
        token accuracy of the current batch

        Example
        -------
        first_batch = next(iter(dataloader['train']))
        feat, feat_len = first_batch['feat'].to(device), first_batch['num_frames'].to(device)
        text, text_len = first_batch['tokenid'].to(device), first_batch['num_tokens'].to(device)
        loss, acc = run_batch(feat, feat_len, text, text_len, train_batch=True)
        """

        dec_input = text[:, 0:-1] # batch_size x dec_length
        dec_target = text[:, 1:] # batch_size x dec_length
        batch_size, dec_length = dec_input.shape

        model.train(train_batch) # for dropout and etc.
        model.reset() # reset the state for each utterance for decoder

        model.encode(feat, feat_len) # encode and set context for decoder

        dec_presoftmax_list = []
        for dec_time_step in range(dec_length):
            dec_presoftmax_cur, _ = model.decode(dec_input[:, dec_time_step]) # batch_size x dec_output_size (or vocab_size or num_classes)
            dec_presoftmax_list.append(dec_presoftmax_cur)
        dec_presoftmax = torch.stack(dec_presoftmax_list, dim=-2) # batch_size x dec_length x vocab_size

        length_denominator = text_len - 1 # batch_size
        loss = loss_func(dec_presoftmax.contiguous().view(batch_size * dec_length, -1),
                         dec_target.contiguous().view(batch_size * dec_length)).view(batch_size, dec_length) # batch_size x dec_length
        loss = loss.sum(-1) / length_denominator.float() # average over the each length of the batch; shape [batch_size]
        loss = loss.mean()

        batch_padded_token_matching = dec_presoftmax.argmax(dim=-1).eq(dec_target) # batch_size x dec_length
        batch_token_matching = batch_padded_token_matching.masked_select(dec_target.ne(padding_tokenid)) # shape: [num_tokens_of_current_batch] type: bool
        batch_num_tokens = length_denominator.sum()
        acc = batch_token_matching.sum() / batch_num_tokens.float() # torch.Tensor([True, True, False]).sum() = 2

        if train_batch:
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
            optimizer.step()

        return loss.item(), acc.item()

    def continue_train(message=""):
        """ Returns a bool indicating continue training or not and an integer of how more epochs to train"""
        continue_or_not = ""
        while continue_or_not not in {'yes', 'y', 'no', 'n'}:
            continue_or_not = input(message + "Continue to train [y/n]?").lower().strip()

        add_epochs = "0" if continue_or_not in {'no', 'n'} else ""
        while not add_epochs.isdigit():
            add_epochs = input("How many addition epochs [1 to N]:").lower().strip()

        return continue_or_not in {'yes', 'y'}, int(add_epochs)

    best_dev_loss = sys.float_info.max
    best_dev_epoch = 0

    epoch = 0
    num_epochs = opts['num_epochs']
    while epoch < num_epochs:
        start_time = time.time()
        # take mean over statistics of utterances
        mean_loss = dict(train=0, dev=0, test=0)
        mean_acc = dict(train=0, dev=0, test=0)
        mean_count = dict(train=0, dev=0, test=0)

        for dataset_name, dataset_loader, dataset_train_mode in [['train', dataloader['train'], True],
                                                                 ['dev', dataloader['dev'], False],
                                                                 ['test', dataloader['test'], False]]:
            for batch in tqdm.tqdm(dataset_loader, ascii=True, ncols=50):
                feat, feat_len = batch['feat'].to(device), batch['num_frames'].to(device)
                text, text_len = batch['tokenid'].to(device), batch['num_tokens'].to(device)
                batch_loss, batch_acc = run_batch(feat, feat_len, text, text_len, train_batch=dataset_train_mode)
                if np.isnan(batch_loss): raise ValueError("NaN detected")
                num_utts = len(batch['uttid'])
                mean_loss[dataset_name] += batch_loss * num_utts # sum(average token loss per utterance)
                mean_acc[dataset_name] += batch_acc * num_utts   # sum(average token accuracy per utterance)
                mean_count[dataset_name] += num_utts             # number of utterances of the whole dataset

        info_table = []
        for dataset_name in ['train', 'dev', 'test']:
            # averge over number of utterances of the whole dataset for the current epoch
            mean_loss[dataset_name] /= mean_count[dataset_name]
            mean_acc[dataset_name] /= mean_count[dataset_name]
            info_table.append([epoch, dataset_name + "_set", mean_loss[dataset_name], mean_acc[dataset_name]])

        epoch_duration = time.time() - start_time
        logger.info("Epoch {} -- lrate {} -- time {:.2f}".format(epoch, optimizer.param_groups[0]['lr'], epoch_duration))

        if epoch % opts['save_interval'] == 0:
            save_model_with_config(model, os.path.join(opts['result'], "model_e{}.mdl".format(epoch)))

        if best_dev_loss > mean_loss['dev']:
            best_dev_loss = mean_loss['dev']
            best_dev_epoch = epoch
            logger.info("Get the better dev loss {:.3f} at epoch {} ... saving the model".format(best_dev_loss, best_dev_epoch))
            save_model_with_config(model, os.path.join(opts['result'], "best_model.mdl"))

        logger.info("\n" + tabulate.tabulate(info_table, headers=['epoch', 'dataset', 'loss', 'acc'], floatfmt='.3f', tablefmt='rst'))

        if opts['reducelr']: scheduler.step(mean_loss['dev'], epoch)

        if epoch == num_epochs - 1 and not opts['exit']:
            command = "python " + ' '.join([x for x in sys.argv])
            message = "command: '{}'\nresult: '{}'\n".format(command, opts['result'])
            continue_or_not, add_epochs = continue_train(message) # add 'python command' and 'result directory' to message
            if continue_or_not and add_epochs:
                num_epochs += add_epochs
                logging.info("Add {} more epochs".format(add_epochs))

        epoch += 1

    logger.info("Result path: {}".format(opts['result']))
    logger.info("Get the best dev loss {:.3f} at the epoch {}".format(best_dev_loss, best_dev_epoch))

def recog_asr():
    data_config_default = "conf/data/test_small/data.yaml"
    # set_uttid_default = "conf/data/test_small/set_uttid.txt"
    set_uttid_default = None

    exp_dir="exp/tmp"
    model_name = "test_small_att"
    result_dir = os.path.join(exp_dir, model_name, "eval")
    # model_path = os.path.join(exp_dir, model_name, "train/best_model.mdl")
    model_path = "conf/data/test_small/pretrained_model/model_e2000.mdl"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', type=str, default="0",
                        help="e.g., '--gpu 2' for using 'cuda:2'; '--gpu auto' for using the device with least gpu memory ")
    parser.add_argument('--result', type=str, default=result_dir, help="result directory")
    parser.add_argument('--data_config', type=str, default=data_config_default,
                        help="configuration for dataset (e.g., train, dev and test jsons; \
                        see: conf/data/test_small/data.yaml or conf/data/test_small/create_simple_utts_json.py)")
    parser.add_argument('--batch_size', type=int, default=3, help="batch size for the dataloader")
    parser.add_argument('--model', type=str, default=model_path,
                        help="the path of model from training. \
                        e.g., exp/test/train/best_model.mdl; assume best_model.conf is at same directory.")

    parser.add_argument('--search', type=str, choices=['greedy', 'beam'], default='greedy', help="beam search or greedy search")
    parser.add_argument('--max_target', type=int, default=4, help="the maximum length of decoded sequences")
    parser.add_argument('--save_att', action='store_true', default=False, help="save and plot attention")
    # beam search
    parser.add_argument('--beam_size', type=int, default=2,
                        help="the number of nodes all nodes totally allowed to the next time step (beam_search)")
    parser.add_argument("--coeff_length_penalty", type=float, default=1,
                        help="coefficient to add penalty for decoding the long sequence (beam_search)")

    parser.add_argument('--set_uttid', type=str, default=set_uttid_default, help="a list of uttids of the subset of utterances for testing")
    parser.add_argument('--const_token', type=json.loads, default=dict(unk='<unk>', pad='<pad>', sos='<sos>', eos='<eos>', spc='<space>'),
                            help="constant token dict used in text, default as '{\"unk\":\"<unk>\", \"pad\":\"<pad>\", \"sos\":\"<sos>\", \"eos\":\"<eos>\", \"spc\": \"<space>\"}'")
    args = parser.parse_args()

    ###########################
    opts = vars(args)

    if opts['gpu'] != 'auto':
        device = torch.device("cuda:{}".format(opts['gpu']) if torch.cuda.is_available() else "cpu")
    else:
        import GPUtil # Get the device using the least GPU memory.
        device = torch.device("cuda:{}".format(GPUtil.getAvailable(order='memory')[0]) if torch.cuda.is_available() and \
                              GPUtil.getAvailable(order='memory') else "cpu")

    if not os.path.exists(opts['result']): os.makedirs(opts['result'])
    save_options(opts, os.path.join(opts['result'], f"options.json"))
    logger = init_logger(os.path.join(opts['result'], "report.log"))

    logger.info("python " + ' '.join([x for x in sys.argv])) # save current script command
    logger.info("Getting Options...")
    logger.info("\n" + pprint.pformat(opts))
    logger.info("Device: '{}'".format(device))

    ###########################
    logger.info("Loading Dataset...")
    data_config = yaml.load(open(opts['data_config']), Loader=yaml.FullLoader) # contains token2id map file and (train, dev, test) utterance json file
    logger.info("\n" + pprint.pformat(data_config))

    token2id, id2token = {}, {}
    with open(data_config['token2id'], encoding='utf8') as ft2d:
        for line in ft2d:
            token, token_id = line.split()
            token2id[token] = int(token_id)
            id2token[int(token_id)] = token
    assert len(token2id) == len(id2token), \
        "token and id in token2id file '{}' should be one-to-one correspondence".format(data_config['token2id'])
    assert opts['const_token']['pad'] in token2id, \
        "Required token {} by option const_token, for padding the token sequence, not found in '{}' file".format(opts['padding_token'], data_config['token2id'])
    padding_tokenid = token2id[opts['const_token']['pad']] # global config.
    sos_id = token2id[opts['const_token']['sos']]
    eos_id = token2id[opts['const_token']['eos']]

    dataloader = {}
    for dset in {'test'}:
        # OrderedDict to keep the order the key
        uttid2instance = json.load(open(data_config[dset], encoding='utf8'), object_pairs_hook=OrderedDict) # json file mapping utterance id to instance (e.g., {'02c': {'uttid': '02c', 'num_frames': 20}, ...})
        ordered_uttids = uttid2instance.keys()
        instances = uttid2instance.values()
        if opts['set_uttid'] is not None and opts['set_uttid'] != "None":
            instances = KaldiDataset.subset_instances(instances, key_set_file=opts['set_uttid'], key='uttid') # Only the utterance id in the set_uttid file will be used for testing
            logger.info(f"Get subset of instances according to uttids at '{opts['set_uttid']}'")
        dataset = KaldiDataset(instances, field_to_sort='num_frames') # Every batch has instances with similar lengths, thus less padded elements; required by pad_packed_sequence (pytorch < 1.3)
        shuffle_batch = True if dset == 'train' else False # shuffle the batch when training, with each batch has instances with similar lengths.
        dataloader[dset] = KaldiDataLoader(dataset=dataset, batch_size=opts['batch_size'], shuffle_batch=shuffle_batch, padding_tokenid=padding_tokenid)

    ###########################
    logger.info("Loading Model...")
    model = load_pretrained_model_with_config(opts['model']).to(device)
    model.train(False)
    logger.info("Loading the trained model '{}'".format(opts['model']))

    ###########################
    logger.info("Start Evaluating...")

    metainfo_list = []
    loader = dataloader['test']
    for batch in tqdm.tqdm(loader, ascii=True, ncols=50):
        uttids = batch['uttid']
        feat, feat_len = batch['feat'].to(device), batch['num_frames'].to(device)
        # text, text_len = batch['tokenid'].to(device), batch['num_tokens'].to(device)

        print("--------------")
        cur_best_hypo = None # a list with the length batch_size (element: tensor with shape [hypo_lengths[i]]), excluding <sos> and <eos>.
        cur_best_att = None  # a list with the length batch_size (element: tensor with shape [hypo_lengths[i], context_length[i]])
        if opts['search'] == 'beam':
            cur_best_hypo, _ , cur_best_att, _ = beam_search(model,
                                                             source=feat,
                                                             source_lengths=feat_len,
                                                             sos_id=sos_id,
                                                             eos_id=eos_id,
                                                             max_dec_length=opts['max_target'],
                                                             beam_size=opts['beam_size'],
                                                             expand_size=opts['beam_size'],
                                                             coeff_length_penalty=opts['coeff_length_penalty'],
                                                             nbest=1)
        else:
            cur_best_hypo, _ , cur_best_att, _ = greedy_search(model,
                                                               source=feat,
                                                               source_lengths=feat_len,
                                                               sos_id=sos_id,
                                                               eos_id=eos_id,
                                                               max_dec_length=opts['max_target'])

        for i in range(len(cur_best_hypo)):
            uttid = uttids[i]
            hypo = cur_best_hypo[i].detach().cpu().numpy() # shape [hypo_length]
            att = cur_best_att[i].detach().cpu().numpy() # shape [hypo_length, context_length]
            text = [id2token[tokenid] for tokenid in hypo] # length [hypo_length]
            info = {'uttid': uttid, 'text': ' '.join(text), 'att': att}
            metainfo_list.append(info)


    ###########################
    logger.info("Saving result of metainfo...")

    # meta_dir = os.path.join(opts['result'], "meta")
    meta_dir = opts['result']
    att_dir = opts['result']
    if not os.path.exists(meta_dir): os.makedirs(meta_dir)

    # metainfo of instances back to original order in the json file
    metainfo = []
    for uttid in ordered_uttids:
        for info in metainfo_list:
            if info['uttid'] == uttid:
                metainfo.append(info)
    assert len(metainfo) == len(instances), "Every instance should have its metainfo."

    # save the text of hypothesis characters
    # generate the ref_char if possible (make sure the keys shared by hypothesis and reference, avoiding the subset instance confilict.)
    space_token = opts['const_token']['spc']
    has_ref_char = 'token' in list(instances)[0].keys() # instances [{'uttid': '02c', 'num_frames': 20, 'text', 'token', '<sos> A B <eos>'}, ...]
    with open(os.path.join(meta_dir, "hypo_char.txt"), 'w', encoding='utf8') as hypo_char, \
         open(os.path.join(meta_dir, "ref_char.txt"), 'w', encoding='utf8') as ref_char, \
         open(os.path.join(meta_dir, "hypo_word.txt"), 'w', encoding='utf8') as hypo_word, \
         open(os.path.join(meta_dir, "ref_word.txt"), 'w', encoding='utf8') as ref_word:
        for info in metainfo:
            uttid, text_char = info['uttid'], info['text']
            hypo_char.write(f"{uttid} {text_char}\n")
            text_word = text_char.replace(' ', '').replace(space_token, ' ') # 'A B <space> C' => 'AB C'
            hypo_word.write(f"{uttid} {text_word}\n")
            if has_ref_char:
                text_char = re.sub("<sos>\s+(.+)\s+<eos>", "\\1", uttid2instance[uttid]['token']) # '<sos> A B <eos>' => 'A B'
                ref_char.write(f"{uttid} {text_char}\n")
                text_word = text_char.replace(' ', '').replace(space_token, ' ') # 'A B <space> C' => 'AB C'
                ref_word.write(f"{uttid} {text_word}\n")

    # save and plot attentions
    def save_att_plot(att: np.array, label: List = None, path: str ="att.png") -> None:
        """
        Plot the softmax attention and save the plot.

        Parameters
        ----------
        att: a numpy array with shape [num_decoder_steps, context_length]
        label: a list of labels with length of num_decoder_steps.
        path: the path to save the attention picture
        att = np.array([[0.00565603, 0.994344 ], [0.00560927, 0.9943908 ],
                        [0.00501599, 0.99498403], [0.90557455, 0.1 ]])
        label = ['a', 'b', 'c', 'd']
        save_att_plot(att, label, path='att.png')
        """

        import matplotlib
        matplotlib.use('Agg') # without using x server
        import matplotlib.pyplot as plt

        decoder_length, encoder_length = att.shape # num_decoder_time_steps, context_length

        fig, ax = plt.subplots()
        ax.imshow(att, aspect='auto', origin='lower', cmap='Greys')
        plt.gca().invert_yaxis()
        plt.xlabel("Encoder timestep")
        plt.ylabel("Decoder timestep")
        plt.xticks(range(encoder_length))
        plt.yticks(range(decoder_length))
        if label: ax.set_yticklabels(label)

        plt.tight_layout()
        plt.savefig(path, format='png')
        plt.close()

    def save_att(info: Dict, att_dir: str) -> Tuple[str, str]:
        """
        Save the metainfo of attention named by its uttid.
        The info is a dict with key of 'att' and 'uttid'.

        The image and npz_file of attention matrix
        with its uttid will be save at att_dir

        The path of the image and npz file is returned.

        attention is a matrix with shape [decoder_length, encoder_length]
        """
        att, uttid = info['att'], info['uttid']
        att_image_path = os.path.join(att_dir, f"{uttid}_att.png")
        save_att_plot(att, label=None, path=att_image_path)
        att_npz_path = os.path.join(att_dir, f"{uttid}_att.npz")
        np.savez(att_npz_path, key=uttid, feat=att)
        return att_image_path, att_npz_path

    if opts['save_att']:
        with open(os.path.join(att_dir, "att_mat.scp"), 'w') as f_att_mat, \
             open(os.path.join(att_dir, "att_mat_len.scp"), 'w') as f_att_mat_len:
            for info in metainfo:
                _, att_mat_path = save_att(info, att_dir)
                f_att_mat.write(f"{info['uttid']} {att_mat_path}\n")
                f_att_mat_len.write(f"{info['uttid']} {len(info['att'])}\n")

    logger.info("Result path: {}".format(opts['result']))

print("ASR training...")
train_asr()
print("ASR recognizing...")
recog_asr()

# # Test functions
# test_cross_entropy_label_smooth()
# test_encoder()
# test_attention()
# test_luong_decoder()
# test_EncRNNDecRNNAtt()
# train_asr()
# test_greedy_search()
# test_beam_search()
# recog_asr()
#
# subcommand = None
# subcommand = '1'
# subcommand = 'skip'
# while subcommand not in {'1', 'train_asr', '2', 'recog_asr', 'skip'}:
#     subcommand = input("index name\n[1] train_asr\n[2] recog_asr\nEnter index or name (e.g. 1 or train_asr)\n").lower().strip()
# if subcommand in {'1', 'train_asr'}: train_asr()
# if subcommand in {'2', 'recog_asr'}: recog_asr()
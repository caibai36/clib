from typing import List, Dict, Tuple, Union, Any

import sys
import os
import shutil
import glob
import time
import logging
# Add clib package at current directory to the binary searching path.
sys.path.append(os.getcwd())

import argparse
import math
import re # parser class name
import json # for data files
import yaml # for config files
import pprint
import tabulate
import tqdm

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack

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
                        "adamw": torch.optim.AdamW,
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
        raise NotImplementedError("The optim module '{}' is not implemented\n".format(name) +
                                  "Avaliable optim modules include {}".format(avaliable_optim))

def length2mask(sequence_lengths: torch.Tensor, max_length: Union[int, None] = None) -> torch.Tensor:
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
        max_length = int(torch.max(sequence_lengths).item())
    ones_seqs = sequence_lengths.new_ones(len(sequence_lengths), max_length)
    cumsum_ones = ones_seqs.cumsum(dim=-1)

    return (cumsum_ones <= sequence_lengths.unsqueeze(-1)).long()

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

    def forward(self, input: torch.Tensor) -> Dict:
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
        if mask is not None: score.masked_fill(mask==0, -1e9)
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

    def forward(self, input: torch.Tensor) -> Dict:
        query = input['query'] # batch_size x query_size
        batch_size, query_size = query.shape
        context = input['context'] # batch_size x context_length x context_size
        batch_size, context_length, context_size = context.shape
        mask = input.get('mask', None)
        need_expected_context = input.get('need_expected_context', True)

        # score = V*tanh(W[context,query]) formula (8) of "Effective MNT".
        concat = torch.cat([context, query.unsqueeze(-2).expand(batch_size, context_length, query_size)], dim=-1) # batch_size x context_length x (context_size + query_size)
        score = self.proj2score(self.att_act(self.concat2proj(concat))).squeeze(-1) # batch_size x context_length

        if mask is not None: score.masked_fill(mask==0, -1e9)
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
        self.rnn_layers = nn.ModuleList()
        pre_size = input_size + context_proj_size # input feeding
        for i in range(num_rnn_layers):
            self.rnn_layers.append(get_rnn(rnn_config['type'])(pre_size, rnn_sizes[i]))
            pre_size = rnn_sizes[i]

        # Get expected context vector from attention
        self.attention_layer = get_att(att_config['type'])(context_size, pre_size)

        # Combine hidden state and context vector to be attentional vector.
        self.context_proj_layer = nn.Linear(pre_size + context_size, context_proj_size)

        self.output_size = context_proj_size

    def set_context(self, context: torch.Tensor, context_mask: torch.Tensor = None) -> None:
        self.context = context
        self.context_mask = context_mask

    def reset(self) -> None:
        self.attentional_vector_pre = None

    def decode(self, input: torch.Tensor, dec_mask: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
        """
        input: batch_size x input_size
        dec_mask: batch_size
        # target batch 3 with length 2, 1, 3 => mask = [[1, 1, 0], [1, 0, 0], [1, 1, 1]]
        # Each time step corresponds to each column of the mask.
        # In time step 2, the second column [1, 0, 1] as the dec_mask
        # dec_mask with shape [batch_size]
        # target * dec_mask.unsqueeze(-1).expend_as(target) will mask out
        # the feature of the second element of batch at time step 2, while the element with length 1
        """
        batch_size, input_size = input.shape
        if self.attentional_vector_pre is None:
            self.attentional_vector_pre = input.new_zeros(batch_size, self.context_proj_size)

        # Input feeding: initialize the input of LSTM with previous attentional vector information
        output = torch.cat([input, self.attentional_vector_pre], dim=-1)
        for i in range(self.num_rnn_layers):
            output, _ = self.rnn_layers[i](output) # LSTM cell return (h, c)
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
               dec_mask: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
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
        the dec_output with shape (batch_size x dec_output_size)
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
        """ Reset the decoder state. """
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

def train_asr():
    data_config_default = "conf/data/test_small/data.yaml"
    model_config_default = "conf/model/test_small/model.yaml"

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2020, help="seed")
    parser.add_argument('--gpu', type=str, default="0",
                        help="e.g., '--gpu 2' for using 'cuda:2'; '--gpu auto' for using the device with least gpu memory ")

    parser.add_argument('--data_config', type=str, default=data_config_default,
                        help="configuration for dataset (e.g., train, dev and test jsons; \
                        see: conf/data/test_small/data.yaml or conf/data/test_small/create_simple_utts_json.py)")
    parser.add_argument('--cutoff', type=int, default=-1, help="cut off the utterances with the frames more than x.")
    parser.add_argument('--padding_token', type=str, default="<pad>", help="name of token for padding")
    parser.add_argument('--batch_size', type=int, default=3, help="batch size for the dataloader")

    parser.add_argument('--model_config', type=str, default=model_config_default,
                        help="configuration for model; see: conf/data/test_small/model.yaml")
    parser.add_argument('--pretrained_model', default="",
                        help="the path to pretrained model (model.mdl) with its configuration (model.conf) at same directory")
    parser.add_argument('--label_smoothing', type=float, default=0, help="label smoothing for loss function")
    parser.add_argument('--optim', type=str, default='Adam', help="optimizer")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate for optimizer")
    parser.add_argument('--reducelr', type=dict, default={'factor':0.5, 'patience':3},
                        help="None or a dict with keys of 'factor' and 'patience'. \
                        If performance keeps bad more than 'patience' epochs, \
                        reduce the lr by lr = lr * 'factor'")

    parser.add_argument('--num_epochs', type=int, default=10, help="number of epochs")
    parser.add_argument('--grad_clip', type=float, default=20, help="gradient clipping to prevent exploding gradient (NaN).")
    parser.add_argument('--save_interval', type=int, default=1, help='save the model every x epoch')

    parser.add_argument('--result', type=str, default="tmp_result", help="result directory")
    parser.add_argument('--overwrite_result', action='store_true', help='over write the result')
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
        overwrite_or_not = 'yes' if opts['overwrite_result'] else None
        while overwrite_or_not not in {'yes', 'no', 'n', 'y'}:
            overwrite_or_not = input("Overwriting the result directory ('{}') [y/n]?".format(opts['result']).lower().strip())
            if overwrite_or_not in {'yes', 'y'}:
                for x in glob.glob(os.path.join(opts['result'], "*")):
                    if os.path.isdir(x): shutil.rmtree(x)
                    if os.path.isfile(x): os.remove(x)
                overwrite_warning = "!!!Overwriting the result directory: '{}'".format(opts['result'])
            elif overwrite_or_not in {'no', 'n'}: sys.exit(0)
            else: continue

    logger = init_logger(os.path.join(opts['result'], "report.log"))

    if overwrite_warning: logger.warning(overwrite_warning)
    logger.info('{}'.format("python " + ' '.join([x for x in sys.argv]))) # save current script command
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
    assert opts['padding_token'] in token2id, \
        "Required token {}, for padding the token sequence, not found in '{}' file".format(opts['padding_token'], data_config['token2id'])

    dataloader = {}
    padding_tokenid=token2id[opts['padding_token']] # global config.
    for dset in {'train', 'dev', 'test'}:
        instances = json.load(open(data_config[dset], encoding='utf8')).values() # the json file mapping utterance id to instance (e.g., {'02c': {'uttid': '02c' 'num_frames': 20}, ...})

        save_json_path = os.path.join(opts['result'], "excluded_utts_" + dset + ".json") # json file (e.g., '$result_dir/excluded_utts_train.json') to save the excluded long utterances
        if (opts['cutoff'] > 0):
            instances, _ = KaldiDataset.cutoff_long_instances(instances, cutoff=opts['cutoff'], dataset=dset, save_excluded_utts_to=save_json_path, logger=logger) # cutoff the long utterances

        dataset = KaldiDataset(instances, field_to_sort='num_frames') # Every batch has instances with similar lengths, thus less padded elements; required by pad_packed_sequence (pytorch < 1.3)
        shuffle_batch = True if dset == 'train' else False # shuffle the batch when training, with each batch has instances with similar lengths.
        dataloader[dset] = KaldiDataLoader(dataset=dataset, batch_size=opts['batch_size'], shuffle_batch=shuffle_batch, padding_tokenid=padding_tokenid)

    ###########################
    logger.info("Loading Model...")

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

    def save_model(model_name: str, model: nn.Module) -> None:
        save_options(opts, os.path.join(opts['result'], f"{model_name}.opt"))
        save_model_config(model.get_config(), os.path.join(opts['result'], f"{model_name}.conf"))
        save_model_state_dict(model.state_dict(), os.path.join(opts['result'], f"{model_name}.mdl"))

    def load_pretrained_model_with_config(model_path: str) -> nn.Module:
        """ Given the path to the model, load the model ($dir/model_name.mdl)
        along with its configuration ($dir/model_name.conf) at the same time. """
        assert model_path.endswith(".mdl"), "model '{}' should end with '.mdl'".format(model_path)
        config_path = os.path.splitext(model_path)[0] + ".conf"
        model_config = yaml.load(open(config_path), Loader=yaml.FullLoader)
        pretrained_model = load_model_config(model_config)
        pretrained_model.load_state_dict(torch.load(model_path))
        return pretrained_model

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

    def continue_train():
        """ Returns a bool indicating continue training or not and an integer of how more epochs to train"""
        continue_or_not = ""
        while continue_or_not not in {'yes', 'y', 'no', 'n'}:
            continue_or_not = input("Continue to train [y/n]?").lower().strip()

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
            save_model("model_e{}".format(epoch), model)

        if best_dev_loss > mean_loss['dev']:
            best_dev_loss = mean_loss['dev']
            best_dev_epoch = epoch
            logger.info("Get the better dev loss {:.3f} at epoch {} ... saving the model".format(best_dev_loss, best_dev_epoch))
            save_model("best_model", model)

        logger.info("\n" + tabulate.tabulate(info_table, headers=['epoch', 'dataset', 'loss', 'acc'], floatfmt='.3f', tablefmt='rst'))

        if opts['reducelr']: scheduler.step(mean_loss['dev'], epoch)

        if epoch == num_epochs - 1 and not opts['exit']:
            continue_or_not, add_epochs = continue_train()
            if continue_or_not and add_epochs:
                num_epochs += add_epochs
                logging.info("Add {} more epochs".format(add_epochs))

        epoch += 1

    logger.info("Result path: {}".format(opts['result']))
    logger.info("Get the best dev loss {:.3f} at the epoch {}".format(best_dev_loss, best_dev_epoch))

# test_cross_entropy_label_smooth()
# test_encoder()
# test_attention()
# test_luong_decoder()
# test_EncRNNDecRNNAtt()
# train_asr()

subcommand = None
# subcommand = 'skip'
while subcommand not in {'1', 'train_asr', 'skip'}:
    subcommand = input("index name\n[1] train_asr\n[2] eval_asr\nEnter index or name (e.g. 1 or train_asr)\n").lower().strip()
if subcommand in {'1', 'train_asr'}: train_asr()

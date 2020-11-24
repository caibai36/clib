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
from clib.ctorch.utils.module_util import get_rnn, get_act, get_att, get_optim
from clib.ctorch.utils.tensor_util import length2mask, mask2length
from clib.ctorch.nn.modules.loss import CrossEntropyLossLabelSmoothing
from clib.ctorch.nn.modules.attention import DotProductAttention, MLPAttention
from clib.ctorch.nn.modules.encoder import PyramidRNNEncoder
from clib.ctorch.nn.modules.decoder import LuongDecoder
from clib.ctorch.models.asr.seq2seq.encrnn_decrnn_att_asr import EncRNNDecRNNAtt
from clib.ctorch.nn.search import beam_search, beam_search_torch, greedy_search, greedy_search_torch
from clib.ctorch.utils.model_util import save_options, save_model_with_config, load_pretrained_model_with_config, load_model_config, save_model_config, save_model_state_dict

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

# test_cross_entropy_label_smooth()
# test_encoder()
# test_attention()
# test_luong_decoder()
# test_EncRNNDecRNNAtt()
# test_greedy_search()
# test_beam_search()

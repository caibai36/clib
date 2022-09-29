# Implemented by bin-wu at 10:20 on 12 April 2020
# Heavily inspired by andros's code

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

if 'clib' in os.listdir(os.getcwd()):
    sys.path.append(os.getcwd()) # clib at the current working directory
elif "CLIB" in os.environ:
    sys.path.append(os.environ['CLIB']) # or clib at $CLIB
else:
    print("Please give the path of the dir where the clib locates (e.g., export CLIB=$clib_path), or put the clib in the current directory")
    sys.exit()

from clib.kaldi.kaldi_data import KaldiDataLoader, KaldiDataset
from clib.ctorch.nn.search import beam_search, greedy_search
from clib.ctorch.models.asr.seq2seq.encrnn_decrnn_att_asr import EncRNNDecRNNAtt # needed when loading model
from clib.common.utils.log_util import init_logger
from clib.ctorch.utils.model_util import save_options, load_pretrained_model_with_config
from clib.ctorch.utils.attention_util import save_att

def shrink_repeated_tail_tokens(token_str,  min_repeated_num=3):
    """
    "S C" => "S C"
    "S C C C C" => "S C"
    "C C C C S C" => "C C C C S C"
    "C C C C S C C C" => "C C C C S C"
    "S C K B C A B C A B C A B C" => "S C K B C A"
    "C A B C A B C A B C" => "C A B"
    "C A B B A A A B B A A A B B" => "C A B B A A A B B A A A B B" # repeated two times; shrink for at least three times by default
    "C A A B B A A A A B B A A A A B B A A A A B B A A A" => "C A A B B A A"

    if (len(re.split("\s+", token_str)) == max_target):
       shrinked_token_str = shrink_repeated_tail_tokens(token_str)
    """
    tokens = re.split("\s+", token_str)
    size = len(tokens)

    loop_start = 0 # start position of loop
    final_repeated_num = 0
    final_seg_len = 0
    for seg_len in range(1, size-1):
        for repeated_num in range(size):
            if (size-seg_len*(repeated_num+2) < 0): break
            if tokens[size-seg_len*(repeated_num+2):size-seg_len*(repeated_num+1)] == tokens[size-seg_len*(repeated_num+1):size-seg_len*repeated_num]:
                continue
            else:
                break
        repeated_num += 1 # "S C K B C A B C A B C A B C" repeat 3 times
        if (repeated_num >= min_repeated_num):
            if (repeated_num * seg_len > final_repeated_num * final_seg_len):
                final_repeated_num = repeated_num
                final_seg_len = seg_len
    repeated_pattern = tokens[size - final_seg_len: size] # S C K B C   A B C   A B C   A B C

    residue = tokens[:size-final_repeated_num * final_seg_len]
    i = 0
    for i in range(1, len(repeated_pattern)+1):
        if (len(residue) < i):
            break;
        if repeated_pattern[-i] == residue[-i]:
            continue
        else:
            break
    residue = [] if (len(residue) < i) else residue[:len(residue)-(i-1)] # S C K for S C K B C A B C A B C A B C

    shrinked_token_str = " ".join(tokens[0:len(residue) + len(repeated_pattern)])
    # print("\"{}\" => \"{}\"".format(token_str, shrinked_token_str))
    return shrinked_token_str

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# common
parser.add_argument('--gpu', type=str, default="0",
                    help="e.g., '--gpu 2' for using 'cuda:2'; '--gpu auto' for using the device with least gpu memory")
# data
parser.add_argument('--data_config', type=str, required=True,
                    help=f"configuration for dataset (e.g., train, dev and test jsons;"
                    f"see: conf/data/test_small/data.yaml or conf/data/test_small/create_simple_utts_json.py)")
parser.add_argument('--set_uttid', type=str, default=None, help="a list of uttids of the subset of utterances for testing (e.g. conf/data/test_small/set_uttid.txt)")
parser.add_argument('--batch_size', type=int, default=10, help="batch size for the dataloader")
parser.add_argument('--const_token', type=json.loads, default=dict(unk='<unk>', pad='<pad>', sos='<sos>', eos='<eos>', spc='<space>'),
                        help="constant token dict used in text, default as '{\"unk\":\"<unk>\", \"pad\":\"<pad>\", \"sos\":\"<sos>\", \"eos\":\"<eos>\", \"spc\": \"<space>\"}'")
# model
parser.add_argument('--model', type=str, required=True,
                    help=f"the path of model from training."
                    f"assume best_model.conf is at same directory."
                    f"(e.g., exp/test/train/best_model.mdl, conf/data/test_small/pretrained_model/model_e2000.mdl) ")
# search
parser.add_argument('--max_target', type=int, required=True, help="the maximum length of decoded sequences")
parser.add_argument('--search', type=str, choices=['greedy', 'beam'], default='greedy', help="beam search or greedy search")
parser.add_argument('--beam_size', type=int, default=10,
                    help="the number of nodes all nodes totally allowed to the next time step (beam_search)")
parser.add_argument("--coeff_length_penalty", type=float, default=1,
                    help="coefficient to add penalty for decoding the long sequence (beam_search)")
# others
parser.add_argument('--result', type=str, required=True, help="result directory (e.g., exp/tmp/test_small_att/eval).")
parser.add_argument('--save_att', action='store_true', default=False, help="save and plot attention")
parser.add_argument("--min_repeated_num", type=int, default=5,
                    help="When the generated token sequence by the decoder hits the max limiting length (max_target), shrink the repeated patterns at end of the sequence if the repeated patterns occur more than $num_repeated_num time.")

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
        if len(re.split("\s+", text_char)) == args.max_target:
            shrinked_text_char = shrink_repeated_tail_tokens(text_char,  min_repeated_num=args.min_repeated_num)
            if (shrinked_text_char != text_char):
                logger.warning("Shrinking the repeated tail tokens:\n{}\n=>\n{}\n".format(text_char, shrinked_text_char))
                text_char = shrinked_text_char
        hypo_char.write(f"{uttid} {text_char}\n")
        text_word = text_char.replace(' ', '').replace(space_token, ' ') # 'A B <space> C' => 'AB C'
        hypo_word.write(f"{uttid} {text_word}\n")
        if has_ref_char:
            text_char = re.sub("<sos>\s+(.+)\s+<eos>", "\\1", uttid2instance[uttid]['token']) # '<sos> A B <eos>' => 'A B'
            ref_char.write(f"{uttid} {text_char}\n")
            text_word = text_char.replace(' ', '').replace(space_token, ' ') # 'A B <space> C' => 'AB C'
            ref_word.write(f"{uttid} {text_word}\n")

if opts['save_att']:
    with open(os.path.join(att_dir, "att_mat.scp"), 'w') as f_att_mat, \
         open(os.path.join(att_dir, "att_mat_len.scp"), 'w') as f_att_mat_len:
        for info in metainfo:
            _, att_mat_path = save_att(info, att_dir)
            f_att_mat.write(f"{info['uttid']} {att_mat_path}\n")
            f_att_mat_len.write(f"{info['uttid']} {len(info['att'])}\n")

logger.info("Result path: {}".format(opts['result']))

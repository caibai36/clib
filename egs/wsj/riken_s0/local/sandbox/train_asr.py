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
from torch.utils.tensorboard import SummaryWriter

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
from clib.ctorch.utils.module_util import get_optim
from clib.ctorch.utils.training_util import continue_train
from clib.ctorch.nn.modules.loss import CrossEntropyLossLabelSmoothing
from clib.ctorch.models.asr.seq2seq.encrnn_decrnn_att_asr import EncRNNDecRNNAtt # needed when loading model
from clib.common.utils.log_util import init_logger
from clib.ctorch.utils.model_util import save_options, save_model_with_config, load_pretrained_model_with_config, load_model_config

parser = argparse.ArgumentParser(description="asr training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# common
parser.add_argument('--seed', type=int, default=2020, help="seed")
parser.add_argument('--gpu', type=str, default="0", # if default is 'auto', running three times in ipython will occupy three different gpus.
                    help="e.g., '--gpu 2' for using 'cuda:2'; '--gpu auto' for using the device with least gpu memory ")
# data
parser.add_argument('--data_config', type=str, required=True,
                    help="configuration for dataset (e.g., train, dev and test jsons; \
                    see: conf/data/test_small/data.yaml or conf/data/test_small/create_simple_utts_json.py)")
parser.add_argument('--batch_size', type=int, default=10, help="batch size for the dataloader")
parser.add_argument('--cutoff', type=int, default=-1, help="cut off the utterances with the frames more than x.")
parser.add_argument('--const_token', type=json.loads, default=dict(unk='<unk>', pad='<pad>', sos='<sos>', eos='<eos>', spc='<space>'),
                    help="constant token dict used in text, default as '{\"unk\":\"<unk>\", \"pad\":\"<pad>\", \"sos\":\"<sos>\", \"eos\":\"<eos>\", \"spc\": \"<space>\"}'")
# model
parser.add_argument('--model_config', type=str, required=True,
                    help="configuration for model; see: conf/data/test_small/model.yaml")
parser.add_argument('--pretrained_model', default="",
                    help="the path to pretrained model (model.mdl) with its configuration (model.conf) at same directory")
# loss
parser.add_argument('--label_smoothing', type=float, default=0.0, help="label smoothing for loss function")
# optimizer
parser.add_argument('--optim', type=str, default='Adam', help="optimizer")
parser.add_argument('--lr', type=float, default=0.001, help="learning rate for optimizer")
parser.add_argument('--reducelr', type=json.loads, default={'factor':0.5, 'patience':3},
                    help=f"None or a dict with keys of 'factor' and 'patience'."
                    f"If performance keeps bad more than 'patience' epochs,"
                    f"reduce the lr by lr = lr * 'factor'")
# training
parser.add_argument('--num_epochs', type=int, default=30, help="number of epochs")
parser.add_argument('--grad_clip', type=float, default=20, help="gradient clipping to prevent exploding gradient (NaN).")
parser.add_argument('--save_interval', type=int, default=1, help='save the model every x epoch')
# others
parser.add_argument('--result', type=str, required=True, help="result directory, e.g.,exp/tmp/test_small_att/train")
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

best_dev_loss = sys.float_info.max
best_dev_epoch = 0

epoch = 0
num_epochs = opts['num_epochs']
tensorboard_logdir=os.path.join(opts['result'], "runs")
writer = SummaryWriter(log_dir=tensorboard_logdir)

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
    writer.add_scalar("Loss/train", mean_loss['train'], epoch)
    writer.add_scalar("Loss/dev", mean_loss['dev'], epoch)
    writer.add_scalar("Loss/test", mean_loss['test'], epoch)
    writer.add_scalar("Accuracy/train", mean_acc['train'], epoch)
    writer.add_scalar("Accuracy/dev", mean_acc['dev'], epoch)
    writer.add_scalar("Accuracy/test", mean_acc['test'], epoch)

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

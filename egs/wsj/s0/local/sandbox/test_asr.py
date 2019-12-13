# Add clib package at current directory to the binary searching path.
import sys
import os
sys.path.append(os.getcwd())

import json
import pprint
import argparse
import torch
from clib.kaldi.kaldi_data import KaldiDataLoader, KaldiDataset 

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

    print("the kaldi dataset")
    for instance in dataset:
        pprint.pprint(instance)

    print("the kaldi dataloader")
    batches = []
    for batch in dataloader:
        batches.append(batch)

    for batch in batches:
        pprint.pprint(batch)

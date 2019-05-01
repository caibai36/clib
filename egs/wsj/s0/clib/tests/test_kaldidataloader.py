import json
import pprint
import argparse
import torch
from clib.kaldi.kaldi_data import KaldiDataLoader, KaldiDataset 

file = 'clib/tests/data/test_utts.json'
parser = argparse.ArgumentParser()
parser.add_argument('--json-file', type=str, default=file, help="the test utterance json file")
parser.add_argument('--padding_tokenid', type=int, default=file, help="the id of padding token")

args = parser.parse_args()

with open(args.json_file, encoding='utf8') as f:
    # utts_json is a dictionary mapping utt_id to fields of each utterance
    utts_json = json.load(f)
    # Each utterance instance is a list of fields includes 'feat', 'tokenid' and etc.
    utts_instances = list(utts_json.values())

    dataset = KaldiDataset(utts_instances)
    dataloader = KaldiDataLoader(dataset=dataset, batch_size=2, padding_tokenid=args.padding_tokenid)

    print("the kaldi dataset")
    for instance in dataset:
        pprint.pprint(instance)

    print("the kaldi dataloader")
    for batch in dataloader:
        pprint.pprint(batch)

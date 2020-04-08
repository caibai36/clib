#!/usr/bin/env python3

# re-implemented by bin-wu at 9:54 in 2018.12.26

from typing import List, Set, Dict
import sys
import io
import argparse
import json
import logging

example_string = '''
A json file is a collection of values of a certain attribute indexed by the utterance id.
We merge all attributes of different json files and divide them into input attributes, output attributes and others.
example format of mergedjson:
    "440c0401": {
        "feat": "mfcc/test_eval92/feats_cmvn.1.ark:9",
        "feat_dim": "40",
        "num_frames": "1135",
        "num_tokens": "145",
        "text": "DRA"
        "token": "<sos> d r a <eos>",
        "tokenid": "2 11 25 3",
        "utt2spk": "440",
        "uttid": "440c0401",
        "vocab_size": "34"
    },
}
example:
$ cutils/mergejson.py <json-files-to-merge> [--output-utts-json <merged-json-file>]
$ cutils/mergejson.py cutils/tests/data/dump/*json
$ cutils/mergejson.py cutils/tests/data/dump/*json --output-utts-json cutils/tests/data/utts.json
'''


def main():
    parser = argparse.ArgumentParser(
        description="merge the json files with different attributes for each utterance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=example_string)
    parser.add_argument("jsons", type=str, nargs="+",
                        help="json files")
    parser.add_argument("--verbose", type=int, default=1,
                        help="verbose option")
    parser.add_argument("--output-json", type=str, default="",
                        help="output json file in espnet format (deprecated)")
    parser.add_argument("--output-utts-json", type=str, default="",
                        help="output json file of utterances with their attributes")
    args = parser.parse_args()

    # logging info
    if args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    # Read all json files along with utterance ids.
    json_dicts: List[Dict] = list()
    utt_ids_list: List[Set] = list()
    for j in args.jsons:
        with open(j, 'r', encoding='utf-8') as f:
            json_dict = json.load(f)
            json_dicts.append(json_dict)

            utt_ids: Set = set(json_dict['utts'].keys())
            utt_ids_list.append(utt_ids)

            logging.info(f"{j} has {len(utt_ids)} utterances")

    # Get the common utterance ids of all the json files.
    # Note that we requires to have at least one json file.
    comm_utt_ids: Set = utt_ids_list[0].intersection(*utt_ids_list)
    logging.info('new json has ' + str(len(comm_utt_ids)) + ' utterances')

    # Merge the json files to one all attributes
    all_attrs_json: Dict = {'utts': dict()}
    for utt_id in comm_utt_ids:
        all_attrs_json['utts'][utt_id]: Dict = {'uttid': utt_id}
        for json_dict in json_dicts:
            attr_pair = json_dict["utts"][utt_id]
            all_attrs_json["utts"][utt_id].update(attr_pair)

    # Get final merged json by dividing attributes into the input attrbutes, output attributes and others.
    # input  attributes: feat.json num_frames.json feat_dim.json
    # output attributes: num_tokens.json vocab_size.json text.json  token.json  tokenid.json
    # other  attributes: utt2spk.json
    if args.output_json:
        merged_json = {"utts": dict()}
        for utt_id in comm_utt_ids:
            all_attrs_dict = all_attrs_json["utts"][utt_id]
            input_attrs_dict: Dict = {'name': 'input1',
                                      'feat': all_attrs_dict['feat'],
                                      'shape': [int(all_attrs_dict['num_frames']), int(all_attrs_dict['feat_dim'])]}
            output_attrs_dict: Dict = {'name': 'target1',
                                       'text': all_attrs_dict['text'],
                                       'token': all_attrs_dict['token'],
                                       'tokenid': all_attrs_dict['tokenid'],
                                       'shape': [int(all_attrs_dict['num_tokens']), int(all_attrs_dict['vocab_size'])]}
            merged_json['utts'][utt_id] = {'input': [input_attrs_dict],
                                           'output': [output_attrs_dict],
                                           'utt2spk': all_attrs_dict['utt2spk']}

    if args.output_utts_json:
        with open(args.output_utts_json, 'w', encoding='utf-8') as fuo:
            json.dump(all_attrs_json['utts'], fp=fuo, indent=4, sort_keys=True, ensure_ascii=False)
    else:
        json.dump(all_attrs_json['utts'], fp=io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8'), indent=4, sort_keys=True, ensure_ascii=False)

    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as fo:
            json.dump(merged_json, fp=fo, indent=4, sort_keys=True, ensure_ascii=False)

if __name__ == "__main__":
    main()

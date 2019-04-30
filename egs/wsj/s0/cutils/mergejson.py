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
{
    "utts": {
        "011c0201": {
            "input": [
                {
                    "feat": "espnet/egs/wsj/asr1/dump/train_si284/deltafalse/feats.1.ark:9",
                    "name": "input1",
                    "shape": [
                        652,
                        83
                    ]
                }
            ],
            "output": [
                {
                    "name": "target1",
                    "shape": [
                        5,
                        52
                    ],
                    "text": "THE S",
                    "token": "T H E <space> S",
                    "tokenid": "39 27 24 18 38"
                }
            ],
            "utt2spk": "011"
        }
}
example:
$ mergejson.py tests/data/dump/*json
$ mergejson.py tests/data/dump/*json --output-json tests/data/data.json
'''


def main():
    parser = argparse.ArgumentParser(
        description="merge the json files with different attributes for each utterance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=example_string)
    parser.add_argument("jsons", type=str, nargs="+",
                        help="json files")
    parser.add_argument("--verbose", type=int, default=0,
                        help="verbose option")
    parser.add_argument("--output-json", type=str, default="",
                        help="output json file")
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
    
    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as fo:
            json.dump(merged_json, fp=fo, indent=4, sort_keys=True, ensure_ascii=False)
    else:
        json.dump(merged_json, fp=io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8'), indent=4, sort_keys=True, ensure_ascii=False)

if __name__ == "__main__":
    main()

# Detail tests:
# (mlp) [bin-wu@ahcgpc02 asr0]$(asr *) head tmp/ilen.json tmp/token.json
# ==> tmp/ilen.json <==
# {
#     "utts": {
#         "011c0201": {
#             "ilen": "652"
#         },
#         "011c0202": {
#             "ilen": "693"
#         },
#         "011c0203": {
#             "ilen": "1069"

# ==> tmp/token.json <==
# {
#     "utts": {
#         "011c0201": {
#             "token": "T H E <space> S A L E"
#         },
#         "011c0202": {
#             "token": "T H E <space> H O T E L <space> O P E"
#         }
#     }
# }
#
# # Get all_attrs_json dict.
# (mlp) [bin-wu@ahcgpc02 asr0]$(asr *) python tmp/mergejson.py tmp/ilen.json tmp/token.json
# {
#     "utts": {
#         "011c0201": {
#             "ilen": "652",
#             "token": "T H E <space> S A L E"
#         },
#         "011c0202": {
#             "ilen": "693",
#             "token": "T H E <space> H O T E L <space> O P E"
#         }
#     }
# }

# (mlp) [bin-wu@ahcgpc02 asr0]$(asr *) for file in data/train_si284/tmp-mT75s/*json;do head -8 $file | sed '$ s/,//' >tmp/$(basename $file); echo -e '    }\n}' >>tmp/$(basename $file); done
# (mlp) [bin-wu@ahcgpc02 asr0]$(asr *) ls tmp/*json
# tmp/feat.json  tmp/ilen.json  tmp/olen.json  tmp/token.json    tmp/utt2spk.json
# tmp/idim.json  tmp/odim.json  tmp/text.json  tmp/tokenid.json
# (mlp) [bin-wu@ahcgpc02 asr0]$(asr *) python tmp/mergejson.py --verbose 1 tmp/*json
# 2018-12-26 21:38:25,048 (mergejson:45) INFO: tmp/feat.json has 2 utterances
# 2018-12-26 21:38:25,048 (mergejson:45) INFO: tmp/idim.json has 2 utterances
# 2018-12-26 21:38:25,049 (mergejson:45) INFO: tmp/ilen.json has 2 utterances
# 2018-12-26 21:38:25,049 (mergejson:45) INFO: tmp/odim.json has 2 utterances
# 2018-12-26 21:38:25,050 (mergejson:45) INFO: tmp/olen.json has 2 utterances
# 2018-12-26 21:38:25,050 (mergejson:45) INFO: tmp/text.json has 2 utterances
# 2018-12-26 21:38:25,051 (mergejson:45) INFO: tmp/token.json has 2 utterances
# 2018-12-26 21:38:25,051 (mergejson:45) INFO: tmp/tokenid.json has 2 utterances
# 2018-12-26 21:38:25,052 (mergejson:45) INFO: tmp/utt2spk.json has 2 utterances
# 2018-12-26 21:38:25,052 (mergejson:50) INFO: new json has 2 utterances
# {
#     "utts": {
#         "011c0201": {
#             "input": [
#                 {
#                     "feat": "/project/nakamura-lab08/Work/bin-wu/workspace/sandbox/asrs/espnet/egs/wsj/asr1/dump/train_si284/deltafalse/feats.1.ark:9",
#                     "name": "input1",
#                     "shape": [
#                         652,
#                         83
#                     ]
#                 }
#             ],
#             "output": [
#                 {
#                     "name": "target1",
#                     "shape": [
#                         110,
#                         52
#                     ],
#                     "text": "THE SALE OF THE HOTELS IS PART OF HOLIDAY'S STRATEGY TO SELL OFF ASSETS AND CONCENTRATE ON PROPERTY MANAGEMENT",
#                     "token": "T H E <space> S A L E <space> O F <space> T H E <space> H O T E L S <space> I S <space> P A R T <space> O F <space> H O L I D A Y ' S <space> S T R A T E G Y <space> T O <space> S E L L <space> O F F <space> A S S E T S <space> A N D <space> C O N C E N T R A T E <space> O N <space> P R O P E R T Y <space> M A N A G E M E N T",
#                     "tokenid": "39 27 24 18 38 20 31 24 18 34 25 18 39 27 24 18 27 34 39 24 31 38 18 28 38 18 35 20 37 39 18 34 25 18 27 34 31 28 23 20 44 5 38 18 38 39 37 20 39 24 26 44 18 39 34 18 38 24 31 31 18 34 25 25 18 20 38 38 24 39 38 18 20 33 23 18 22 34 33 22 24 33 39 37 20 39 24 18 34 33 18 35 37 34 35 24 37 39 44 18 32 20 33 20 26 24 32 24 33 39"
#                 }
#             ],
#             "utt2spk": "011"
#         },
#         "011c0202": {
#             "input": [
#                 {
#                     "feat": "/project/nakamura-lab08/Work/bin-wu/workspace/sandbox/asrs/espnet/egs/wsj/asr1/dump/train_si284/deltafalse/feats.1.ark:54819",
#                     "name": "input1",
#                     "shape": [
#                         693,
#                         83
#                     ]
#                 }
#             ],
#             "output": [
#                 {
#                     "name": "target1",
#                     "shape": [
#                         105,
#                         52
#                     ],
#                     "text": "THE HOTEL OPERATOR'S EMBASSY SUITES HOTELS INCORPORATED SUBSIDIARY WILL CONTINUE TO MANAGE THE PROPERTIES",
#                     "token": "T H E <space> H O T E L <space> O P E R A T O R ' S <space> E M B A S S Y <space> S U I T E S <space> H O T E L S <space> I N C O R P O R A T E D <space> S U B S I D I A R Y <space> W I L L <space> C O N T I N U E <space> T O <space> M A N A G E <space> T H E <space> P R O P E R T I E S",
#                     "tokenid": "39 27 24 18 27 34 39 24 31 18 34 35 24 37 20 39 34 37 5 38 18 24 32 21 20 38 38 44 18 38 40 28 39 24 38 18 27 34 39 24 31 38 18 28 33 22 34 37 35 34 37 20 39 24 23 18 38 40 21 38 28 23 28 20 37 44 18 42 28 31 31 18 22 34 33 39 28 33 40 24 18 39 34 18 32 20 33 20 26 24 18 39 27 24 18 35 37 34 35 24 37 39 28 24 38"
#                 }
#             ],
#             "utt2spk": "011"
#         }
#     }
# }

# (mlp) [bin-wu@ahcgpc02 asr0]$(asr *)head tmp/*json
# ==> tmp/feat.json <==
# {
#     "utts": {
#         "011c0201": {
#             "feat": "/project/nakamura-lab08/Work/bin-wu/workspace/sandbox/asrs/espnet/egs/wsj/asr1/dump/train_si284/deltafalse/feats.1.ark:9"
#         },
#         "011c0202": {
#             "feat": "/project/nakamura-lab08/Work/bin-wu/workspace/sandbox/asrs/espnet/egs/wsj/asr1/dump/train_si284/deltafalse/feats.1.ark:54819"
#         }
#     }
# }

# ==> tmp/idim.json <==
# {
#     "utts": {
#         "011c0201": {
#             "idim": "83"
#         },
#         "011c0202": {
#             "idim": "83"
#         }
#     }
# }

# ==> tmp/ilen.json <==
# {
#     "utts": {
#         "011c0201": {
#             "ilen": "652"
#         },
#         "011c0202": {
#             "ilen": "693"
#         }
#     }
# }

# ==> tmp/odim.json <==
# {
#     "utts": {
#         "011c0201": {
#             "odim": "52"
#         },
#         "011c0202": {
#             "odim": "52"
#         }
#     }
# }

# ==> tmp/olen.json <==
# {
#     "utts": {
#         "011c0201": {
#             "olen": "110"
#         },
#         "011c0202": {
#             "olen": "105"
#         }
#     }
# }

# ==> tmp/text.json <==
# {
#     "utts": {
#         "011c0201": {
#             "text": "THE SALE OF THE HOTELS IS PART OF HOLIDAY'S STRATEGY TO SELL OFF ASSETS AND CONCENTRATE ON PROPERTY MANAGEMENT"
#         },
#         "011c0202": {
#             "text": "THE HOTEL OPERATOR'S EMBASSY SUITES HOTELS INCORPORATED SUBSIDIARY WILL CONTINUE TO MANAGE THE PROPERTIES"
#         }
#     }
# }

# ==> tmp/token.json <==
# {
#     "utts": {
#         "011c0201": {
#             "token": "T H E <space> S A L E <space> O F <space> T H E <space> H O T E L S <space> I S <space> P A R T <space> O F <space> H O L I D A Y ' S <space> S T R A T E G Y <space> T O <space> S E L L <space> O F F <space> A S S E T S <space> A N D <space> C O N C E N T R A T E <space> O N <space> P R O P E R T Y <space> M A N A G E M E N T"
#         },
#         "011c0202": {
#             "token": "T H E <space> H O T E L <space> O P E R A T O R ' S <space> E M B A S S Y <space> S U I T E S <space> H O T E L S <space> I N C O R P O R A T E D <space> S U B S I D I A R Y <space> W I L L <space> C O N T I N U E <space> T O <space> M A N A G E <space> T H E <space> P R O P E R T I E S"
#         }
#     }
# }

# ==> tmp/tokenid.json <==
# {
#     "utts": {
#         "011c0201": {
#             "tokenid": "39 27 24 18 38 20 31 24 18 34 25 18 39 27 24 18 27 34 39 24 31 38 18 28 38 18 35 20 37 39 18 34 25 18 27 34 31 28 23 20 44 5 38 18 38 39 37 20 39 24 26 44 18 39 34 18 38 24 31 31 18 34 25 25 18 20 38 38 24 39 38 18 20 33 23 18 22 34 33 22 24 33 39 37 20 39 24 18 34 33 18 35 37 34 35 24 37 39 44 18 32 20 33 20 26 24 32 24 33 39"
#         },
#         "011c0202": {
#             "tokenid": "39 27 24 18 27 34 39 24 31 18 34 35 24 37 20 39 34 37 5 38 18 24 32 21 20 38 38 44 18 38 40 28 39 24 38 18 27 34 39 24 31 38 18 28 33 22 34 37 35 34 37 20 39 24 23 18 38 40 21 38 28 23 28 20 37 44 18 42 28 31 31 18 22 34 33 39 28 33 40 24 18 39 34 18 32 20 33 20 26 24 18 39 27 24 18 35 37 34 35 24 37 39 28 24 38"
#         }
#     }
# }

# ==> tmp/utt2spk.json <==
# {
#     "utts": {
#         "011c0201": {
#             "utt2spk": "011"
#         },
#         "011c0202": {
#             "utt2spk": "011"
#         }
#     }
# }

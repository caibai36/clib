#!/usr/bin/env python3

# re-implemented by bin-wu at 23:33 in 2018.12.24
# Script to convert the scp format to the json format
#
# format of key.scp:
# utter_name1 value1
# utter_name2 value2
#
# format of json output:
# utter_pair := <"utters": utter_map>
# utter_map := {<utter_name: pair>}
# pair := <key, value>

import argparse
import sys
import json
import io

example_string = '''
The scp file is a collection of attributes indexed by utterance id.
see http://kaldi-asr.org/doc/io.html for more details
The json file a collection of attribute name-value pairs indexed by utterance id.
example:
    $ cat ilen.scp 
    011c0201 652 
    011c0202 693 
    $ cat ilen.scp | scp2json.py -k ilen
    {
        "utts": {
            "011c0201": {
                "ilen": "652"
            },
            "011c0202": {
                "ilen": "693"
            }
        }
'''


def main():
    parser = argparse.ArgumentParser(
        description="Script to convert the scp format to the json format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=example_string)
    parser.add_argument("--key", "-k", type=str,
                        help="key")
    args = parser.parse_args()

    utter_map = {}

    # for line in sys.stdin:
    for line in io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8'):
        tokens = line.strip().split()
        utter_name = tokens[0]
        value = " ".join(tokens[1:])

        pair = {args.key: value}
        utter_map[utter_name] = pair

    utter_pair = {"utts": utter_map}

    # print(json.dumps(utter_pair, indent=4))
    # Two versions to deal with the utf-8 string
    # version1: dump
    # json.dump(utter_pair, indent=4, fp=io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8'), ensure_ascii=False)
    # version2: dumps (dump string)
    sys.stdout.buffer.write(json.dumps(utter_pair, indent=4, ensure_ascii=False).encode('utf-8'))


if __name__ == "__main__":
    main()

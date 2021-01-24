import os
import argparse
import logging
# from pyxdameraulevenshtein import damerau_levenshtein_distance, normalized_damerau_levenshtein_distance
import editdistance
import re

# regex collection #
"""
regex_key_val function to extract pair key and value from following format 
<key><space><value>
UTT_1 FEAT_PATH_1
"""
# regex_key_val = re.compile('(^[^\s]+) (.+)$', re.MULTILINE)
regex_key_val = re.compile(r"^(^[^\s]+)\s(.*)$", re.MULTILINE)
regex_key = re.compile('^([^\s]+)$', re.MULTILINE)

def parse() :
    parser = argparse.ArgumentParser(description='python wrapper for Kaldi\'s computer-wer')
    parser.add_argument('--ref', required=True, help='reference transcription  (with key)')
    parser.add_argument('--hypo', required=True, help='hypothesis transcription (with key)')
    return parser.parse_args()
    pass

if __name__ == '__main__' :
    args = parse()
    list_kv_ref = regex_key_val.findall(open(args.ref).read())
    list_kv_hypo = regex_key_val.findall(open(args.hypo).read())

    list_key_ref = [x[0] for x in list_kv_ref]
    list_key_hypo = [x[0] for x in list_kv_hypo]
    if list_key_hypo != list_key_ref :
        # logging.warning('key hypo: {} || ref: {}'.format(len(list_key_hypo), len(list_key_ref)))
        assert set(list_key_hypo).issubset(set(list_key_ref))

    list_kv_hypo = dict((x, y.split()) for (x, y) in list_kv_hypo)
    list_kv_ref = dict((x, y.split()) for (x, y) in list_kv_ref)
        
    total_edist = 0
    total_len = 0
    # for key_hypo in list_key_hypo :
    #     total_edist += editdistance.eval(list_kv_ref[key_hypo], list_kv_hypo[key_hypo])
    #     total_len += len(list_kv_ref[key_hypo])
    # print('{:.3f}% [ {} / {} ]'.format(total_edist / total_len * 100, total_edist, total_len))

    for key_hypo in list_key_hypo :
        key = key_hypo
        dist = editdistance.eval(list_kv_ref[key_hypo], list_kv_hypo[key_hypo])
        print("{} {}".format(key, dist))

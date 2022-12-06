from typing import TextIO
import io
import sys
import os
import re
import argparse
import pprint

def readfile(filename: str = "-") -> TextIO:
    """
    Open an utf8-encoded file to read (bin-wu).

    Parameters
    ----------
    filename: the filename or "-" where "-" means stardard input
    
    Returns
    -------
    the file object

    Usage
    -----
    with readfile("text") as f:  # or f = readfile("text")
        for line in f:
            line = line.strip()
            print(line)
    """
    if filename == "-":
        f = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8') # read utf8 stdin
    else:
        f = open(filename, 'rt', encoding='utf-8') # option `t' means text model

    return f


input_file = "exp/scores/test.txt"
def get_utt2inf(f):
    """
    Get a dict from uttid to sclite information
    
    Example
    -------
    with readfile(args.input) as f:
        print(get_utt2inf(f))

    # id: (MWVW0_SX396)
    # Scores: (#C #S #D #I) 38 10 3 1
    # Attributes: Case_sensitve 
    # REF:  sil dh ix f ih sh vcl b iy vcl g ae n cl t  l iy cl f r ae n cl  t  ix cl k l iy ax n dh ax s er f ix s ax v dh ax s epi m ao * l ey cl k sil 
    # HYP:  sil dh ax f ih sh cl  p ix vcl g ae n cl ch l iy cl f r ae n vcl jh ix cl k l iy ** * ah m  s er f ix s ax v ** ax s epi m ao l l iy cl k sil 
    # Eval:        S          S   S S                S                   S   S               D  D S  S                   D                I   S           

    # {'MWVW0_SX36': {'all': 5,
    #                 'att': 'Case_sensitve',
    #                 'count': 41,
    #                 'del': 2,
    #                 'evl': '       D   S                                        S                                                 D               S      ',
    #                 'hyp': 'sil ow * l ey dh ax m ow s cl t ix cl k aa m cl p l ix sh aa r dx ih s sil ax vcl t ey n cl p aa cl p * ix l eh er dx iy sil',
    #                 'ins': 0,
    #                 'ref': 'sil ow n l ix dh ax m ow s cl t ix cl k aa m cl p l ih sh aa r dx ih s sil ax vcl t ey n cl p aa cl p y ix l eh er dx ix sil',
    #                 'sub': 3},
    #  'MWVW0_SX396': {'all': 14,
    #                  'att': 'Case_sensitve',
    #                  'count': 38,
    #                  'del': 3,
    #                  'evl': '       S          S   S S                S                   S   S               D  D S  S                   D                I   S           ',
    #                  'hyp': 'sil dh ax f ih sh cl  p ix vcl g ae n cl ch l iy cl f r ae n vcl jh ix cl k l iy ** * ah m  s er f ix s ax v ** ax s epi m ao l l iy cl k sil',
    #                  'ins': 1,
    #                  'ref': 'sil dh ix f ih sh vcl b iy vcl g ae n cl t  l iy cl f r ae n cl  t  ix cl k l iy ax n dh ax s er f ix s ax v dh ax s epi m ao * l ey cl k sil',
    #                  'sub': 10}}

    """
    utt2inf = {}
    for line in f:
        if line.strip() and ("Scores:" in line or "id:" in line or "REF:" in line or "HYP:" in line or "Eval:" in line or "Attributes:" in line) : # not empty line
            if "Eval" not in line: line = line.strip()
            if "Eval" not in line:
                key, value = line.split(maxsplit=1)
            else:
                key, value = line.split(" ", maxsplit=1)
            if (key == "id:"):
                uttid = value[1:len(value)-1] # (MWVW0_SX396)
                utt2inf[uttid] = {}
                utt2inf[uttid]['uttid'] = uttid
            if (key == "Scores:"):
                _, _, _, _, count, substitution, deletion, insertion = value.split()
                utt2inf[uttid]['count'] = int(count)
                utt2inf[uttid]['sub'] = int(substitution)
                utt2inf[uttid]['del'] = int(deletion)
                utt2inf[uttid]['ins'] = int(insertion)
                utt2inf[uttid]['all'] = int(substitution) + int(deletion) + int(insertion)
            if (key ==  "REF:"):
                utt2inf[uttid]['ref'] = value
            if (key == "HYP:"):
                utt2inf[uttid]['hyp'] = value
            if (key == "Eval:"):
                utt2inf[uttid]['evl'] = value[:-1]
            if (key == "Attributes:"):
                utt2inf[uttid]['att'] = value
    return utt2inf

get_file_command = r"""hyp=exp/yonden/text_chasen.kana
ref=exp/yonden/text.kana
tag=KER_$(basename $ref)_$(basename $hyp)
./local/sclite_score.sh  --tag $tag --ref $ref --hyp $hyp
"""
parser = argparse.ArgumentParser(description=u"Print all utterances with errors from sclite's file.\n\nExample:\npython local/grep_sclite_error.py --input exp/scores/KER_text.kana_text_chasen.kana/result --text exp/yonden/text\n\nFull example:\n" + get_file_command + "python local/grep_sclite_error.py --input exp/scores/KER_text.kana_text_chasen.kana/result --text exp/yonden/text", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--input", type=str, default="-", help="name of file or '-' of stdin")
parser.add_argument("--text", type=str, default=None, help="the kaldi text file with line format of uttid and text.")
parser.add_argument("--print_all", action="store_true", help="print information of all utterances besides those making the errors")
args = parser.parse_args()

if input != '-':
    filename = os.path.basename(os.path.dirname(args.input)) # exp/scores/KER_text.kana_text_chasen.kana/result
else:
    filename = ""

uttid2content = {}
if (args.text):
    with open(args.text, 'r', encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if (len(re.split("\s+", line)) <= 1):
                uttid = line
                content = ""
            else:
                uttid, content = re.split("\s+", line, maxsplit=1)
            uttid2content[uttid] = content

with readfile(args.input) as f:
    utt2info = get_utt2inf(f)
    for uttid, info in utt2info.items():
        if info["all"] > 0 and not args.print_all:
            print("{},err:A {} S {} I {} D {} all_tokens {} corr_tokens {}".format(uttid, info['all'], info['sub'], info['ins'], info['del'], len(re.split("[\s\*]+", info['ref'])), info['count']))
            if(args.text):
                print("{},utt:{}".format(uttid, uttid2content[uttid]))
            print("{},ref:{}".format(uttid, info["ref"]))
            print("{},hyp:{}".format(uttid, info["hyp"]))
            print("{},evl:{}".format(uttid, info["evl"]))
            print()
        if args.print_all:
            print("{},err:A {} S {} I {} D {} all_tokens {} corr_tokens {}".format(uttid, info['all'], info['sub'], info['ins'], info['del'], len(re.split("[\s\*]+", info['ref'])), info['count']))
            if(args.text):
                print("{},utt:{}".format(uttid, uttid2content[uttid]))
            print("{},ref:{}".format(uttid, info["ref"]))
            print("{},hyp:{}".format(uttid, info["hyp"]))
            print("{},evl:{}".format(uttid, info["evl"]))
            print()

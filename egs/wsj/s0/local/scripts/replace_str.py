#!/usr/bin/env python3

import argparse
import io
import sys

def main():
    parser = argparse.ArgumentParser(description="replace or remove certain strings from utf8 streams from text or stdin",
                                     epilog='python cutils/replace_str.py --text_in=cutils/tests/data/dump/text.scp --rep_in=cutils/tests/data/str_rep')
    parser.add_argument("--text_in", type=str, default="", help="the input text or the standard input")
    parser.add_argument("--text_out", type=str, default="", help="the output text or the standard output")
    parser.add_argument("--rep_in", type=str, required=True, default="", help="the string replacement file. Each line with formats of <oldStr>:<newStr> or <delStr>: ")
    parser.add_argument("--sep", type=str, default=":", help="the separator of the string replacement file. The default separator is :")
    parser.add_argument("--str2lower", action='store_true', default=False, help="convert characters of string to lower case before replacement and after replacement")
    
    args = parser.parse_args()

    text_in = args.text_in
    text_out = args.text_out
    rep_in = args.rep_in
    sep = args.sep
    str2lower = args.str2lower

    if text_in:
        f_in = open(text_in, 'r', encoding='utf-8')
    else:
        f_in = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

    if text_out:
        f_out = open(text_out, 'w', encoding='utf-8')
    else:
        f_out = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    rep_pairs = []
    with open(rep_in, 'r', encoding='utf-8') as f:
        for line in f:
            old, new = line.strip().split(sep)
            rep_pairs.append({'old': old, 'new': new})

    for line in f_in:
        line = line.strip()
        if str2lower: line = line.lower()
        for pair in rep_pairs:
            line = line.replace(pair['old'], pair['new'])
        if str2lower: line = line.lower()
        f_out.write(line + "\n")

if __name__ == "__main__":
    main()

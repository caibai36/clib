from typing import TextIO
import io
import os
import sys

import re
import csv
import argparse

import pandas as pd

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

parser = argparse.ArgumentParser(description=u"dump the info.json into a directory that contains XXX.scp where XXX is a field in the info.json",
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--input", type=str, default="-", help="name of file or '-' of stdin")
parser.add_argument("--dir", type=str, default="scps", help="directory to dump the info.json that contains all ${field}.scp files for info.json")

args = parser.parse_args()
if not os.path.exists(args.dir):
        os.makedirs(args.dir)

with readfile(args.input) as f_input:
    df = pd.read_json(f_input).T
    for column in df.columns:
        with open(os.path.join(args.dir, column+".scp"), 'w', encoding="utf8") as f:
            for index in df.index:
                print(index + " " +  str(df.loc[index, column]), file=f)

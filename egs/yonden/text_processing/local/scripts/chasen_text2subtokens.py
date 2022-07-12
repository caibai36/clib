from typing import TextIO
import io
import sys
import re
import argparse

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

def writefile(filename: str = "-") -> TextIO:
    """
    Open an utf8-encoded file to write (bin-wu).

    Parameters
    ----------
    filename: the filename or "-" where "-" means stardard output
    
    Returns
    -------
    the file object

    Usage
    -----
    with readfile("text") as fr, writefile("output") as fw:
        for line in fr:
            line = line.strip()
            fw.write(line + "\n")
    """
    if filename == "-":
        f = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8') # write utf8 stdout
    else:
        f = open(filename, 'wt', encoding='utf-8') # option `t' means text model

    return f

parser = argparse.ArgumentParser(description=u"Extract subtokens (fields) from the text. \nLine format for text is `uttid token1 token2'. Token format is `field1|field2|field3'.\nExample line of text:\nID22 発表+名詞 形式+名詞 と+助詞/格助詞\nID44 作業|サギョー|作業|名詞-サ変接続|||| 員|イン|員|名詞-接尾-一般|||| 。|。|。|記号-句点||||\nSample line of output: (cat sample | python text2subtokens.py --fields 2 4 --in_sep \| --out_sep +)\nID44 サギョー+名詞-サ変接続 イン+名詞-接尾-一般  。+記号-句点",
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--input", type=str, default="-", help="name of file or '-' of stdin")
parser.add_argument("--output", type=str, default="-", help="name of file or '-' of stdout")
parser.add_argument("--fields", type=int, nargs="+", default=[2], help="extract one field or several fields of a token, starting from 1")
parser.add_argument("--in_sep", type=str, default="|", help="separator between fields of an token from input stream")
parser.add_argument("--out_sep", type=str, default="|", help="separator between fields of an token from output stream")
parser.add_argument("--has_uttid", action="store_true", help="input in Kaldi script format: the first column is an uttid and the remaining is the content for a line.")
args = parser.parse_args()

with readfile(args.input) as f_input, writefile(args.output) as f_output:
    for line in f_input:
        line = line.strip()
        # input line format: uttid f1|f2|f3 f1|f2|f3 f1|f2|f3
        if (args.has_uttid):
            if (len(re.split('\s+', line)) <= 1): # empty line or uttid with empty content
                 uttid = line
                 tokens = ""
            else:
                 uttid, tokens = re.split('\s+', line, maxsplit=1)
            f_output.write(uttid + ' ')
        else:
            tokens = line
        for token in re.split('\s+', tokens):
            fields = token.split(args.in_sep)
            # output line format: uttid f1|f3 f1|f3 f1|f3
            output_fields = []
            for field_index in args.fields:
                if (fields != [''] and len(fields) > 0): #  print("".split("|")) # [''] or print("".split("")) # []
                    if (field_index-1 >= len(fields)):
                        print("IndexError: Index out of range: extracting index {}: for fields of {}".format(field_index-1, fields))
                    output_fields.append(fields[field_index-1]) # field_index starting from 1
            f_output.write((args.out_sep).join(output_fields) + " ")
        f_output.write("\n")

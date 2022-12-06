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

parser = argparse.ArgumentParser(description=u"standard chasen output to line format\nchasen input:\n配電	ハイデン	配電	名詞-サ変接続		\n部	ブ	部	名詞-接尾-一般		\nEOS\n平岡	ヒラオカ	平岡	名詞-固有名詞-人名-姓		\n班	ハン	班	名詞-一般		\nEOS\n\nchasen output:\n配電+ハイデン+配電+名詞-サ変接続 部+ブ+部+名詞-接尾-一般\n平岡+ヒラオカ+平岡+名詞-固有名詞-人名-姓 班+ハン+班+名詞-一般\n", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--input", type=str, default="-", help="name of file or '-' of stdin")
parser.add_argument("--in_elem_sep", type=str, default="\s+", help="the separator between elements/subtokens of a token in an input line.")
parser.add_argument("--out_elem_sep", type=str, default="|", help="the separator between elements/subtokens of a token in a output line. e.g. '+' for '配電+ハイデン+配電+名詞-サ変接続 部+ブ+部+名詞-接尾-一般'")
parser.add_argument("--out_token_sep", type=str, default=" ", help="the separator between tokens of an output line. e.g. ' ' for '配電+ハイデン+配電+名詞-サ変接続 部+ブ+部+名詞-接尾-一般'")

args = parser.parse_args()

in_elem_sep=args.in_elem_sep
elem_sep=args.out_elem_sep
token_sep=args.out_token_sep

with readfile(args.input) as f:
    lines = re.split("EOS", f.read(), flags=re.MULTILINE)
    if lines[-1] == '\n':
        lines = lines[:-1] # remove the last empty line

    for line in lines:
        line = line.strip()
        tokens = re.split('\n', line)
        new_tokens = []
        for token in tokens:
            token = token.strip()
            elements = re.split(in_elem_sep, token)
            new_token = (elem_sep).join(elements)
            new_tokens.append(new_token)
        print((token_sep).join(new_tokens))

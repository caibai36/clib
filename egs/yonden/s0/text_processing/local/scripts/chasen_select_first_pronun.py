import io
import sys

import re
import argparse

from typing import TextIO

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

parser = argparse.ArgumentParser(description="Selecting the first pronunciation in chasen format\nInput line format:...(pronun1$pronun2$pronun3)...(pronun11$pronun12)...\nOutput line format:...pronun1...pronun11...\nThe default begin_char is '(', sep_char is '$' and the end_char is ')'\nexample: cat chasen_text.txt | python chasen_select_first_pronun.py", formatter_class = argparse.RawTextHelpFormatter)
parser.add_argument("--input", type=str, default="-", help="name of file or '-' of stdin")
parser.add_argument("--begin_char", type=str, default="(", help="the beginning character of group of pronunciations (e.g., '(' for group of (pronun1$pronun2$pronun3))")
parser.add_argument("--end_char", type=str, default=")", help="the end character of group of pronunciations (e.g., ')' for group of (pronun1$pronun2$pronun3))")
parser.add_argument("--sep_char", type=str, default="$", help="the seperator character of group of pronunciations (e.g., '$' for group of (pronun1$pronun2$pronun3))")
args = parser.parse_args()

begin_char = args.begin_char
sep_char = args.sep_char
end_char = args.end_char

# add escape
begin_char = "\\" + begin_char
end_char = "\\" + end_char
sep_char = "\\" + sep_char

# (pronun1$pronun2$pronun3) => pronun1 (selected part) pronun2$pronun3 (remaining part)
selected_part = "[^"+ end_char + sep_char +"]*"
remaining_part = "[^" + end_char +"]*"
regex = begin_char +"(" + selected_part + ")"+ sep_char + remaining_part + end_char

print(re.sub(regex, "\\1" , readfile(args.input).read(), flags=re.MULTILINE), end="")

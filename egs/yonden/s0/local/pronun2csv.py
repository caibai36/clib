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


parser = argparse.ArgumentParser(description=u"Convert pronunciation to csv format for mecab or openjtalk", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--input", type=str, default="-", help="name of file or '-' of stdin")
parser.add_argument("--openjtalk", action="store_true", help="Convert to openjtalk format (e.g., 柱上|チュウジョー => 柱上,,,1,名詞,一般,*,*,*,*,柱上,チュウジョー,チュウジョー,2/3,* or 柱上|チュウジョー|名詞|固有名詞 => 柱上,,,1,名詞,固有名詞,*,*,*,*,柱上,チュウジョー,チュウジョー,2/3,*)")
parser.add_argument("--mecab", action="store_true", help="Convert to mecab format (default) (e.g., 柱上|チュウジョー => 柱上,,,,,,,,,,,チュウジョー,チュウジョー)")
parser.add_argument("--mecab_unidic", action="store_true", help="Convert to mecab format for unidic (e.g., 柱上|チュウジョー => 柱上,,,1,,,,*,*,*,チュウジョー,柱上,柱上,チュウジョー,柱上,チュウジョー,*,*,*,*,*,*,*,*,チュウジョー,チュウジョー,チュウジョー,チュウジョー,*,*,*,*,*)")
args = parser.parse_args()

# pronun: 表層形|発音
# 柱上|チュウジョー
# pronun => mecab (ipadic): 表層形,左文脈ID,右文脈ID,コスト,品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音
# 柱上,,,,,,,,,,,チュウジョー,チュウジョー
#
# pronun => openjtalk: see mecab format (The field of `2/3', in following example, is accent related for TTS)
# 柱上,,,1,名詞,一般,*,*,*,*,柱上,チュウジョー,チュウジョー,2/3,*

with readfile(args.input) as f:
    for line in f:
        line = line.strip()
        if line[0] == '#': continue # comments

        pos=u"名詞"
        pos1=u"一般"
        fields = re.split("\|", line)
        if len(fields) == 2:
            text, pronun = fields
        elif len(fields) == 3:
            text, pronun, pos = fields
        else:
            text, pronun, pos, pos1 = fields

        if (args.mecab):
            print("{},,,1,,,,,,,,{},{}".format(text, pronun, pronun))
        elif (args.mecab_unidic):
            print("{},,,1,,,,*,*,*,{},{},{},{},{},{},*,*,*,*,*,*,*,*,{},{},{},{},*,*,*,*,*".format(text, pronun, text, text, pronun, text, pronun, pronun, pronun, pronun, pronun))
        elif (args.openjtalk):
            print("{},,,1,{},{},*,*,*,*,{},{},{},2/3,*".format(text, pos, pos1, text, pronun, pronun))
        else:
            print("{},,,1,,,,,,,,{},{}".format(text, pronun, pronun)) # default mecab format

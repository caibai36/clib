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


parser = argparse.ArgumentParser(description=u"Convert the dict between the format of openjtalk, mecab, and mecab_unidic_csj3\nopenjtalk format: 柱上,,,1,名詞,固有名詞,*,*,*,*,柱上,チュウジョー,チュウジョー,2/3,*\nmecab format: 表層形,左文脈ID,右文脈ID,コスト,品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音\n    e.g. (ipadic), 柱上,,,,,,,,,,,チュウジョー,チュウジョー\n   or 瑤泉院,1349,1349,7464,名詞,固有名詞,人名,一般,*,*,瑤泉院,ヨウゼイイン,ヨーゼーイン,3/6,C1\nmecab_unidic_csj3 format: 柱上,,,1,,,,*,*,*,チュウジョー,柱上,柱上,チュウジョー,柱上,チュウジョー,*,*,*,*,*,*,*,*,チュウジョー,チュウジョー,チュウジョー,チュウジョー,*,*,*,*,*\n   or 千町,12743,11454,4514,名詞,固有名詞,地名,一般,*,*,センチョウ,センチョウ,千町,センチョー,千町,センチョー,固,*,*,*,*,*,*,地名,センチョウ,センチョウ,センチョウ,センチョウ,1,*,*,5655346681094656,20574)", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--input", type=str, default="-", help="name of file or '-' of stdin")
parser.add_argument("--from_format", type=str, default="openjtalk", help="name of dict convert from. For example, by default the 'openjtalk' format.")
parser.add_argument("--to_format", type=str, default="mecab_unidic_csj3", help="name of dict convert to. For example: by default the 'mecab_unidic_csj3' format.")
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
        if not line: continue
        if line[0] == '#': continue # comments

        if (args.from_format == "openjtalk"):
            non_comma="[^,]*"
            to_extract="([^,]*)"

            # 検相,,,1,名詞,一般,*,*,*,*,検相,ケンソー,ケンソー,2/3,*
            pattern="{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(to_extract, non_comma, non_comma, to_extract, to_extract, to_extract, non_comma, non_comma, non_comma, non_comma, non_comma, non_comma, to_extract,non_comma, non_comma)
            extracted = re.findall(pattern, line)

            # あー,2,2,100,フィラー,*,*,*,*,あー,アー,アー,2/3,*
            pattern="{},{},{},{},{},{},{},{},{},{},{},{},{},{}".format(to_extract, non_comma, non_comma, to_extract, to_extract, non_comma, non_comma, non_comma, non_comma, non_comma, non_comma, to_extract,non_comma, non_comma)
            extracted2 = re.findall(pattern, line)
            
            if len(extracted) == 1:
                text, cost, pos, pos1, pronun = extracted[0]
            elif len(extracted2) == 1:
                text, cost, pos, pronun = extracted2[0]
                pos1 = ""
            else:
                print("Warning: {} should have format same as \n柱上,,,1,名詞,固有名詞,*,*,*,*,柱上,チュウジョー,チュウジョー,2/3,*".format(line))
                sys.exit(1)
        else:
            print("Only support openjtalk dict source")
            sys.exit(1)

        if (args.to_format == "mecab_unidic_csj3"):
            print("{},,,{},{},{},,*,*,*,{},{},{},{},{},{},*,*,*,*,*,*,*,*,{},{},{},{},*,*,*,*,*".format(text, cost, pos, pos1, pronun, text, text, pronun, text, pronun, pronun, pronun, pronun, pronun))
        else:
            print("Only support mecab_unidic_csj3")
            sys.exit(1)

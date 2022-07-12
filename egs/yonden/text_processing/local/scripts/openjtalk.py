# Implemented by bin-wu at 00:13 on 7th July 2022
#
# Install the pyopenjalk that support user dictionary:
# source /project/nakamura-lab08/Work/bin-wu/share/tools/gcc/path.sh # gcc 5.4.0, needed for successfully compiling mecab C code.
# git clone https://github.com/Yosshi999/pyopenjtalk.git
# cd pyopenjtalk/
# git branch -a
# git checkout PR-user-dic
# git submodule update --recursive --init
# pip install -e .

import pyopenjtalk

from typing import TextIO
import io
import os
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

user_dict_example=r"""$ cat yonden_openjtalk_test.csv
ＧＮＵ,,,1,名詞,一般,*,*,*,*,ＧＮＵ,グヌー,グヌー,2/3,*
中線,,,1,名詞,一般,*,*,*,*,中線,ナカセン,ナカセン,2/4,*
柱上,,,1,名詞,一般,*,*,*,*,柱上,チュウジョー,チュウジョー,2/3,*
"""

install_user_dict=r"""# Install the pyopenjalk that support user dictionary:
# source /project/nakamura-lab08/Work/bin-wu/share/tools/gcc/path.sh # gcc 5.4.0, needed for successfully compiling mecab C code.
# git clone https://github.com/Yosshi999/pyopenjtalk.git
# cd pyopenjtalk/
# git branch -a
# git checkout PR-user-dic
# git submodule update --recursive --init
# pip install -e .
"""
parser = argparse.ArgumentParser(description=u"Python version of openjtalk (openjtalk processing + mecab parsing)\n\nNote that a special openjtalk version that supports adding user dictionary is needed:\n" + install_user_dict, formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--input", type=str, default="-", help="name of file or '-' of stdin")
parser.add_argument("--keep_accent", action="store_true", help="keep the accent of the pronunciation (e.g., 接地|セッチ’ 体|タイ)")
parser.add_argument("--dict_dir", type=str, default="/project/nakamura-lab08/Work/bin-wu/share/data/openjtalk_dict/open_jtalk_dic_utf_8-1.11", help='Compiled openjtalk dictionary.\ne.g., /project/nakamura-lab08/Work/bin-wu/share/data/openjtalk_dict/open_jtalk_dic_utf_8-1.11 (default)\nor /project/nakamura-lab08/Work/bin-wu/share/data/openjtalk_dict/openjtalk_default_unidic_ipadic\nor /project/nakamura-lab08/Work/bin-wu/share/data/openjtalk_dict/openjtalk_ipadic_unidic_neologd')
parser.add_argument("--user_dict", type=str, default=None, help="Specify an user dict: either a compiled dict of XXX.dic or a raw openjtalk dict of XXX.csv\n" + user_dict_example + "Note that the field of '1' might be cost and the field of '2/3' might be accent related for TTS.\nFields such as '名詞' and '一般' are necessary and take a finite set of options.)")
parser.add_argument("--has_uttid", action="store_true", help="input in Kaldi script format: the first column is an uttid and the remaining is the content for a line.")
parser.add_argument("--csv2dic", type=str, default=None, help="XXX.csv. Convert dict from XXX.csv to XXX.dic and stop")
args = parser.parse_args()

input = args.input
mecab_dict = args.dict_dir
keep_accent=args.keep_accent
user_dict = args.user_dict

# Deal with mecab dict and user dict
pyopenjtalk.OPEN_JTALK_DICT_DIR = mecab_dict.encode("utf8")

if args.csv2dic:
    dict_prefix, dict_postfix = os.path.splitext(args.csv2dic) # ('path/file', '.csv') or ('path/file', '.dic')
    dict_csv = dict_prefix + ".csv"
    dict_dic = dict_prefix + ".dic"
    assert dict_postfix == ".csv", "Error: option 'csv2dic' needs input to be 'XXX.csv', but now getting '{}'".format(args.csv2dic)
    pyopenjtalk.create_user_dict(dict_csv, dict_dic)
    sys.exit()

if user_dict:
    dict_prefix, dict_postfix = os.path.splitext(user_dict) # ('path/file', '.csv') or ('path/file', '.dic')
    dict_csv = dict_prefix + ".csv"
    dict_dic = dict_prefix + ".dic"
    if dict_postfix == ".csv": pyopenjtalk.create_user_dict(dict_csv, dict_dic) # compile from csv file to dic file for mecab.
    pyopenjtalk.set_user_dict(dict_dic)

with readfile(args.input) as f:
    for line in f:
        line = line.strip()
        if not line:
            print() # print an empty line.
            continue

        if (args.has_uttid):
             if (len(re.split('\s+', line)) <= 1): # empty line or uttid with empty content
                 uttid = line
                 line = ""
             else:
                 uttid, line = re.split('\s+', line, maxsplit=1)
             print(uttid, end=" ")

        for token in pyopenjtalk.run_frontend(line)[0]:
            if (not token.split(',')[0].isspace()): # Skip an empty token
                text_field = token.split(',')[0]
                pronun_field = token.split(',')[9] # default for mecab-naist-jdic
                if (not keep_accent): pronun_field = pronun_field.replace("’", "") # 接地|セッチ’ => 接地|セッチ

                if (pronun_field == u"、" and text_field != pronun_field): pronun_field = text_field # 。|、 (text of '。' has pronunciation of '、')
                print(text_field + "|" + pronun_field, end=" ")
        print()

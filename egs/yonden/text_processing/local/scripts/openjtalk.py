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
parser.add_argument("--pos", action="store_true", help="add fields of part of speech of '表層形|読み|品詞|品詞細分類1'\n(e.g., 柱|ハシラ|名詞|一般 上|ジョー|名詞|接尾 or 瑤泉院|ヨーゼーイン|名詞|固有名詞)")
parser.add_argument("--full", action="store_true", help="print all fields of '表層形|読み|品詞|品詞細分類1|品詞細分類2|品詞細分類3|活用型|活用形'\n(e.g., 柱|ハシラ|名詞|一般|*|*|*|* 上|ジョー|名詞|接尾|副詞可能|*|*|*\nor 瑤泉院|ヨーゼーイン|名詞|固有名詞|人名|一般|*|* or ああなり|アーナリ|動詞|自立|*|*|五段・ラ行|連用形)")
parser.add_argument("--dict_dir", type=str, default="/project/nakamura-lab08/Work/bin-wu/share/data/openjtalk_dict/open_jtalk_dic_utf_8-1.11", help='Compiled openjtalk dictionary.\ne.g., /project/nakamura-lab08/Work/bin-wu/share/data/openjtalk_dict/open_jtalk_dic_utf_8-1.11 (default)\nor /project/nakamura-lab08/Work/bin-wu/share/data/openjtalk_dict/openjtalk_default_unidic_ipadic (default, same as open_jtalk_dic_utf_8-1.11)\nor /project/nakamura-lab08/Work/bin-wu/share/data/openjtalk_dict/openjtalk_ipadic_unidic_neologd\nNote that the openjtalk_default_unidic_ipadic is compiled from the csv dicts from\n/project/nakamura-lab08/Work/bin-wu/installers/open_jtalk/open_jtalk-1.11/mecab-naist-jdic/{naist-jdic.csv,unidic-csj.csv}')
parser.add_argument("--user_dict", type=str, default=None, help="Specify an user dict: either a compiled dict of XXX.dic or a raw openjtalk dict of XXX.csv\n" + user_dict_example + "Note that the field of '1' might be cost and the field of '2/3' might be accent related for TTS.\nFields such as '名詞' and '一般' are necessary and take a finite set of options.")
parser.add_argument("--has_uttid", action="store_true", help="input in Kaldi script format: the first column is an uttid and the remaining is the content for a line.")
parser.add_argument("--csv2dic", type=str, default=None, help="Convert dict from a given XXX.csv to its XXX.dic and stop")
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
                fields = token.split(',')
                text_field = token.split(',')[0]
                pronun_field = token.split(',')[9] # default for mecab-naist-jdic
                if (not keep_accent): pronun_field = pronun_field.replace("’", "") # 接地|セッチ’ => 接地|セッチ

                if (pronun_field == u"、" and text_field != pronun_field): pronun_field = text_field # 。|、 (text of '。' has pronunciation of '、')

                if (text_field == u"．" and pronun_field == u"テン"): text_field = u"点" # ．|テン => 点|テン
                if (text_field == u"０" and pronun_field == u"ゼロ"): text_field = u"ゼロ" # ０|ゼロ => ゼロ|ゼロ
                if (text_field == u"０" and pronun_field == u"レー"): text_field = u"零" # ０|レー => 零|レー
                # if (text_field == u"一" and pronun_field == u"イッ"): text_field = u"イッ" # 一|イッ => イッ|イッ, but has effects of texts of 一本 (イッ本) and 一個 (イッ個)
                if (text_field == u"剝" and pronun_field == u"剝"): pronun_field = u"ム" # 剝|剝 => 剝|ム
                if (text_field == u"窩" and pronun_field == u"窩"): pronun_field = u"カ" # 窩|窩 => 窩|カ
                if (text_field == u"濡" and pronun_field == u"濡"): pronun_field = u"ヌ" # 濡|濡 => 濡|ヌ failure in mecab csj3.1
                if (text_field == u"繹" and pronun_field == u"繹"): pronun_field = u"エキ" # 繹|繹 => 繹|エキ
                if (text_field == u"駱" and pronun_field == u"駱"): pronun_field = u"ロ" # 駱|駱 => 駱|ロ mecab
                if (text_field == u"旒" and pronun_field == u"旒"): pronun_field = u"リュー" # 旒|旒 => 旒|リュー
                if (text_field == u"箇" and pronun_field == u"箇"): pronun_field = u"カ" # 箇|箇 => 箇|カ mecab
                if (text_field == u"廷" and pronun_field == u"廷"): pronun_field = u"テー" # 廷|廷 => 廷|テー
                if (text_field == u"忘記" and pronun_field == u"忘記"): pronun_field = u"ワスレキ" # 忘記|忘記 => 忘記|ワスレキ
                if (text_field == u"曖" and pronun_field == u"曖"): pronun_field = u"アイ" # 曖|曖 => 曖|アイ
                if (text_field == u"抽" and pronun_field == u"抽"): pronun_field = u"チュー" # 抽|抽 => 抽|チュー
                if (text_field == u"漱漱" and pronun_field == u"漱漱"): pronun_field = u"ソーソー" # 漱漱|漱漱 => 漱漱|ソーソー
                if (text_field == u"漱" and pronun_field == u"漱"): pronun_field = u"ソー" # 漱|漱 => 漱|ソー
                if (text_field == u"θ" and pronun_field == u"θ"): pronun_field = u"シータ" # θ|θ => θ|シータ
                if (text_field == u"陀" and pronun_field == u"陀"): pronun_field = u"ダ" # 陀|陀 => 陀|ダ
                if (text_field == u"貯" and pronun_field == u"貯"): pronun_field = u"タクワ" # 貯|貯 => 貯|タクワ
                if (text_field == u"購" and pronun_field == u"購"): pronun_field = u"コー" # 購|購 => 購|コー
                if (text_field == u"吸" and pronun_field == u"吸"): pronun_field = u"ス" # 吸|吸 => 吸|ス
                if (text_field == u"Λ" and pronun_field == u"Λ"): pronun_field = u"ラムダ" # Λ|Λ => Λ|ラムダ
                if (text_field == u"忘" and pronun_field == u"忘"): pronun_field = u"ワス" # 忘|忘 => 忘|ワス
                if (text_field == u"避" and pronun_field == u"避"): pronun_field = u"ヒ" # 避|避 => 避|ヒ
                if (text_field == u"該" and pronun_field == u"該"): pronun_field = u"ガイ" # 該|該 => 該|ガイ
                if (text_field == u"描" and pronun_field == u"描"): pronun_field = u"カ" # 描|描 => 描|カ
                if (text_field == u"彙" and pronun_field == u"彙"): pronun_field = u"イ" # 彙|彙 => 彙|イ
                if (text_field == u"βα" and pronun_field == u"βα"): pronun_field = u"アルファベータ" # βα|βα => βα|アルファベータ
                if (text_field == u"警警" and pronun_field == u"警警"): pronun_field = u"ケーケー" # 警警|警警 => 警警|ケーケー
                if (text_field == u"警" and pronun_field == u"警"): pronun_field = u"ケー" # 警|警 => 警|ケー
                if (text_field == u"閣型" and pronun_field == u"閣型"): pronun_field = u"カクガタ" # 閣型|閣型 => 閣型|カクガタ
                if (text_field == u"茶碗" and pronun_field == u"ヂャワン"): pronun_field = u"チャワン" # 茶碗|ヂャワン => 茶碗|チャワン
                if (text_field == u"以" and pronun_field == u"以"): pronun_field = u"イ" # 以|以 => 以|イ
                if (text_field == u"以以" and pronun_field == u"以以"): pronun_field = u"イイ" # 以以|以以 => 以以|イイ
                if (text_field == u"τ" and pronun_field == u"τ"): pronun_field = u"タウ" # τ|τ => τ|タウ
                if (text_field == u"Φ" and pronun_field == u"Φ"): pronun_field = u"ファイ" # Φ|Φ => Φ|ファイ
                if (text_field == u"φ" and pronun_field == u"φ"): pronun_field = u"ファイ" # φ|φ => φ|ファイ
                if (text_field == u"α" and pronun_field == u"α"): pronun_field = u"アルファ" # α|α => α|アルファ
                if (text_field == u"挿" and pronun_field == u"挿"): pronun_field = u"ソー" # 挿|挿 => 挿|ソー
                if (text_field == u"推" and pronun_field == u"推"): pronun_field = u"スイ" # 推|推 => 推|スイ
                if (text_field == u"棄" and pronun_field == u"棄"): pronun_field = u"キ" # 棄|棄 => 棄|キ
                if (text_field == u"抽" and pronun_field == u"抽"): pronun_field = u"チュー" # 抽|抽 => 抽|チュー
                if (text_field == u"批" and pronun_field == u"批"): pronun_field = u"ヒ" # 批|批 => 批|ヒ

                if (args.pos):
                    print(text_field + "|" + pronun_field + "|" + fields[1] + "|" + fields[2], end=" ")
                elif(args.full):
                    print(text_field + "|" + pronun_field + "|" + fields[1] + "|" + fields[2] + "|" + fields[3] + "|" + fields[4] + "|" + fields[5] + "|" + fields[6], end=" ")
                else:
                    print(text_field + "|" + pronun_field, end=" ")
        print()

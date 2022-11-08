# Implemented by bin-wu at 00:13 on 7th July 2022
#
# Install python warpper for macab:
# pip install mecab-python3

from typing import TextIO
import os
import io
import sys
import re
import argparse
import fileinput

import MeCab

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

def execute_command(command: str):
    """ Runs a kaldi job in the foreground and waits for it to complete; raises an
        exception if its return status is nonzero.  The command is executed in
        'shell' mode so 'command' can involve things like pipes.  Often,
        'command' will start with 'run.pl' or 'queue.pl'.  The stdout and stderr
        are merged with the calling process's stdout and stderr so they will
        appear on the screen.
        See also: get_command_stdout, background_command
    """
    import subprocess    
    p = subprocess.Popen(command, shell=True)
    p.communicate()
    if p.returncode != 0:
        raise Exception("Command exited with status {0}: {1}".format(
            p.returncode, command))
                
def add_chasen_format(dict_dir):
    """ Add chasen format for a dictionary (e.g., original unidict doesn't have chasen format). """
    has_chasen_format = False
    with open(os.path.join(dict_dir, "dicrc"), 'r') as f:
        for line in f:
            line = line.strip()
            if ("chasen" in line):
                has_chasen_format = True

    has_kana_chasen_format = False
    with open(os.path.join(dict_dir, "dicrc"), 'r') as f:
        for line in f:
            line = line.strip()
            if (line == r"node-format-chasen = %m\t%f[7]\t%f[6]\t%F-[0,1,2,3]\t%f[4]\t%f[5]\n"):
                has_kana_chasen_format = True

    unidic_chasen_format = r"""; ChaSen
node-format-chasen = %m\t%f[9]\t%f[8]\t%F-[0,1,2,3]\t%f[4]\t%f[5]\n
unk-format-chasen  = %m\t%m\t%m\t%F-[0,1,2,3]\t\t\n
eos-format-chasen  = EOS\n

; ChaSen (include spaces)
node-format-chasen2 = %M\t%f[9]\t%f[8]\t%F-[0,1,2,3]\t%f[4]\t%f[5]\n
unk-format-chasen2  = %M\t%m\t%m\t%F-[0,1,2,3]\t\t\n
eos-format-chasen2  = EOS\n"""

    if (not has_chasen_format and ("unidic-csj-3.1.0-full" in dict_dir or "unidic-mecab-2.1.2_src-neologd-20200910" in dict_dir)):
        print("{} does not include the chasen format".format(os.path.join(dict_dir, "dicrc")))
        print("Appending chasen_format:\n\n"  + unidic_chasen_format + "\n\nin {}".format(os.path.join(args.dict_dir, "dicrc")))
        with open(os.path.join(dict_dir, "dicrc"), 'a') as f:
            f.write("\n" + unidic_chasen_format + "\n")

    # kana to pronun:  上|ジョウ => 上|ジョー
    if (has_kana_chasen_format and ("mecab-ipadic-2.7.0-20070801" in dict_dir)):
        before1 = r"node-format-chasen = %m\t%f[7]\t%f[6]\t%F-[0,1,2,3]\t%f[4]\t%f[5]\n"
        after1 = r"node-format-chasen = %m\t%f[8]\t%f[6]\t%F-[0,1,2,3]\t%f[4]\t%f[5]\n"
        before2 = r"node-format-chasen2 = %M\t%f[7]\t%f[6]\t%F-[0,1,2,3]\t%f[4]\t%f[5]\n"
        after2 = r"node-format-chasen2 = %M\t%f[8]\t%f[6]\t%F-[0,1,2,3]\t%f[4]\t%f[5]\n"
        print("Convert kana to pronunciation for Chasen format of {}".format(os.path.join(dict_dir, "dicrc")))
        print("Substituting {}\nto {}".format(before1, after1))
        print("Substituting {}\nto {}".format(before2, after2)) 
        with fileinput.FileInput(os.path.join(dict_dir, "dicrc"), inplace=True, backup = ".bak") as f:
            for line in f:
                line = line.strip()
                if (line == before1):
                    print(after1)
                elif (line == before2):
                    print(after2)
                else:
                    print(line)
                
user_dict_example=r"""$ cat yonden.csv
表層形,左文脈ID,右文脈ID,コスト,品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音
柱上,,,,,,,,,,,チュウジョー,チュウジョー
中線,,,,,,,,,,,ナカセン,ナカセン
線色,,,,,,,,,,,センイロ,センイロ
$ # $mecab_dir/libexec/mecab/mecab-dict-index -d $dict_dir -f utf-8 -t utf-8 -u yonden.dic yonden.csv 
$ /project/nakamura-lab08/Work/bin-wu/.local/mecab/libexec/mecab/mecab-dict-index -d /project/nakamura-lab08/Work/bin-wu/share/data/mecab_dict/mecab-ipadic-2.7.0-20070801-neologd-20200910 -f utf-8 -t utf-8 -u yonden.dic yonden.csv
"""
parser = argparse.ArgumentParser(description=u"A python wrapper of mecab", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--input", type=str, default="-", help="name of file or '-' of stdin")
parser.add_argument("--output_format", type=str, default="default", help="the output formats include default, Ochasen, Oyomi, Odump, and Owakati,\nwhere a line of the default format is '次側' converting to '次|ツギ 側|ガワ'")
parser.add_argument("--pos", action="store_true", help="add fields of part of speech of '表層形|読み|品詞|品詞細分類1'\n(e.g., 柱|ハシラ|名詞|一般 上|ジョー|名詞|接尾 or 瑤泉院|ヨーゼーイン|名詞|固有名詞)")
parser.add_argument("--dict_dir", type=str, default="/project/nakamura-lab08/Work/bin-wu/share/data/mecab_dict/unidic-csj-3.1.0-full", help='the path that contains a dicrc file.\ne.g., /project/nakamura-lab08/Work/bin-wu/share/data/mecab_dict/mecab-ipadic-2.7.0-20070801-neologd-20200910\nor /project/nakamura-lab08/Work/bin-wu/share/data/mecab_dict/mecab-ipadic-2.7.0-20070801\nor /project/nakamura-lab08/Work/bin-wu/share/data/mecab_dict/unidic-csj-3.1.0-full (default)\nor /project/nakamura-lab08/Work/bin-wu/share/data/mecab_dict/unidic-mecab-2.1.2_src-neologd-20200910')
parser.add_argument("--user_dict", type=str, default=None, help="the user defined dictionary (XXX.dic).\nFor example, we can make a dictionary of 'yonden.dic' from 'yonden.csv' as follows:\n" + user_dict_example)
parser.add_argument("--has_uttid", action="store_true", help="input in Kaldi script format: the first column is an uttid and the remaining is the content for a line.")
parser.add_argument("--csv2dic", type=str, default=None, help="XXX.csv. Convert dict from XXX.csv to XXX.dic and stop\nNote: mecab installation is needed for converting the dictionary")
parser.add_argument("--mecab-dict-index", type=str, default="/project/nakamura-lab08/Work/bin-wu/.local/mecab/libexec/mecab/mecab-dict-index", help="path of mecab-dict-index inside installed mecab to covert XXX.csv to XXX.dic and stop.\nDefault: /project/nakamura-lab08/Work/bin-wu/.local/mecab/libexec/mecab/mecab-dict-index\nNote: mecab installation is needed for converting the dictionary (only activated with invoking option of csv2dic)\n")

args = parser.parse_args()

if args.csv2dic:
    dict_prefix, dict_postfix = os.path.splitext(args.csv2dic) # ('path/file', '.csv') or ('path/file', '.dic')
    dict_csv = dict_prefix + ".csv"
    dict_dic = dict_prefix + ".dic"
    assert dict_postfix == ".csv", "Error: option 'csv2dic' needs input to be 'XXX.csv', but now getting '{}'".format(args.csv2dic)

    command = "{} -d {} -f utf-8 -t utf-8 -u {} {}".format(args.mecab_dict_index, args.dict_dir, dict_dic, dict_csv)
    print(command)
    execute_command(command)
    sys.exit()

add_chasen_format(args.dict_dir)

if (args.output_format == "default"):
    mecab_output_format = "Ochasen" # We will convert from the Ochasen format to the default format
else:
    mecab_output_format = args.output_format

if (args.user_dict):
    tagger = MeCab.Tagger('-r/dev/null -{} -d "{}" -u "{}"'.format(mecab_output_format, args.dict_dir, args.user_dict))
else:
    tagger = MeCab.Tagger('-r/dev/null -{} -d "{}"'.format(mecab_output_format, args.dict_dir))
    
with readfile(args.input) as f:
     for line in f:
         line = line.strip()
         if (args.has_uttid):
             assert args.output_format != "Ochasen", "Ochasen output format does not support the option of 'has_uttid'"
             if (len(re.split('\s+', line)) <= 1): # empty line or uttid with empty content
                 uttid = line
                 line = ""
             else:
                 uttid, line = re.split('\s+', line, maxsplit=1)
             print(uttid, end=" ")
         result = tagger.parse(line)
         result = result.strip()
         parsed_lines = re.split('\n', result)
         for parsed_line in parsed_lines:
             # unidic does not have pronuciation for special symbols (e.g., '、\t\t、\t補助記号-読点\t\t' => '、\t、\t、\t補助記号-読点\t\t')
             parsed_line = re.sub("^([^\s])+\t\t", "\\1\t\\1\t", parsed_line)
             if (args.output_format == "default" or args.pos): # convert from the Ochasen format to the default format
                 if (parsed_line == "EOS"):
                     print()
                 else:
                     chasen_fields = re.split("\s+", parsed_line) # ['百', 'ヒャク', '百', '名詞-数詞', '']
                     if (chasen_fields[0] == u'っ' and chasen_fields[1] == u'っ'): # っ|っ|補助記号|一般
                         chasen_fields[1] = u'ッ'

                     pos_fields = re.split("-", chasen_fields[3])
                     if len(pos_fields) == 0: pos_fields = ["*", "*"]
                     elif len(pos_fields) == 1: pos_fields = [pos_fields[0], "*"]
                     else:  pos_fields = [pos_fields[0], pos_fields[1]]

                     if (args.pos): print(chasen_fields[0] + "|" + chasen_fields[1] + "|" + pos_fields[0] + "|" + pos_fields[1], end=" ")
                     else: print(chasen_fields[0] + "|" + chasen_fields[1], end=" ")
             else:
                 print(parsed_line)

# Implemented by bin-wu at 11:05 p.m. on 13th October 2022

from typing import TextIO
import io
import sys
import re
import argparse

import numpy as np

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

f_err = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8') # write utf8 stderr


parser = argparse.ArgumentParser(description=u"Replace the openjtalk transcription that has tokens starting with chouon with the mecab transcription. Assume the pronunciation is at the second field.\ne.g., イ|イ ーロンマスク|ーロンマスク => イー|イー ロン|ロン マスク|マスク\nあの|アノ ーホリケ|ーホリケ => あのー|アノー ホリケ|ホリケ",
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--openjtalk_text", type=str, default="exp/output/openjtalk_text_problem_openjtalk.txt", help="path to openjtalk transcription. (e.g., あの|アノ ーホリケ|ーホリケ or あの|アノ|連体詞|* ーホリケ|ーホリケ|フィラー|*")
parser.add_argument("--mecab_text", type=str, default="exp/output/openjtalk_text_problem_mecab_unidic_csj3.txt", help="path to mecab transcription. (e.g., あのー|アノー ホリケ|ホリケ or あのー|アノー|感動詞|フィラー ホリケ|ホリケ|名詞|普通名詞")
parser.add_argument("--output", type=str, default="-", help="name of file or '-' of stdout")
parser.add_argument("--in_sep", type=str, default="|", help="separator between fields of an token from input stream")
parser.add_argument("--has_uttid", action="store_true", help="input in Kaldi script format: the first column is an uttid and the remaining is the content for a line")
parser.add_argument("--verbose", action="store_true", help="verbose (to the stderr) each utterance that with replacement from openjtalk transcription to mecab ones")
parser.add_argument('--mode', type=str, choices=['text', 'kana'], default='text', help="Ignore the mecab replacements when kana/text length of transcriptions are different.\ne.g., openjtalk: 間|アイダ 吊っ|ツッ いこ|イコ う|ー vs. mecab 間|カン 吊っ|ツッ いこう|イコー\ntext model will deal with the case; kana mode will ignore the case. Default is text mode.")
args = parser.parse_args()

def parse_uttid_tokens(line: str):
    """ Parse the line into utterace id and the tokens. """
    if (len(re.split('\s+', line)) <= 1): # empty line or uttid with empty content
        uttid = line
        tokens = ""
    else:
        uttid, tokens = re.split('\s+', line, maxsplit=1)

    return uttid, tokens

def line_has_token_start_with_chouon(line: str, field_sep="|", chouon_symbol=u'ー'):
    """ Whether a line contains a token starts with chouon. """
    token_start_with_chouon = False
    for token in re.split('\s+', line):
        fields = token.split(field_sep)
        # output line format: uttid f1|f3 f1|f3 f1|f3
        assert len(fields) >= 2, "the tokens should have at least two fields such as あのー|アノー, but found '{}' in '{}'\n==> Maybe you should add the '--has_uttid' option if the transcription has uttids.".format(token, line)
        if (fields[1][0] == chouon_symbol):
            token_start_with_chouon = True

    return token_start_with_chouon

def identical_kana_length(line1: str, line2: str, token_sep="\s+", field_sep="|"):
    """ Whether two line contains the identical kana length.
    (e.g., "len(ビーシーエイ) == len(ビーシーエー)"

    This is a private function for the case that we prefer openjtalk kana representation
    if transcriptions of mecab and openjtalk with different kana length.
    We choose kana here. Kana transcriptions can be different in openjtalk and mecab cases.
    Alternatively we can also choose char. Char transcriptions also can be different.
    For example, in openjtalk auto-text-conversion of certain number, time, or day text.
    """
    token_start_with_chouon = False
    line1_kana = extract_fields_from_tokens(line1, field_index=1, in_token_sep=token_sep, field_sep=field_sep, out_token_sep=" ").replace(" ", "")
    line2_kana = extract_fields_from_tokens(line2, field_index=1, in_token_sep=token_sep, field_sep=field_sep, out_token_sep=" ").replace(" ", "")

    return (len(line1_kana) == len(line2_kana))

def identical_text_length(line1: str, line2: str, token_sep="\s+", field_sep="|"):
    """ Whether two line contains the identical text length.
    (e.g., "text_len(間|アイダ 吊っ|ツッ) == text_len(間|アイダ 吊っ|ツッ)".)
    """
    token_start_with_chouon = False
    line1_text = extract_fields_from_tokens(line1, field_index=0, in_token_sep=token_sep, field_sep=field_sep, out_token_sep=" ").replace(" ", "")
    line2_text = extract_fields_from_tokens(line2, field_index=0, in_token_sep=token_sep, field_sep=field_sep, out_token_sep=" ").replace(" ", "")

    return (len(line1_text) == len(line2_text))

def extract_fields_from_tokens(line: str, field_index=0, in_token_sep="\s+", field_sep="|", out_token_sep=" "):
    """ Extract a sequence of fields from line that is a sequence of tokens.

    Example:
    そしたら|ソシタラ いこ|イコ う|ー か|カ な|ナ => そしたら いこ う か な
    そ|ソ し|シ たら|タラ いこう|イコー か|カ な|ナ => そ し たら いこう か な
    """
    line_field = []
    for token in re.split(in_token_sep, line):
        fields = token.split(field_sep)
        # output line format: uttid f1|f3 f1|f3 f1|f3
        assert len(fields) >= 2, "the tokens should have at least two fields such as あのー|アノー, but found '{}' in '{}'\n==> Maybe you should add the '--has_uttid' option if the transcription has uttids.".format(token, line1)
        token_field = fields[field_index]
        line_field.append(token_field)

    return out_token_sep.join(line_field)

def extract_fields_index_from_tokens(line: str, field_index=0, token_sep="\s+", field_sep="|", ending_index=True):
    """ Extract a sequence of field indices from line that is a sequence of tokens, a mapping from token index to the field index.

    Example: when field_index is 1,
    そしたら|ソシタラ いこ|イコ う|ー か|カ な|ナ => 0 4 6 7 8 9
    そ|ソ し|シ たら|タラ いこう|イコー か|カ な|ナ => 0 1 2 4 7 8 9,
    where '9' is the ending index.
    """
    fields_index = [0]
    index_counter = 0
    for token in re.split(token_sep, line):
        fields = token.split(field_sep)
        # output line format: uttid f1|f3 f1|f3 f1|f3
        assert len(fields) >= 2, "the tokens should have at least two fields such as あのー|アノー, but found '{}' in '{}'\n==> Maybe you should add the '--has_uttid' option if the transcription has uttids.".format(token, line1)
        token_field = fields[field_index]
        index_counter += len(token_field)
        fields_index.append(index_counter)

    if (not ending_index): fields_index = fields_index[:-1]
    return fields_index

def index_of_first_token_start_with_chouon(line: str, field_sep="|", fail_to_found_index=-1, chouon_symbol=u'ー'):
    """ Get the index of the first token in a line that starts with chouon. Return -1 if search failure happens. """

    for index, token in enumerate(re.split('\s+', line)):
        fields = token.split(field_sep)
        # output line format: uttid f1|f3 f1|f3 f1|f3
        assert len(fields) >= 2, "the tokens should have at least two fields such as あのー|アノー, but found '{}' in '{}'\n==> Maybe you should add the '--has_uttid' option if the transcription has uttids.".format(token, line)
        if (fields[1][0] == chouon_symbol):
            return index

    return fail_to_found_index

def local_replace_openjtalk_with_mecab_single_replacement(tokens_openjtalk, tokens_mecab, field_index=1, chouon_symbol=u'ー'):
    """
    Find the minimum range inside mecab that contain the choun part of openjtalk.
    Replace the first opentjalk chouon part with mecab section.
    The function assumes that at least one replacement happens and it conducts the first replacement.

    Returns: the whole replaced string, (openjtalk_portion, mecab_portion)

    Usage:
    tokens_openjtalk = "そしたら|ソシタラ いこ|イコ う|ー か|カ な|ナ"
    tokens_mecab = "そ|ソ し|シ たら|タラ いこう|イコー か|カ な|ナ"
    print(local_replace_openjtalk_with_mecab_single_replacement(tokens_openjtalk, tokens_mecab))
    ('そしたら|ソシタラ いこう|イコー か|カ な|ナ', ('いこ|イコ う|ー', 'いこう|イコー'))

    For example:
    Indices of 'そしたら いこ う か な' from openjtalk is [0, 4, 6, 7, 8, 9] (token_index_to_kana_index)
    Indices of 'そ し たら いこう か な' from mecab is [0, 1, 2, 4, 7, 8, 9]
    Common indices should be [0, 4, 7, 8, 9]

    Here chouon token index in openjtalk is 2.
    Chouon kana index is 6, which lies in the smallest kana range (4, 7).
    Token range for openjtalk is (1, 3) and for mecab is (3, 4).
    Replace the openjtalk chouon part with mecab chouon part that has minimun range.
    Finally we get 'そしたら いこう か な'.

    The input token has at least two fields "text|kana"
    When field index is 1, operation on the kana level.
    When field index is 0, operation on the text level.
    """
    # kana_line_openjtalk = extract_fields_from_tokens(tokens_openjtalk, field_index=1) # そしたら いこ う か な
    # kana_line_mecab = extract_fields_from_tokens(tokens_mecab, field_index=1) # そ し たら いこう か な

    # token index to kana index (field index 1: kana level; field index 0: text level)
    token_index_to_kana_index_openjtalk = extract_fields_index_from_tokens(tokens_openjtalk, field_index=field_index)
    token_index_to_kana_index_mecab = extract_fields_index_from_tokens(tokens_mecab, field_index=field_index)
    assert token_index_to_kana_index_openjtalk[-1] == token_index_to_kana_index_mecab[-1],  "Openjtalk and mecab kana repr. should have the same length: {} vs. {}".format(tokens_openjtalk, tokens_mecab)
    common_kana_index = np.array(list(set(token_index_to_kana_index_openjtalk).intersection(set(token_index_to_kana_index_mecab))))

    # Get chouon token index
    token_index_chouon_openjtalk = index_of_first_token_start_with_chouon(tokens_openjtalk, chouon_symbol=chouon_symbol)
    assert token_index_chouon_openjtalk != -1, "Fail to find chouon. Opentjalk transcription should have at least one token that starts with a chouon: {}".format(tokens_openjtalk)

    # Get smallest kana index range
    kana_index_chouon_openjtalk = token_index_to_kana_index_openjtalk[token_index_chouon_openjtalk]
    kana_range_left = common_kana_index[common_kana_index <= kana_index_chouon_openjtalk].max()
    kana_range_right = common_kana_index[common_kana_index > kana_index_chouon_openjtalk].min()

    # Get smallest token index range
    token_range_openjtalk = (token_index_to_kana_index_openjtalk.index(kana_range_left), token_index_to_kana_index_openjtalk.index(kana_range_right))
    token_range_mecab = (token_index_to_kana_index_mecab.index(kana_range_left), token_index_to_kana_index_mecab.index(kana_range_right))

    token_list_openjtalk = re.split("\s+", tokens_openjtalk)
    token_list_mecab = re.split("\s+", tokens_mecab)
    left = " ".join(token_list_openjtalk[:token_range_openjtalk[0]])
    ori = " ".join(token_list_openjtalk[token_range_openjtalk[0]:token_range_openjtalk[1]])
    rep = " ".join(token_list_mecab[token_range_mecab[0]:token_range_mecab[1]])
    right = " ".join(token_list_openjtalk[token_range_openjtalk[1]:])
    replacement = " ".join([left, rep, right]).strip() # strip to remove the left or right spaces if the variables of left or right is empty.

    if (rep == ""):
        f_err.write(f"ERROR: The replacement part 'rep' should not be empty\n")

        # f_err.write(f"{tokens_openjtalk=}\n")
        # f_err.write(f"{tokens_mecab=}\n")
        # f_err.write(f"{left=}\n")
        # f_err.write(f"{rep=}\n")
        # f_err.write(f"{right=}\n")
        # f_err.write(f"{replacement=}\n")

        # f_err.write(f"{common_kana_index=}\n")
        # f_err.write(f"{kana_index_chouon_openjtalk=}\n")
        # f_err.write(f"{kana_range_left=}\n")
        # f_err.write(f"{kana_range_right=}\n")

        sys.exit(1)

    return replacement, (ori, rep)

def local_replace_openjtalk_with_mecab(tokens_openjtalk, tokens_mecab, field_index=1, chouon_symbol=u'ー'):
    """ Replace several portions of openjtalk that contain chouon-starting-tokens. Return the final replacement and a list of replaced parts.

    The input token has at least two fields "text|kana"
    When field index is 1, operation on the kana level.
    When field index is 0, operation on the text level.
    """
    replacement_list = []
    while (True):
        tokens_openjtalk, replaced_parts = local_replace_openjtalk_with_mecab_single_replacement(tokens_openjtalk, tokens_mecab, field_index=field_index, chouon_symbol=u'ー')
        replacement_list.append(replaced_parts)
        if (not line_has_token_start_with_chouon(tokens_openjtalk, chouon_symbol=u'ー')): break

    return tokens_openjtalk, replacement_list

openjtalk = []
with open(args.openjtalk_text) as f_openjtalk, open(args.mecab_text) as f_mecab, writefile(args.output) as f_output:
    for line_openjtalk, line_mecab in zip(f_openjtalk, f_mecab):
        line_openjtalk = line_openjtalk.strip()
        line_mecab = line_mecab.strip()
        result_line = ""

        # input line format: uttid f1|f2|f3 f1|f2|f3 f1|f2|f3
        if (args.has_uttid):
            uttid_openjtalk, tokens_openjtalk = parse_uttid_tokens(line_openjtalk)
            uttid_mecab, tokens_mecab = parse_uttid_tokens(line_mecab)
            assert uttid_openjtalk == uttid_mecab,  "The nth line of {} and {} should get same uttid, but find '{}' at the openjtalk file and '{}' at the mecab file".format(args.openjtalk_text, args.mecab_text, line_openjtalk, line_mecab)
            uttid = uttid_openjtalk
            result_line = uttid + " "
        else:
            tokens_openjtalk = line_openjtalk
            tokens_mecab = line_mecab

        token_start_with_chouon_openjtalk = line_has_token_start_with_chouon(tokens_openjtalk, chouon_symbol=u'ー')
        token_start_with_chouon_mecab = line_has_token_start_with_chouon(tokens_mecab, chouon_symbol=u'ー')
        has_identical_kana_length = identical_kana_length(tokens_openjtalk, tokens_mecab)
        has_identical_text_length = identical_text_length(tokens_openjtalk, tokens_mecab)

        replace_to_mecab = False
        if token_start_with_chouon_openjtalk:
            if (args.mode == "kana"):
                if (not has_identical_kana_length):
                    f_err.write("Found a token starting with a chouon 'ー': {}\n".format(result_line + tokens_openjtalk))
                    f_err.write("==> Warning1: mecab and openjtalk trans has diff. kana lengths (preserving openjtalk trans): mecab trans: {}\n".format(result_line + tokens_mecab))
                elif (token_start_with_chouon_mecab):
                    f_err.write("Found a token starting with a chouon 'ー': {}\n".format(result_line + tokens_openjtalk))
                    f_err.write("==> Warning2: mecab trans has the chouon 'ー' (preserving openjtalk trans): mecab trans: {}\n".format(result_line + tokens_mecab))
                    f_err.write("--\n")
                else:
                    if (args.verbose):
                        replace_to_mecab = True
                        final_replacement, replacement_list = local_replace_openjtalk_with_mecab(tokens_openjtalk, tokens_mecab, field_index=1, chouon_symbol=u'ー') # field_index=1 for kana
                        tokens_mecab = final_replacement
                        f_err.write("Found a token starting with a chouon 'ー': {}\n".format(result_line + tokens_openjtalk))
                        f_err.write("Replacing trans from openjtalk with mecab: {}\n".format(result_line + tokens_mecab))
                        for replacement in replacement_list:
                            f_err.write("### Replacement from openjtalk with mecab: {} => {}\n".format(replacement[0], replacement[1]))
                        f_err.write("--\n")
            else:
                assert args.mode == "text", "arg.mode should be either text or kana, but now it is '{}'".format(args.mode)
                if (not has_identical_text_length):
                    f_err.write("Found a token starting with a chouon 'ー': {}\n".format(result_line + tokens_openjtalk))
                    f_err.write("==> Warning1: mecab and openjtalk trans has diff. text lengths (preserving openjtalk trans): mecab trans: {}\n".format(result_line + tokens_mecab))
                elif (token_start_with_chouon_mecab):
                    f_err.write("Found a token starting with a chouon 'ー': {}\n".format(result_line + tokens_openjtalk))
                    f_err.write("==> Warning2: mecab trans has the chouon 'ー' (preserving openjtalk trans): mecab trans: {}\n".format(result_line + tokens_mecab))
                    f_err.write("--\n")
                else:
                    if (args.verbose):
                        replace_to_mecab = True
                        final_replacement, replacement_list = local_replace_openjtalk_with_mecab(tokens_openjtalk, tokens_mecab, field_index=0, chouon_symbol=u'ー') # field_index=0 for text
                        tokens_mecab = final_replacement
                        f_err.write("Found a token starting with a chouon 'ー': {}\n".format(result_line + tokens_openjtalk))
                        f_err.write("Replacing trans from openjtalk with mecab: {}\n".format(result_line + tokens_mecab))
                        for replacement in replacement_list:
                            f_err.write("### Replacement from openjtalk with mecab: {} => {}\n".format(replacement[0], replacement[1]))
                        f_err.write("--\n")

        if replace_to_mecab:
            result_line = result_line + tokens_mecab
        else:
            result_line = result_line + tokens_openjtalk

        f_output.write(result_line + " \n")

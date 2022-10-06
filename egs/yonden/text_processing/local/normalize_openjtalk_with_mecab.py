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

f_err = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8') # write utf8 stderr


parser = argparse.ArgumentParser(description=u"Replace the openjtalk transcription that has tokens starting with chouon or sokuon with the mecab transcription. Assume the pronunciation is at the second field.\ne.g., イ|イ ーロンマスク|ーロンマスク => イー|イー ロン|ロン マスク|マスク\nあの|アノ ーホリケ|ーホリケ => あのー|アノー ホリケ|ホリケ",
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--openjtalk_text", type=str, default="exp/output/openjtalk_text_problem_openjtalk.txt", help="path to openjtalk transcription. (e.g., あの|アノ ーホリケ|ーホリケ or あの|アノ|連体詞|* ーホリケ|ーホリケ|フィラー|*")
parser.add_argument("--mecab_text", type=str, default="exp/output/openjtalk_text_problem_mecab_unidic_csj3.txt", help="path to mecab transcription. (e.g., あのー|アノー ホリケ|ホリケ or あのー|アノー|感動詞|フィラー ホリケ|ホリケ|名詞|普通名詞")
parser.add_argument("--ignore_token_start_with_chouon", action="store_true", help="ignore the replacement of the openjtalk transcription with mecab transcription if any token starts with chouon")
parser.add_argument("--ignore_token_start_with_sokuon", action="store_true", help="ignore the replacement of the openjtalk transcription with mecab transcription if any token starts with sokuon")
parser.add_argument("--output", type=str, default="-", help="name of file or '-' of stdout")
parser.add_argument("--in_sep", type=str, default="|", help="separator between fields of an token from input stream")
parser.add_argument("--has_uttid", action="store_true", help="input in Kaldi script format: the first column is an uttid and the remaining is the content for a line")
parser.add_argument("--verbose", action="store_true", help="verbose (to the stderr) each utterance that with replacement from openjtalk transcription to mecab ones")
parser.add_argument("--silence", action="store_true", help="without any message (to the stderr) of replacement details")
args = parser.parse_args()

def parse_uttid_tokens(line: str):
    """ Parse the line into utterace id and the tokens """
    if (len(re.split('\s+', line)) <= 1): # empty line or uttid with empty content
        uttid = line
        tokens = ""
    else:
        uttid, tokens = re.split('\s+', line, maxsplit=1)

    return uttid, tokens

def line_has_token_start_with_chouon(line: str):
    """ whether a line contains a token starts with chouon """
    token_start_with_chouon = False
    for token in re.split('\s+', line):
        fields = token.split(args.in_sep)
        # output line format: uttid f1|f3 f1|f3 f1|f3
        assert len(fields) >= 2, "the tokens should have at least two fields such as あのー|アノー, but found '{}' in '{}'\n==> Maybe you should add the '--has_uttid' option if the transcription has uttids.".format(token, line_openjtalk)
        if (fields[1][0] == u'ー'):
            token_start_with_chouon = True

    return token_start_with_chouon

def line_has_token_start_with_sokuon(line: str):
    """ whether a line contains a token starts with sokuon """
    token_start_with_sokuon = False
    for token in re.split('\s+', line):
        fields = token.split(args.in_sep)
        # output line format: uttid f1|f3 f1|f3 f1|f3
        assert len(fields) >= 2, "the tokens should have at least two fields such as あのー|アノー, but found '{}' in '{}'\n==> Maybe you should add the '--has_uttid' option if the transcription has uttids.".format(token, line_openjtalk)
        if (fields[1][0] == u'ッ'):
            token_start_with_sokuon = True
        
    return token_start_with_sokuon

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

        token_start_with_chouon_openjtalk = line_has_token_start_with_chouon(tokens_openjtalk)
        token_start_with_sokuon_openjtalk = line_has_token_start_with_sokuon(tokens_openjtalk)
        token_start_with_chouon_mecab = line_has_token_start_with_chouon(tokens_mecab)
        token_start_with_sokuon_mecab = line_has_token_start_with_sokuon(tokens_mecab)

        replace_to_mecab = False
        if (not args.ignore_token_start_with_chouon) and token_start_with_chouon_openjtalk:
            replace_to_mecab = True
            if (args.verbose):
                f_err.write("Found a token starting with a chouon 'ー': {}\n".format(result_line + tokens_openjtalk))
                f_err.write("Replacing trans from openjtalk with mecab: {}\n".format(result_line + tokens_mecab))
                f_err.write("--\n")
            if (token_start_with_chouon_mecab and (not args.silence)):
                f_err.write("==> Warning: mecab trans has the chouon 'ー': {}\n".format(result_line + tokens_mecab))
                f_err.write("--\n")

        if (not args.ignore_token_start_with_sokuon) and token_start_with_sokuon_openjtalk:
            replace_to_mecab = True
            if (args.verbose):
                f_err.write("Found a token starting with a sokuon 'ッ': {}\n".format(result_line + tokens_openjtalk))
                f_err.write("Replacing trans from openjtalk with mecab: {}\n".format(result_line + tokens_mecab))
                f_err.write("--\n")
            if (token_start_with_sokuon_mecab and (not args.silence)):
                f_err.write("==> Warning: mecab trans has the sokuon 'ッ': {}\n".format(result_line + tokens_mecab))
                f_err.write("--\n")
        
        if replace_to_mecab:
            result_line = result_line + tokens_mecab
        else:
            result_line = result_line + tokens_openjtalk

        f_output.write(result_line + "\n")

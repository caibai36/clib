#!/usr/bin/env python3

# re-implemented text2token.py by bin-wu at 11:34 in 2018.12.24

from typing import List, NamedTuple, Optional
import sys
import re
import io
import argparse

example_string = '''
Example:
    (base) [bin-wu@ahccsclm03 s0]$ text=cutils/tests/data/text
    (base) [bin-wu@ahccsclm03 s0]$ non_ling_sym=cutils/tests/data/non_ling_sym.txt
    (base) [bin-wu@ahccsclm03 s0]$ cat $text
    4k0c0201 THE SEVEN UNITS TO BE OFFERED FOR SALE HAVE A WORK FORCE OF ABOUT TWENTY THOUSAND
    4k0c0202 AND ALLIED SIGNAL EMPLOYS A TOTAL OF ABOUT ONE HUNDRED AND FORTY THOUSAND WORKERS
    (base) [bin-wu@ahccsclm03 s0]$ cat $non_ling_sym
    AND
    ABOUT
    (base) [bin-wu@ahccsclm03 s0]$ cutils/text2token.py -n 2 -s 2 -l $non_ling_sym $text
    4k0c0201 THE SE VE N<space> UN IT S<space> TO <space>B E<space> OF FE RE D<space> FO R<space> SA LE <space>H AV E<space> A<space> WO RK <space>F OR CE <space>O F<space> ABOUT<space> TW EN TY <space>T HO US AND
    4k0c0202 AND AL LI ED <space>S IG NA L<space> EM PL OY S<space> A<space> TO TA L<space> OF <space>ABOUT <space>O NE <space>H UN DR ED <space>AND <space>F OR TY <space>T HO US AND<space> WO RK ER S
    (base) [bin-wu@ahccsclm03 s0]$ chars_del=cutils/tests/data/char_del.txt;chars_rep=cutils/tests/data/char_rep.txt
    (base) [bin-wu@ahccsclm03 s0]$ cat $chars_del
    V
    T
    (base) [bin-wu@ahccsclm03 s0]$ cat $chars_rep
    T t
    O o
    ABOUT <UNK>
    (base) [bin-wu@ahccsclm03 s0]$ cutils/text2token.py -n 2 -s 2 -l $non_ling_sym -d $chars_del -r $chars_rep $text 
4k0c0201 THE SE EN <space>U NI S<space> o<space> BE <space>o FF ER ED <space>F oR <space>S AL E<space> HA E<space> A<space> Wo RK <space>F oR CE <space>o F<space> <UNK><space> WE NY <space>H oU SAND
4k0c0202 AND AL LI ED <space>S IG NA L<space> EM PL oY S<space> A<space> oA L<space> oF <space><UNK> <space>o NE <space>H UN DR ED <space>AND <space>F oR Y<space> Ho US AND<space> Wo RK ER S
'''


class Pos(NamedTuple):
    """ the matching position of a string. """
    start: int
    end: int


def is_non_lang_syms_index(matched_pos: List[Pos], cur_pos: int) -> Optional[Pos]:
    for pos in matched_pos:
        if pos.start <= cur_pos and cur_pos < pos.end:
            return pos
    return None  # if NOT in non_lang_syms_index


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=example_string)
    parser.add_argument('text', type=str, default=False, nargs='?',
                        help='input text')
    parser.add_argument('--non-lang-syms', '-l', default=None, type=str,
                        help='list of non-linguistic symbols with format <XXX>, e.g., <NOISE> etc.')
    parser.add_argument('--skip-ncols', '-s', default=0, type=int,
                        help='skip first n columns')
    parser.add_argument('--space', default='<space>', type=str,
                        help='space symbol')
    parser.add_argument('--nchar', '-n', default=1, type=int,
                        help='number of characters to split, i.e., \
                        aabb -> a a b b with -n 1 and aa bb with -n 2')
    parser.add_argument('--chars-delete', '-d', default="", type=str,
                        help="file contains characters to be deleted at text. eg. ' or <")
    parser.add_argument('--chars-replace', '-r', default="", type=str,
                        help="file contains character pairs to be replaced at text. eg. O o or <*IN*> <UNK>")
    parser.add_argument('--chars-replace-sep', default=" ", type=str,
                        help="separator of the character pairs in the chars-replace file")
    
    args = parser.parse_args()

    if args.text:
        f = open(args.text, 'r', encoding='utf-8')
    else:
        # f = sys.stdin
        f = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8') # read utf8 stdin

    if args.chars_delete:
        with open(args.chars_delete, 'r', encoding='utf-8') as df:
            chars_del = {char.strip() for char in df}

    if args.chars_replace:
        with open(args.chars_replace, 'r', encoding='utf-8') as rf:
            chars_rep = {re.split(args.chars_replace_sep, pair.strip())[0]:re.split(args.chars_replace_sep, pair.strip())[1] for pair in rf}

    # Create patterns of the non_lang_syms.
    patterns: List = []
    if args.non_lang_syms:
        with open(args.non_lang_syms, 'r', encoding='utf-8') as nf:
            for symbol in nf:
                pattern = re.compile(re.escape(symbol.strip()))
                patterns.append(pattern)

    for line in f:
        # tokens = line.strip().split()
        tokens = re.split("\s+", line.strip())
        # Split the first skip_ncols tokens, take the remaining
        skipped_str = " ".join(tokens[:args.skip_ncols])
        remained_str = " ".join(tokens[args.skip_ncols:])

        # print(skipped_str, end=" ")
        sys.stdout.buffer.write((skipped_str + " ").encode('utf8')) # print utf8 string

        # First find the matched positions of the non_lang_syms
        matched_pos: List[Pos] = []

        # Note: the non_lang_syms should be in some special format like <...>, or there will be bugs:
        #       eg. THOUSAND is a substring of the remained_str; AND is in non_lang_syms
        for pattern in patterns:
            start_pos: int = 0
            while True:
                match = pattern.search(remained_str, start_pos)
                if match:
                    matched_pos.append(Pos(match.start(), match.end()))
                    start_pos = match.end()
                    if start_pos == len(remained_str):
                        break
                else:
                    break

        # Split the line into list of character strings.
        # We will treat the non_lang_syms as a character;
        # white space is also treated as a character.
        start_pos: int = 0
        chars: List[str] = []
        while start_pos < len(remained_str):
            p: Optional[Pos] = is_non_lang_syms_index(matched_pos, start_pos)
            if p:  # is non_lang_syms_index
                chars.append(remained_str[p.start:p.end])
                start_pos = start_pos + p.end - p.start
            else:
                chars.append(remained_str[start_pos])
                start_pos += 1
        # replace the whitespace chars with <space>
        chars = [char.replace(" ", "<space>") for char in chars]
        # delete and replace chars
        if args.chars_delete:
            chars = [char for char in chars if char not in chars_del]
        if args.chars_replace:
            chars = list(map(lambda char: char if char not in chars_rep else chars_rep[char], chars))
        # concatenate the chars as char strings
        char_strings = ["".join(chars[i:i+args.nchar])
                        for i in range(0, len(chars), args.nchar)]

        # print(" ".join(char_strings))
        sys.stdout.buffer.write((" ".join(char_strings)+'\n').encode('utf-8')) # print utf8 string

if __name__ == "__main__":
    main()

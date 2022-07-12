#!/usr/bin/env python3

# The code comes from hayashi san with an original name convert_numeral_a2c.py at 17:03 on 2022.10.05

###
### convert_numerals.py
### -- Yet another numeral expression converter
###


import sys, os, io
from argparse import ArgumentParser
import codecs

# UTF8にエンコード
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


mydir = os.path.dirname (os.path.realpath(__file__))
sys.path.append ("%s/Kanjize" % mydir)
from kanjize import int2kanji, kanji2int


def read_fp (filename):
    if filename in ["-", "/dev/stdin"]:
        return sys.stdin
    else:
        return open (filename, "rt")

def write_fp (filename):
    if filename in ["-", "/dev/stdout"]:
        return sys.stdout
    else:
        return open (filename, "wt")


class NumeralConverter (object):
    def __init__ (self, ifp, ofp, debug=False):
        self._name = None
        self._ifp = ifp
        self._ofp = ofp
        self._debug = debug

    def __call__ (self):
        line = self._ifp.readline()
        # 20200704 特殊文字があると落ちるので除去
        line = line.encode("cp932","ignore").decode("cp932")
        if not line: raise StopIteration
        else: line = line.rstrip("\r\n")
        if self._debug:
            print ("[input] {}".format(line), file=sys.stderr)
        print (self.convert(line), file=self._ofp)


class Arabic2ChineseConverter (NumeralConverter):
    def __init__ (self, ifp, ofp, debug=False):
        super().__init__(ifp, ofp, debug=debug)
        self._name = "a2c"

    def convert (self, line):
        def get_digit_group (bufarray, i):
            if i+2 < len(bufarray) and bufarray[i+1] in [",", "，"] and bufarray[i+2][-1].isdigit():
                ret = get_digit_group (bufarray, i+2)
                if ret:
                    return [bufarray[i], bufarray[i+1]] + ret
            else:
                return [bufarray[i]]

        bufarray = []
        if "電話" in line \
            or "℡" in line:
            ### Stupid heuristics--- if there is an expression implying a telephone number, no digit grouping is applied.
            bufarray = [c for c in line]
        else:
            for c in line:
                if c.isdigit() and len(bufarray) > 0 and bufarray[-1][-1].isdigit():
                    bufarray[-1] += c
                else:
                    bufarray.append (c)

            i = 0
            while i < len(bufarray)-2:
                if bufarray[i][-1].isdigit():
                    ret = get_digit_group (bufarray, i)
                    if len(ret) > 1:
                        flag = True
                        followings = [ret[x] for x in range (0, len(ret), 2)]
                        if False in [len(x) == 3 for x in followings[1:]]:
                            ### Each following digit group must consist of exactly three digits. 
                            flag = False
                        elif True in [x[0] in ["0", "０"] for x in followings]:
                            ### We can identify the groups represent one number when one of the following digit group starts with zero
                            flag = True
                        elif False not in [int(followings[i-1])+1 == int(followings[i]) for i in range (1, len(followings))]:
                            ### We consider the groups is just a coordination when they are simple incremented numbers
                            flag = False

                        if flag:
                            bufarray[i] = "".join(ret).replace("，",",").replace(",","")
                            for _ in range(len(ret)-1):
                                bufarray.pop(i+1)
                        else:
                            i += len(ret)-1
                i += 1

        for i in range(len(bufarray)):
            if bufarray[i][-1].isdigit():
                if bufarray[i] in ["0", "０"]:
                    value = "〇"
                else:
                    value = int2kanji ( int(bufarray[i]) )
                if self._debug: print ("{} ---> {}".format(bufarray[i], value), file=sys.stderr)
                bufarray[i] = value

        return "".join(bufarray)


class Chinese2ArabicConverter (NumeralConverter):
    def __init__ (self, ifp, ofp, separator=False, debug=False):
        super().__init__(ifp, ofp, debug=debug)
        self._name = "c2a"
        self._separator = separator
        self._numerals = {
                            '〇': 0,
                            '一': 1,
                            '二': 2,
                            '三': 3,
                            '四': 4,
                            '五': 5,
                            '六': 6,
                            '七': 7,
                            '八': 8,
                            '九': 9,
                            '十': 10,
                            '百': 100,
                            '千': 1000,
                            '万': 10000,
                            '億': 100000000,
                            '兆': 1000000000000,
                            '京': 10000000000000000,
                            '垓': 100000000000000000000
                         }

    def _isnumeral (self, c):
        return (c in self._numerals)

    def convert (self, line):
        bufarray = []
        for c in line:
            if c == '〇':
                c = "0"
            if self._isnumeral(c) and len(bufarray) > 0 and self._isnumeral(bufarray[-1][-1]):
                bufarray[-1] += c
            else:
                bufarray.append (c)

        for i in range(len(bufarray)):
            if len(bufarray[i][-1]) > 0 and self._isnumeral(bufarray[i][-1]):
                value = kanji2int(bufarray[i])
                if self._separator:
                    if self._debug: print ("{} ---> {:,}".format(bufarray[i], value), file=sys.stderr)
                    bufarray[i] = "{:,}".format(value)
                else:
                    if self._debug: print ("{} ---> {}".format(bufarray[i], value), file=sys.stderr)
                    bufarray[i] = "{}".format(value)

        return "".join(bufarray)

def main ():
    name = os.path.basename(__file__)
    c2a = (name == "convert_numerals_c2a.py")
    a2c = (name == "convert_numerals_a2c.py")

    # Convert Arabic numbers to Chinese characters
    a2c = True

    argparser = ArgumentParser(description="Convert Arabic numbers to Chinese characters in Japanese text processing.")
    argparser.add_argument ("-d", "--debug", action="store_true", help="enable debug outputs into stderr")
    argparser.add_argument ("--separator", action="store_true", help="insert commas as thousands separators")
    if not c2a and not a2c:
        direction_group = argparser.add_mutually_exclusive_group(required=True)
        direction_group.add_argument ("-a", "--c2a", action="store_true", help="convert Chinese numerals into Arabic")
        direction_group.add_argument ("-c", "--a2c", action="store_true", help="convert Arabic numerals into Chinese")
    argparser.add_argument ("input",  type=str, nargs="?", metavar="INPUT_FILE", default="-", help="input file ('-' stands for stdin)")
    argparser.add_argument ("output", type=str, nargs="?", metavar="OUTPUT_FILE", default="-", help="output file ('-' stands for stdout)")
    args = argparser.parse_args()
    if 'direction_group' not in locals():
        args.c2a = c2a
        args.a2c = a2c


    with read_fp (args.input) as ifp, write_fp (args.output) as ofp:
        if   args.c2a: processor = Chinese2ArabicConverter(ifp, ofp, separator=args.separator, debug=args.debug)
        elif args.a2c: processor = Arabic2ChineseConverter(ifp, ofp, debug=args.debug)

        try:
            while True:
                processor()
        except StopIteration:
            pass
        except BaseException as e:
            raise e

if __name__ == "__main__":
    try:
        main ()
    except KeyboardInterrupt:
        sys.exit (1)
    except BaseException as e:
        raise e

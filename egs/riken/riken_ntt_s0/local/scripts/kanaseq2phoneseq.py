#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert kana sequences to phone sequences in a text or a Kaldi script file.

This script converts Japanese kana sequences to their corresponding phone sequences
using a kana-to-phone mapping file. It handles special characters like punctuation
and can process both plain text and Kaldi script files with utterance IDs.

The original kana2phone mapping comes from kaldi/egs/csj/s5/local/csj_make_trans/kana2phone
"""

import argparse
import sys
import os
from typing import Tuple, List, Dict


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert kana sequences to phone sequences in a text or Kaldi script file.",
        epilog=("Note: ' ' between words converts to <space>, '、' converts to <comma>, "
                "'。' converts to <period>, and '？' converts to <question_mark>. "
                "Conversion errors are printed to stderr.")
    )
    parser.add_argument(
        "input_file", 
        nargs="?", 
        type=argparse.FileType("r", encoding="utf-8"),
        default=sys.stdin,
        help="Input file containing kana sequences (default: stdin)"
    )
    parser.add_argument(
        "--has_uttid", "-k", 
        action="store_true", 
        help="Input is Kaldi format with utterance ID in the first column"
    )
    parser.add_argument(
        "--kana2phone", "-p", 
        type=str, 
        help="Path to kana-to-phone mapping file (default: built-in mapping)"
    )
    parser.add_argument(
        "--print_kana", 
        action="store_true", 
        help="Print the kana instead of phones"
    )
    
    return parser.parse_args()


def load_kana2phone(kana2phone_file: str = None) -> Dict[str, str]:
    """
    Load kana-to-phone mapping from file or use built-in mapping.
    
    Args:
        kana2phone_file: Path to kana2phone mapping file
        
    Returns:
        Dictionary mapping kana characters to their phone representations
    """
    kana2phone = {}
    
    # If no file is provided, use the built-in mapping
    if kana2phone_file is None:
        # This is the hard-coded version of the kana2phone mapping
        builtin_mapping = """ア+a 
イ+i 
ウ+u 
エ+e 
オ+o 
カ+k a 
キ+k i 
ク+k u 
ケ+k e 
コ+k o 
サ+s a 
ス+s u 
セ+s e 
ソ+s o 
タ+t a 
テ+t e 
ト+t o 
ナ+n a 
ニ+n i 
ヌ+n u 
ネ+n e 
ノ+n o 
ハ+h a 
ヒ+h i 
フ+f u 
ヘ+h e 
ホ+h o 
マ+m a 
ミ+m i 
ム+m u 
メ+m e 
モ+m o 
ヤ+y a 
ユ+y u 
ヨ+y o 
ラ+r a 
リ+r i 
ル+r u 
レ+r e 
ロ+r o 
ワ+w a 
ン+N 
ガ+g a 
ギ+g i 
グ+g u 
ゲ+g e 
ゴ+g o 
ザ+z a 
ジ+j i 
ズ+z u 
ゼ+z e 
ゾ+z o 
ダ+d a 
ヂ+j i 
ヅ+z u 
デ+d e 
ド+d o 
バ+b a 
ビ+b i 
ブ+b u 
ベ+b e 
ボ+b o 
パ+p a 
ピ+p i 
プ+p u 
ペ+p e 
ポ+p o 
ー+: 
ッ+q 
トゥ+t u 
ジェ+j e 
ツァ+ts a 
ヴォ+b o 
ツィ+ts i 
キャ+ky a 
キュ+ky u 
キョ+ky o 
シャ+sh a 
シュ+sh u 
シェ+sh e 
ショ+sh o 
チャ+ch a 
チュ+ch u 
チェ+ch e 
チョ+ch o 
ツェ+ts e 
ニャ+ny a 
ニュ+ny u 
ニョ+ny o 
ヒャ+hy a 
ヒュ+hy u 
ヒョ+hy o 
ミャ+my a 
ミュ+my u 
ミョ+my o 
リャ+ry a 
リュ+ry u 
リョ+ry o 
ギャ+gy a 
ギュ+gy u 
ギョ+gy o 
ビャ+by a 
ビュ+by u 
ビョ+by o 
ヂュ+dy u 
ピャ+py a 
ピュ+py u 
ピョ+py o 
ヲ+o 
ティ+t i 
ファ+f a 
フィ+f i 
フェ+f e 
フォ+f o 
ジャ+j a 
ジュ+j u 
ジョ+j o 
ディ+d i 
デュ+d u 
ウェ+w e 
ウィ+w i 
ヴァ+b a 
ヴィ+b i 
ヴェ+b e 
ウォ+w o 
ズィ+j i 
ジァ+j a 
ドゥ+d u 
フョ+hy o 
フュ+hy u 
イェ+i e 
ツォ+ts o 
ニェ+n e 
ヒェ+h e 
ブィ+b i 
ミェ+m e 
クヮ+k a 
グヮ+g a 
スィ+sh i 
テュ+ts u 
ヴ+b u 
ツ+ts u 
シ+sh i 
チ+ch i 
ヮ+w a"""
        
        for line in builtin_mapping.splitlines():
            if line.strip():
                kana, phone = line.split('+')
                kana2phone[kana] = phone
    else:
        # Load from external file
        with open(kana2phone_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    kana, phone = line.split('+')
                    kana2phone[kana] = phone
    
    # Add punctuation mappings
    kana2phone["、"] = "<comma> "
    kana2phone["。"] = "<period> "
    kana2phone["？"] = "<question_mark> "
    
    return kana2phone


def kanaseq2phoneseq(kanaseq: str, kana2phone: Dict[str, str]) -> Tuple[str, bool]:
    """
    Convert a kana sequence to a phone sequence.
    Long vowels (vowel followed by 'ー') are merged into a single phone.
    
    Args:
        kanaseq: A string of kana characters
        kana2phone: Dictionary mapping kana to phone
        
    Returns:
        Tuple of (phone_sequence, error_flag)
    """
    flg = False
    phoneseq = ""
    syllable = ""
    chars = list(kanaseq)
    
    # First build syllables
    syllables = []
    
    for char in chars:
        if char in "ァィゥェォャュョ":
            # These are part of the previous syllable
            syllable += char
        else:
            if not syllable:
                syllable = char
            else:
                # Add completed syllable and start a new one
                syllables.append(syllable)
                syllable = char
    
    # Don't forget the last syllable
    if syllable:
        syllables.append(syllable)
    
    # Convert each syllable to phone
    for i, syllable in enumerate(syllables):
        if syllable in kana2phone:
            # If it's a long vowel marker ("ー"), merge with previous syllable
            if syllable == "ー" and i > 0:
                # Remove the trailing space from the previous phone if it exists
                if phoneseq and phoneseq[-1] == " ":
                    phoneseq = phoneseq[:-1]
                phoneseq += ": "  # Add the long vowel marker
            else:
                phoneseq += kana2phone[syllable]
        else:
            flg = True
            phoneseq += syllable
    
    # Check for invalid sequences
    if phoneseq in [": ", "q "]:
        flg = True
    
    return phoneseq, flg


def kanaseq_splitter(kanaseq: str, kana2phone: Dict[str, str]) -> Tuple[str, bool]:
    """
    Split kana sequences and preserve compound characters.
    Long vowels (vowel followed by 'ー') are merged into a single phone.
    This replicates the behavior of kanaseq_splitter.pl.
    
    Args:
        kanaseq: A string of kana characters
        kana2phone: Dictionary used as a set of valid kana
        
    Returns:
        Tuple of (splitted_kana_sequence, error_flag)
    """
    flg = False
    splitted_kanaseq = ""
    syllable = ""
    chars = list(kanaseq)
    
    # First build syllables
    syllables = []
    
    for char in chars:
        if char in "ァィゥェォャュョ":
            # These are part of the previous syllable
            syllable += char
        else:
            if not syllable:
                syllable = char
            else:
                # Add completed syllable and start a new one
                syllables.append(syllable)
                syllable = char
    
    # Don't forget the last syllable
    if syllable:
        syllables.append(syllable)
    
    # Process each syllable
    for i, syllable in enumerate(syllables):
        if syllable in kana2phone:
            # Handle special characters
            if syllable == "、":
                splitted_kanaseq += "<comma> "
            elif syllable == "。":
                splitted_kanaseq += "<period> "
            elif syllable == "？":
                splitted_kanaseq += "<question_mark> "
            # Handle long vowel marker, merge with previous syllable
            elif syllable == "ー" and i > 0:
                # Get the last part and merge with ー
                if splitted_kanaseq.endswith(" "):
                    splitted_kanaseq = splitted_kanaseq[:-1] + "ー "
            # For compound characters, preserve them
            elif any(char in "ァィゥェォャュョ" for char in syllable):
                splitted_kanaseq += syllable + " "
            # For single characters, add a space after each
            else:
                splitted_kanaseq += " ".join(list(syllable)) + " "
        else:
            flg = True
            splitted_kanaseq += syllable + " "
    
    # Check for invalid sequences
    if splitted_kanaseq in ["ー ", "ッ "]:
        flg = True
    
    return splitted_kanaseq.rstrip(), flg


def main():
    """Main function to process input and convert kana to phones."""
    args = parse_arguments()
    
    # Load kana-to-phone mapping
    kana2phone = load_kana2phone(args.kana2phone)
    
    line_num = 0
    error_lines = []
    
    # Process each line in the input
    for line in args.input_file:
        line = line.strip()
        if line.startswith('#'):
            continue
            
        line_num += 1
        
        # Parse the line based on format
        if args.has_uttid:
            parts = line.split(' ', 1)
            if len(parts) < 2:
                continue
            uttid, content = parts
        else:
            uttid = None
            content = line
        
        # Split the content into tokens
        tokens = content.split()
        
        result_tokens = []
        for token in tokens:
            if args.print_kana:
                # Use kana splitter to match original format
                result_seq, flag = kanaseq_splitter(token, kana2phone)
            else:
                # Convert to phones
                result_seq, flag = kanaseq2phoneseq(token, kana2phone)
            
            if flag:
                error_lines.append(f"Warning: Line_num: {line_num} Token: {token} => {result_seq} in Line: {line}")
            
            # Remove trailing whitespace
            result_seq = result_seq.rstrip()
            result_tokens.append(result_seq)
        
        # Output the result
        if args.has_uttid:
            print(f"{uttid} {' <space> '.join(result_tokens)}")
        else:
            print(f"{' <space> '.join(result_tokens)}")
    
    # Print any conversion errors to stderr
    if error_lines:
        print("\n".join(error_lines), file=sys.stderr)
        print("", file=sys.stderr)  # Print an extra newline if there were any errors


if __name__ == "__main__":
    main()

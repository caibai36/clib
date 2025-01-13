#!/bin/bash
# Convert both corpus files using pinyin2phone.py - with and without tones

# PD99 corpus - with tones
python local/pinyin2phone.py --corpus conf/data/PD99ToneSeg.txt --phoneme_corpus exp/data/pd99_phoneme_tone.txt

# PD99 corpus - without tones
python local/pinyin2phone.py --corpus conf/data/PD99ToneSeg.txt --phoneme_corpus exp/data/pd99_phoneme.txt --no_tone

# TV07 corpus - with tones
python local/pinyin2phone.py --corpus conf/data/TV07ToneSeg.txt --phoneme_corpus exp/data/tv07_phoneme_tone.txt

# TV07 corpus - without tones
python local/pinyin2phone.py --corpus conf/data/TV07ToneSeg.txt --phoneme_corpus exp/data/tv07_phoneme.txt --no_tone

# (mlp) [bin-wu@s186 riken_fl_s0]$(master *) head -n 323020 exp/data/pd99_phoneme.txt > exp/data/pd99_phoneme_part.txt # same amount as tv
# (mlp) [bin-wu@s186 riken_fl_s0]$(master *) head -n 323020 exp/data/pd99_phoneme_tone.txt > exp/data/pd99_phoneme_tone_part.txt

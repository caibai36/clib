#!/bin/bash
set -x # -x 'print commands',

# Openjtalk can convert complex numbers to their right pronunciation
cat data/text/openjtalk_text_num.scp
# cat openjtalk_text_num.scp | python local/scripts/openjtalk.py --has_uttid
python local/scripts/openjtalk.py --has_uttid --input data/text/openjtalk_text_num.scp

# Without a user dictionary
cat data/text/openjtalk_text_hard.scp
python local/scripts/openjtalk.py --has_uttid --input data/text/openjtalk_text_hard.scp

# With a user dictionary
cat data/dict/yonden_test_openjtalk.csv 
python local/scripts/openjtalk.py --csv2dic data/dict/yonden_test_openjtalk.csv
python local/scripts/openjtalk.py --has_uttid --user_dict data/dict/yonden_test_openjtalk.dic --input data/text/openjtalk_text_hard.scp

# Without a user dictionary
cat data/text/openjtalk_text_hard.scp
python local/scripts/mecab.py --has_uttid --input data/text/openjtalk_text_hard.scp

# With a user dictionary
cat data/dict/yonden_test_mecab_unidic.csv
python local/scripts/mecab.py --csv2dic data/dict/yonden_test_mecab_unidic.csv
python local/scripts/mecab.py --has_uttid --user_dict data/dict/yonden_test_mecab_unidic.dic --input data/text/openjtalk_text_hard.scp

#!/bin/bash
set -x # -x 'print commands',

mkdir -p exp/output
# echo "without user dict..."
# cat data/text/openjtalk_text_problem.scp
# cat data/text/openjtalk_text_problem.scp | python local/scripts/openjtalk.py --dict_dir /project/nakamura-lab08/Work/bin-wu/share/data/openjtalk_dict/openjtalk_default_unidic_ipadic --has_uttid |& tee exp/output/openjtalk_text_problem_openjtalk.txt
# cat data/text/openjtalk_text_problem.scp | python local/scripts/mecab.py --dict_dir  /project/nakamura-lab08/Work/bin-wu/share/data/mecab_dict/mecab-ipadic-2.7.0-20070801 --has_uttid |& tee exp/output/openjtalk_text_problem_mecab_ipadic.txt
# cat data/text/openjtalk_text_problem.scp | python local/scripts/mecab.py --dict_dir /project/nakamura-lab08/Work/bin-wu/share/data/mecab_dict/unidic-csj-3.1.0-full --has_uttid |& tee exp/output/openjtalk_text_problem_mecab_unidic_csj3.txt

echo "create user dict for mecab..."
cp -r /localwork/asrwork/yonden/data/wavdata/tool/data/yonden_ver3_openjtalk.{csv,dic} data/dict/
python local/convert_dict.py --input data/dict/yonden_ver3_openjtalk.csv > data/dict/yonden_ver3_mecab_unidic_csj3.csv
python local/scripts/mecab.py --csv2dic data/dict/yonden_ver3_mecab_unidic_csj3.csv # Get data/dict/yonden_ver3_mecab_unidic_csj3.dic

echo "normalize the openjtalk text with the mecab text with dict..."
cat data/text/openjtalk_text_problem.scp | python local/scripts/openjtalk.py --dict_dir /project/nakamura-lab08/Work/bin-wu/share/data/openjtalk_dict/openjtalk_default_unidic_ipadic --pos --has_uttid --user_dict data/dict/yonden_ver3_openjtalk.dic 2>/dev/null | tee exp/output/text.openjtalk
cat data/text/openjtalk_text_problem.scp | python local/scripts/mecab.py --dict_dir /project/nakamura-lab08/Work/bin-wu/share/data/mecab_dict/unidic-csj-3.1.0-full --pos --has_uttid --user_dict data/dict/yonden_ver3_mecab_unidic_csj3.dic |& tee exp/output/text.mecab
# python local/normalize_openjtalk_with_mecab.py --openjtalk_text exp/output/text.openjtalk --mecab_text exp/output/text.mecab --has_uttid --verbose |& tee exp/output/text.openjtalk.mecab
python local/normalize_openjtalk_with_mecab_chouon_local.py --openjtalk_text exp/output/text.openjtalk --mecab_text exp/output/text.mecab --has_uttid --verbose

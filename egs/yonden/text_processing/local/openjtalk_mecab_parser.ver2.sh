#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail

stage=1

# echo "create user dict for mecab from the openjtalk user dict..."
# mkdir -p conf/dict
# cp -r /localwork/asrwork/yonden/data/wavdata/tool/data/yonden_ver3_openjtalk.{csv,dic} conf/dict
# # python local/scripts/openjtalk.py --csv2dic conf/dict/yonden_ver3_openjtalk.csv # Get conf/dict/yonden_ver3_openjtalk.dic 
# python local/convert_dict.py --input conf/dict/yonden_ver3_openjtalk.csv > conf/dict/yonden_ver3_mecab_unidic_csj3.csv
# python local/scripts/mecab.py --csv2dic conf/dict/yonden_ver3_mecab_unidic_csj3.csv # Get conf/dict/yonden_ver3_mecab_unidic_csj3.dic

openjtalk_dict=/project/nakamura-lab08/Work/bin-wu/share/data/openjtalk_dict/openjtalk_default_unidic_ipadic
mecab_dict=/project/nakamura-lab08/Work/bin-wu/share/data/mecab_dict/unidic-csj-3.1.0-full
openjtalk_user_dict=/project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/csj/s0/conf/dict/yonden_ver3_openjtalk.dic
mecab_user_dict=/project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/csj/s0/conf/dict/yonden_ver3_mecab_unidic_csj3.dic
text= # text with line format uttid text
dir= # output dir

openjtalk_text_name=text.openjtalk
mecab_text_name=text.mecab
openjtalk_normalized_with_mecab_name=text.openjtalk.mecab
# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
KALDI_ROOT=/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi
. $KALDI_ROOT/egs/wsj/s5/utils/parse_options.sh || exit 1

if [ -z $dir ] | [ -z $text ]; then
    echo "./local/yonden_data_prep_all.sh --text \$path_text --dir \$output_dir [ --openjtalk_dict \$openjtalk_dict --mecab_dict \$mecab_dict --openjtalk_user_dict \$openjtalk_user_dict --mecab_user_dict \$mecab_user_dict ]"
    echo '    e.g., ./local/openjtalk_mecab_parser.sh --text exp/del/text --dir exp/del --mecab_user_dict "" --openjtalk_user_dict ""'
    exit 1
fi

if [ ${stage} -le 1 ]; then
    date
    echo "Generating openjtalk transcription..."
    cat $text | python local/scripts/openjtalk.py --dict_dir $openjtalk_dict --user_dict "$openjtalk_user_dict" --has_uttid --pos > $dir/$openjtalk_text_name

    date
    echo "Generating mecab transcription..."
    cat $text | python local/scripts/mecab.py --dict_dir $mecab_dict --user_dict "$mecab_user_dict" --has_uttid --pos > $dir/$mecab_text_name

    date
    echo "Normalize the openjtalk text with the mecab text with dict..."
    # python local/normalize_openjtalk_with_mecab.py --openjtalk_text $dir/$openjtalk_text_name --mecab_text $dir/$mecab_text_name --has_uttid --ignore_token_start_with_sokuon --verbose > $dir/$openjtalk_normalized_with_mecab_name
    python local/scripts/normalize_openjtalk_with_mecab_chouon_local.py --openjtalk_text $dir/$openjtalk_text_name --mecab_text $dir/$mecab_text_name --has_uttid --verbose > $dir/$openjtalk_normalized_with_mecab_name
    date
fi

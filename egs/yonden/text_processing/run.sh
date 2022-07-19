#!/bin/bash

# Please set custom kaldi root.
KALDI_ROOT=/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi # CHECKME
ln -sf $KALDI_ROOT/egs/wsj/s5/steps/ steps
ln -sf $KALDI_ROOT/egs/wsj/s5/utils/ utils # for parsing shell argument vectors

# general configuration
stage=1  # start from 0 if you need to start from data preparation

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

# Create user dictionary for mecab and openjtalk
mkdir -p data/dict
udict=yonden_test # user dict 
cat conf/user_dict/$udict.pronun | python local/pronun2csv.py --mecab > data/dict/${udict}_mecab.csv
cat conf/user_dict/$udict.pronun | python local/pronun2csv.py --mecab_unidic > data/dict/${udict}_mecab_unidic.csv
cat conf/user_dict/$udict.pronun | python local/pronun2csv.py --openjtalk > data/dict/${udict}_openjtalk.csv
udict=simple_num
cat conf/user_dict/$udict.pronun | python local/pronun2csv.py --mecab > data/dict/${udict}_mecab.csv
cat conf/user_dict/$udict.pronun | python local/pronun2csv.py --mecab_unidic > data/dict/${udict}_mecab_unidic.csv
cat conf/user_dict/$udict.pronun | python local/pronun2csv.py --openjtalk > data/dict/${udict}_openjtalk.csv
udict=yonden_ver1
#cat conf/user_dict/{yonden_text_am,yonden_text_openjtalk}.pronun > conf/user_dict/yonden_ver1.pronun
cat conf/user_dict/{yonden_text_am,simple_num,yonden_text_openjtalk}.pronun > conf/user_dict/yonden_ver1.pronun
#cat conf/user_dict/{yonden_text_am,simple_num,yonden_text_openjtalk,yonden_text_unidic3.1}.pronun > conf/user_dict/yonden_ver1.pronun
cat conf/user_dict/$udict.pronun | python local/pronun2csv.py --mecab > data/dict/${udict}_mecab.csv
cat conf/user_dict/$udict.pronun | python local/pronun2csv.py --mecab_unidic > data/dict/${udict}_mecab_unidic.csv
cat conf/user_dict/$udict.pronun | python local/pronun2csv.py --openjtalk > data/dict/${udict}_openjtalk.csv

# yonden_data=/localwork/asrwork/yonden/wavdata
yonden_data=/project/nakamura-lab08/Work/bin-wu/workspace/datasets/yonden/20220627/backup/
if [ ${stage} -le 1 ]; then
    # Datasets are numbered by 四電データNAIST管理表.20220627.xlsx
    # Fixed text
    text3=$yonden_data/収録用ファイル整備/210909_0808_無線機.kana.txt
    text4=$yonden_data/収録用ファイル整備/210909_1249_平岡班_無線機.kana.txt
    text5=$yonden_data/収録用ファイル整備/210914_1029_田中班_無線機.kana.txt
    text6=$yonden_data/収録用ファイル整備/210920_1358_平岡班_無線機.kana.txt
    text7=$yonden_data/収録用ファイル整備/210924_1114_田中班_無線機.kana.txt

    # Chasen text (text.am)
    ctext3=$yonden_data/kaldi_by_date/210909_0808/text.am
    ctext4=$yonden_data/kaldi_by_date/210909_1249/text.am
    ctext5=$yonden_data/kaldi_by_date/210914_1029/text.am
    ctext6=$yonden_data/kaldi_by_date/210920_1358/text.am
    ctext7=$yonden_data/kaldi_by_date/210924_1114/text.am

    # Extract text
    mkdir -p exp/yonden
    cat $text3 $text4 $text5 $text6 $text7 > exp/yonden/text3to7.txt
    cat exp/yonden/text3to7.txt | python local/scripts/chasen_text2subtokens.py --fields 1 --has_uttid | awk '{uttid = $1; $1 = ""; printf("%s %s\n", uttid, gensub(" ", "", "g", $0))}' > exp/yonden/text

    # Extract kana
    cat exp/yonden/text3to7.txt | python local/scripts/chasen_text2subtokens.py --fields 2 --has_uttid | awk '{uttid = $1; $1 = ""; printf("%s %s\n", uttid, gensub(" ", "", "g", $0))}' | perl local/scripts/kanaseq_splitter.pl -k > exp/yonden/text.kana
    # $ grep 210914_1029_田中班_無線機_00100_0014888_0015062  /localwork/asrwork/yonden/wavdata/kaldi_by_date/210914_1029/text.am
    # 210914_1029_田中班_無線機_00100_0014888_0015062 低圧|テーアツ 防護|ボーゴ し|シ ます|マス 。|。保護|ホゴ 具|グ はきます|ハキマス 。|。  # no space " between 。and 保護    
    cat $ctext3 $ctext4 $ctext5 $ctext6 $ctext7 | sed "s/。保護/。 保護/g" | \
	python local/scripts/chasen_select_first_pronun.py | python local/scripts/chasen_text2subtokens.py --fields 2 --has_uttid | \
	awk '{uttid = $1; $1 = ""; printf("%s %s\n", uttid, gensub(" ", "", "g", $0))}' | \
	perl local/scripts/kanaseq_splitter.pl -k > exp/yonden/text_chasen.kana
fi

if [ ${stage} -le 2 ]; then
    # Run simple test
    ./local/sandbox/test_simple.sh
fi

if [ ${stage} -le 3 ]; then
    echo "Computing kana error rate (KER) for text.am..."
    ref=exp/yonden/text.kana
    hyp=exp/yonden/text_chasen.kana
    COMPUTE_WER=/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi/src/bin/compute-wer
    $COMPUTE_WER --mode=present ark,t:$ref ark,t:$hyp |& tee exp/yonden/KER_$(basename $ref)_$(basename $hyp)
    ./local/sclite_score.sh  --tag KER_$(basename $ref)_$(basename $hyp) --ref $ref --hyp $hyp --text exp/yonden/text
    echo
fi

mecab_dict_dir="/project/nakamura-lab08/Work/bin-wu/share/data/mecab_dict/mecab-ipadic-2.7.0-20070801 /project/nakamura-lab08/Work/bin-wu/share/data/mecab_dict/mecab-ipadic-2.7.0-20070801-neologd-20200910  /project/nakamura-lab08/Work/bin-wu/share/data/mecab_dict/unidic-csj-3.1.0-full /project/nakamura-lab08/Work/bin-wu/share/data/mecab_dict/unidic-mecab-2.1.2_src-neologd-20200910"
user_dict=     # Without any user dict.
if [ ${stage} -le 4 ]; then
    for dict_dir in $mecab_dict_dir; do
	output=exp/yonden/text_$(basename $dict_dir).kana
	echo "DICT_DIR: $dict_dir"
	echo "USER_DICT: $user_dict"
	cat exp/yonden/text | python local/scripts/mecab.py --has_uttid --dict_dir $dict_dir --user_dict "$user_dict" | \
    	    python local/scripts/chasen_text2subtokens.py --fields 2 --has_uttid | \
    	    awk '{uttid = $1; $1 = ""; printf("%s %s\n", uttid, gensub(" ", "", "g", $0))}' | \
    	    perl local/scripts/kanaseq_splitter.pl -k > $output

	echo "Computing kana error rate (KER) for $output..."
	ref=exp/yonden/text.kana
	hyp=$output
	COMPUTE_WER=/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi/src/bin/compute-wer
	$COMPUTE_WER --mode=present ark,t:$ref ark,t:$hyp |& tee exp/yonden/KER_$(basename $ref)_$(basename $hyp)
	./local/sclite_score.sh  --tag KER_$(basename $ref)_$(basename $hyp) --ref $ref --hyp $hyp --text exp/yonden/text
	echo
    done
fi

openjtalk_dict_dir="/project/nakamura-lab08/Work/bin-wu/share/data/openjtalk_dict/open_jtalk_dic_utf_8-1.11 /project/nakamura-lab08/Work/bin-wu/share/data/openjtalk_dict/openjtalk_default_unidic_ipadic /project/nakamura-lab08/Work/bin-wu/share/data/openjtalk_dict/openjtalk_ipadic_unidic_neologd"
user_dict=  # Without any user dict.
if [ ${stage} -le 5 ]; then
    for dict_dir in $openjtalk_dict_dir; do
	output=exp/yonden/text_$(basename $dict_dir).kana
	echo "DICT_DIR: $dict_dir"
	echo "USER_DICT: $user_dict"
	cat exp/yonden/text | python local/scripts/openjtalk.py --has_uttid --dict_dir $dict_dir --user_dict "$user_dict" | \
    	    python local/scripts/chasen_text2subtokens.py --fields 2 --has_uttid | \
    	    awk '{uttid = $1; $1 = ""; printf("%s %s\n", uttid, gensub(" ", "", "g", $0))}' | \
    	    perl local/scripts/kanaseq_splitter.pl -k > $output

	echo "Computing kana error rate (KER) for $output..."
	ref=exp/yonden/text.kana
	hyp=$output
	COMPUTE_WER=/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi/src/bin/compute-wer
	$COMPUTE_WER --mode=present ark,t:$ref ark,t:$hyp |& tee exp/yonden/KER_$(basename $ref)_$(basename $hyp)
	./local/sclite_score.sh  --tag KER_$(basename $ref)_$(basename $hyp) --ref $ref --hyp $hyp --text exp/yonden/text
	echo
    done
fi

mecab_dict_dir="/project/nakamura-lab08/Work/bin-wu/share/data/mecab_dict/mecab-ipadic-2.7.0-20070801 /project/nakamura-lab08/Work/bin-wu/share/data/mecab_dict/mecab-ipadic-2.7.0-20070801-neologd-20200910"
udict=simple_num
user_dict=data/dict/${udict}_mecab.dic
if [ ${stage} -le 6 ]; then
    for dict_dir in $mecab_dict_dir; do
	output=exp/yonden/text_$(basename $dict_dir).kana
	echo "DICT_DIR: $dict_dir"
	echo "USER_DICT: $user_dict"
	python local/scripts/mecab.py --csv2dic data/dict/$(basename $user_dict .dic).csv --dict_dir $dict_dir
	cat exp/yonden/text | python local/scripts/mecab.py --has_uttid --dict_dir $dict_dir --user_dict "$user_dict" | \
    	    python local/scripts/chasen_text2subtokens.py --fields 2 --has_uttid | \
    	    awk '{uttid = $1; $1 = ""; printf("%s %s\n", uttid, gensub(" ", "", "g", $0))}' | \
    	    perl local/scripts/kanaseq_splitter.pl -k > $output

	echo "Computing kana error rate (KER) for $output..."
	ref=exp/yonden/text.kana
	hyp=$output
	COMPUTE_WER=/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi/src/bin/compute-wer
	$COMPUTE_WER --mode=present ark,t:$ref ark,t:$hyp |& tee exp/yonden/KER_$(basename $ref)_$(basename $hyp)
	./local/sclite_score.sh  --tag KER_$(basename $ref)_$(basename $hyp)_$(basename $dict_dir)_$(basename $user_dict) --ref $ref --hyp $hyp --text exp/yonden/text
	echo
    done
fi

mecab_dict_dir="/project/nakamura-lab08/Work/bin-wu/share/data/mecab_dict/unidic-csj-3.1.0-full /project/nakamura-lab08/Work/bin-wu/share/data/mecab_dict/unidic-mecab-2.1.2_src-neologd-20200910"
udict=simple_num
user_dict=data/dict/${udict}_mecab_unidic.dic
if [ ${stage} -le 6 ]; then
    for dict_dir in $mecab_dict_dir; do
	output=exp/yonden/text_$(basename $dict_dir).kana
	echo "DICT_DIR: $dict_dir"
	echo "USER_DICT: $user_dict"
	python local/scripts/mecab.py --csv2dic data/dict/$(basename $user_dict .dic).csv --dict_dir $dict_dir
	cat exp/yonden/text | python local/scripts/mecab.py --has_uttid --dict_dir $dict_dir --user_dict "$user_dict" | \
    	    python local/scripts/chasen_text2subtokens.py --fields 2 --has_uttid | \
    	    awk '{uttid = $1; $1 = ""; printf("%s %s\n", uttid, gensub(" ", "", "g", $0))}' | \
    	    perl local/scripts/kanaseq_splitter.pl -k > $output

	echo "Computing kana error rate (KER) for $output..."
	ref=exp/yonden/text.kana
	hyp=$output
	COMPUTE_WER=/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi/src/bin/compute-wer
	$COMPUTE_WER --mode=present ark,t:$ref ark,t:$hyp |& tee exp/yonden/KER_$(basename $ref)_$(basename $hyp)
	./local/sclite_score.sh  --tag KER_$(basename $ref)_$(basename $hyp)_$(basename $dict_dir)_$(basename $user_dict) --ref $ref --hyp $hyp --text exp/yonden/text
	echo
    done
fi

# open_jtalk_dic_utf_8-1.11 (default) and openjtalk_default_unidic_ipadic (compiled from mecab with unidic and ipadic) are same.
# openjtalk_dict_dir="/project/nakamura-lab08/Work/bin-wu/share/data/openjtalk_dict/open_jtalk_dic_utf_8-1.11 /project/nakamura-lab08/Work/bin-wu/share/data/openjtalk_dict/openjtalk_default_unidic_ipadic /project/nakamura-lab08/Work/bin-wu/share/data/openjtalk_dict/openjtalk_ipadic_unidic_neologd"
openjtalk_dict_dir="/project/nakamura-lab08/Work/bin-wu/share/data/openjtalk_dict/openjtalk_default_unidic_ipadic /project/nakamura-lab08/Work/bin-wu/share/data/openjtalk_dict/openjtalk_ipadic_unidic_neologd"
udict=simple_num
user_dict=data/dict/${udict}_openjtalk.dic
if [ ${stage} -le 6 ]; then
    for dict_dir in $openjtalk_dict_dir; do
	output=exp/yonden/text_$(basename $dict_dir).kana
	echo "DICT_DIR: $dict_dir"
	echo "USER_DICT: $user_dict"
	python local/scripts/openjtalk.py --csv2dic data/dict/$(basename $user_dict .dic).csv --dict_dir $dict_dir
	cat exp/yonden/text | python local/scripts/openjtalk.py --has_uttid --dict_dir $dict_dir --user_dict "$user_dict" | \
    	    python local/scripts/chasen_text2subtokens.py --fields 2 --has_uttid | \
    	    awk '{uttid = $1; $1 = ""; printf("%s %s\n", uttid, gensub(" ", "", "g", $0))}' | \
    	    perl local/scripts/kanaseq_splitter.pl -k > $output

	echo "Computing kana error rate (KER) for $output..."
	ref=exp/yonden/text.kana
	hyp=$output
	COMPUTE_WER=/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi/src/bin/compute-wer
	$COMPUTE_WER --mode=present ark,t:$ref ark,t:$hyp |& tee exp/yonden/KER_$(basename $ref)_$(basename $hyp)
	./local/sclite_score.sh  --tag KER_$(basename $ref)_$(basename $hyp)_$(basename $dict_dir)_$(basename $user_dict) --ref $ref --hyp $hyp --text exp/yonden/text
	echo
    done
fi

# Add user dict
mecab_dict_dir="/project/nakamura-lab08/Work/bin-wu/share/data/mecab_dict/mecab-ipadic-2.7.0-20070801 /project/nakamura-lab08/Work/bin-wu/share/data/mecab_dict/mecab-ipadic-2.7.0-20070801-neologd-20200910"
udict=yonden_ver1
user_dict=data/dict/${udict}_mecab.dic
if [ ${stage} -le 8 ]; then
    for dict_dir in $mecab_dict_dir; do
	output=exp/yonden/text_$(basename $dict_dir).kana
	echo "DICT_DIR: $dict_dir"
	echo "USER_DICT: $user_dict"
	python local/scripts/mecab.py --csv2dic data/dict/$(basename $user_dict .dic).csv --dict_dir $dict_dir
	cat exp/yonden/text | python local/scripts/mecab.py --has_uttid --dict_dir $dict_dir --user_dict "$user_dict" | \
    	    python local/scripts/chasen_text2subtokens.py --fields 2 --has_uttid | \
    	    awk '{uttid = $1; $1 = ""; printf("%s %s\n", uttid, gensub(" ", "", "g", $0))}' | \
    	    perl local/scripts/kanaseq_splitter.pl -k > $output

	echo "Computing kana error rate (KER) for $output..."
	ref=exp/yonden/text.kana
	hyp=$output
	COMPUTE_WER=/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi/src/bin/compute-wer
	$COMPUTE_WER --mode=present ark,t:$ref ark,t:$hyp |& tee exp/yonden/KER_$(basename $ref)_$(basename $hyp)
	./local/sclite_score.sh  --tag KER_$(basename $ref)_$(basename $hyp)_$(basename $dict_dir)_$(basename $user_dict) --ref $ref --hyp $hyp --text exp/yonden/text
	echo
    done
fi

mecab_dict_dir="/project/nakamura-lab08/Work/bin-wu/share/data/mecab_dict/unidic-csj-3.1.0-full /project/nakamura-lab08/Work/bin-wu/share/data/mecab_dict/unidic-mecab-2.1.2_src-neologd-20200910"
udict=yonden_ver1
user_dict=data/dict/${udict}_mecab_unidic.dic
if [ ${stage} -le 8 ]; then
    for dict_dir in $mecab_dict_dir; do
	output=exp/yonden/text_$(basename $dict_dir).kana
	echo "DICT_DIR: $dict_dir"
	echo "USER_DICT: $user_dict"
	python local/scripts/mecab.py --csv2dic data/dict/$(basename $user_dict .dic).csv --dict_dir $dict_dir
	cat exp/yonden/text | python local/scripts/mecab.py --has_uttid --dict_dir $dict_dir --user_dict "$user_dict" | \
    	    python local/scripts/chasen_text2subtokens.py --fields 2 --has_uttid | \
    	    awk '{uttid = $1; $1 = ""; printf("%s %s\n", uttid, gensub(" ", "", "g", $0))}' | \
    	    perl local/scripts/kanaseq_splitter.pl -k > $output

	echo "Computing kana error rate (KER) for $output..."
	ref=exp/yonden/text.kana
	hyp=$output
	COMPUTE_WER=/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi/src/bin/compute-wer
	$COMPUTE_WER --mode=present ark,t:$ref ark,t:$hyp |& tee exp/yonden/KER_$(basename $ref)_$(basename $hyp)
	./local/sclite_score.sh --tag KER_$(basename $ref)_$(basename $hyp)_$(basename $dict_dir)_$(basename $user_dict) --ref $ref --hyp $hyp --text exp/yonden/text
	echo
    done
fi

openjtalk_dict_dir="/project/nakamura-lab08/Work/bin-wu/share/data/openjtalk_dict/openjtalk_default_unidic_ipadic /project/nakamura-lab08/Work/bin-wu/share/data/openjtalk_dict/openjtalk_ipadic_unidic_neologd"
udict=yonden_ver1
user_dict=data/dict/${udict}_openjtalk.dic
if [ ${stage} -le 8 ]; then
    for dict_dir in $openjtalk_dict_dir; do
	output=exp/yonden/text_$(basename $dict_dir).kana
	echo "DICT_DIR: $dict_dir"
	echo "USER_DICT: $user_dict"
	python local/scripts/openjtalk.py --csv2dic data/dict/$(basename $user_dict .dic).csv --dict_dir $dict_dir
	cat exp/yonden/text | python local/scripts/openjtalk.py --has_uttid --dict_dir $dict_dir --user_dict "$user_dict" | \
    	    python local/scripts/chasen_text2subtokens.py --fields 2 --has_uttid | \
    	    awk '{uttid = $1; $1 = ""; printf("%s %s\n", uttid, gensub(" ", "", "g", $0))}' | \
    	    perl local/scripts/kanaseq_splitter.pl -k > $output

	echo "Computing kana error rate (KER) for $output..."
	ref=exp/yonden/text.kana
	hyp=$output
	COMPUTE_WER=/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi/src/bin/compute-wer
	$COMPUTE_WER --mode=present ark,t:$ref ark,t:$hyp |& tee exp/yonden/KER_$(basename $ref)_$(basename $hyp)
	./local/sclite_score.sh  --tag KER_$(basename $ref)_$(basename $hyp)_$(basename $dict_dir)_$(basename $user_dict) --ref $ref --hyp $hyp --text exp/yonden/text
	echo
    done
fi

openjtalk_dict_dir="/project/nakamura-lab08/Work/bin-wu/share/data/openjtalk_dict/openjtalk_default_unidic_ipadic /project/nakamura-lab08/Work/bin-wu/share/data/openjtalk_dict/openjtalk_ipadic_unidic_neologd"
udict=yonden_ver1
user_dict=data/dict/${udict}_openjtalk.dic
if [ ${stage} -le 8 ]; then
    for dict_dir in $openjtalk_dict_dir; do
	output=exp/yonden/text_$(basename $dict_dir).kana
	echo "DICT_DIR: $dict_dir"
	echo "USER_DICT: $user_dict"
	python local/scripts/openjtalk.py --csv2dic data/dict/$(basename $user_dict .dic).csv --dict_dir $dict_dir
	cat exp/yonden/text |  python local/scripts/replace_str.py --rep_in conf/char2num | \
	    python local/scripts/openjtalk.py --has_uttid --dict_dir $dict_dir --user_dict "$user_dict" | \
    	    python local/scripts/chasen_text2subtokens.py --fields 2 --has_uttid | \
    	    awk '{uttid = $1; $1 = ""; printf("%s %s\n", uttid, gensub(" ", "", "g", $0))}' | \
    	    perl local/scripts/kanaseq_splitter.pl -k > $output

	echo "Computing kana error rate (KER) for $output..."
	ref=exp/yonden/text.kana
	hyp=$output
	COMPUTE_WER=/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi/src/bin/compute-wer
	$COMPUTE_WER --mode=present ark,t:$ref ark,t:$hyp |& tee exp/yonden/KER_$(basename $ref)_$(basename $hyp)
	./local/sclite_score.sh  --tag KER_$(basename $ref)_$(basename $hyp)_$(basename $dict_dir)_$(basename $user_dict)_num --ref $ref --hyp $hyp --text exp/yonden/text
	echo
    done
fi

if [ ${stage} -le 9 ]; then
    udict=yonden_ver1.1 # user dict
    output=exp/yonden/text_openjtalk_default_dict.kana # kana text generated from openjtalk default dict: open_jtalk_dic_utf_8-1.11 (same as openjtalk_default_unidic_ipadic)

    user_dict_csv=data/dict/${udict}_openjtalk.csv
    user_dict_dic=data/dict/${udict}_openjtalk.dic

    echo "DICT_DIR: $dict_dir"
    echo "USER_DICT: $user_dict_csv"
    cat conf/user_dict/{yonden_text_am,yonden_text_openjtalk}.pronun > conf/user_dict/${udict}.pronun # simple_num dict not needed
    cat conf/user_dict/${udict}.pronun | python local/pronun2csv.py --openjtalk > $user_dict_csv # pronun2csv
    python local/scripts/openjtalk.py --csv2dic $user_dict_csv # csv2dic

    cat exp/yonden/text |  python local/scripts/replace_str.py --rep_in conf/char2num | \
	python local/scripts/replace_str.py --rep_in conf/text2fixed | \
	python local/scripts/openjtalk.py --has_uttid --user_dict "$user_dict_dic" | \
    	python local/scripts/chasen_text2subtokens.py --fields 2 --has_uttid | \
    	awk '{uttid = $1; $1 = ""; printf("%s %s\n", uttid, gensub(" ", "", "g", $0))}' | \
    	perl local/scripts/kanaseq_splitter.pl -k > $output

    echo "Computing kana error rate (KER) for $output..."
    ref=exp/yonden/text.kana
    hyp=$output
    COMPUTE_WER=/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi/src/bin/compute-wer
    $COMPUTE_WER --mode=present ark,t:$ref ark,t:$hyp |& tee exp/yonden/KER_$(basename $ref)_$(basename $hyp)
    ./local/sclite_score.sh  --tag KER_$(basename $ref)_$(basename $hyp)_openjtalk_default_dict_${udict}_user_dict --ref $ref --hyp $hyp --text exp/yonden/text
fi

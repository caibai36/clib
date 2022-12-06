#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail

stage=1
csj_data=
datasets=

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

if [ -z "$datasets" ] || [ -z "$csj_data" ]; then
    echo "./local/csj_data_prep_all.sh --csj_data \"\$csj_data\" --datasets \"\$datasets\" --stage 1"
    exit 1
fi

if [ ${stage} -le 1 ]; then
    echo "Copying the kaldi data files..."
    for dataset in $datasets; do
	mkdir -p data/$dataset
	awk 'NF==4 {print $1, $3}' $csj_data/$dataset/wav.scp > data/$dataset/wav.scp # "uttid cat path |" => "uttid path" 
	cp -r $csj_data/$dataset/{segments,spk2utt,text,utt2spk} data/$dataset
    done
fi

if [ ${stage} -le 2 ]; then
    for dataset in $datasets; do
	echo "Generating the openjtalk transcriptions..."
	mkdir -p data/$dataset/local
	cat data/$dataset/text | python local/scripts/chasen_text2subtokens.py --fields 1 --has_uttid --in_sep=+ | \
	    awk '{uttid=$1;$1="";print(uttid, gensub(" ", "", "g", $0))}'| sed 's/<sp>/。/g' | \
	    python local/scripts/replace_str.py --rep_in conf/csj_chars_rep.txt > data/$dataset/local/text.scp
	# Generated $dir/local/text.openjtalk.mecab that replaces the openjtalk phrases that have chouon problems with mecab ones that do not have chouon problem.
	./local/openjtalk_mecab_parser.ver2.sh --text data/$dataset/local/text.scp --dir data/$dataset/local
    done
fi

if [ ${stage} -le 3 ]; then
    for dataset in $datasets; do
	echo "Generating the kana and phoneme representation..."
	mkdir -p data/$dataset/local
	data_dir=data/$dataset
	sed -r -f conf/openjtalk_pos_punctuation_norm.sed data/$dataset/local/text.openjtalk.mecab > $data_dir/text.am.pos # normalize every utterance with period ending 1) =>. 2) ,.=>. 3) ,=>.
	
	cat $data_dir/text.am.pos | python local/scripts/chasen_text2subtokens.py --fields 1 2 --has_uttid | sed 's/[ \t]*$//' > $data_dir/text.am
	cat $data_dir/text.am.pos | python local/scripts/chasen_text2subtokens.py --fields 1 --has_uttid | sed 's/[ \t]*$//' > $data_dir/text.char # Removing tail spaces: 'sed 's/[ \t]*$//'
	cat $data_dir/text.am.pos | python local/scripts/chasen_text2subtokens.py --fields 2 --has_uttid | \
    	    perl local/scripts/kanaseq_splitter.pl -k | sed 's/[ \t]*$//' > $data_dir/text.kana
	cat $data_dir/text.am.pos | python local/scripts/chasen_text2subtokens.py --fields 2 --has_uttid | \
    	    perl local/scripts/kanaseq2phoneseq.pl -k | sed 's/[ \t]*$//' > $data_dir/text.phone
    done
fi

if [ ${stage} -le 4 ]; then
    for dataset in $datasets; do
	data_dir=data/$dataset
	cp $data_dir/text $data_dir/text.csj.pos
	cp $data_dir/text.char $data_dir/text
    done
fi

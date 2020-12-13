# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -uo pipefail

corpus=/project/nakamura-lab08/Work/bin-wu/share/data/mboshi/full_corpus_newsplit
stage=8
# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

if [ -z $corpus -o ! -d $corpus ]; then
    echo "$corpus not found"
    exit
fi

if [ $stage -le 1 ]; then
    mkdir -p data/{train_raw_mb,dev_raw_mb,train_mb,dev_mb,test_mb}
    # mkdir -p data/{train_raw_fr,dev_raw_fr,train_fr,dev_fr,test_fr}
    for file in  $(find $corpus/train/ -name *wav);do
	echo $(basename $file .wav) $file;
    done | sort -u > data/train_raw_mb/wav.scp

    for file in  $(find $corpus/dev/ -name *wav);do
	echo $(basename $file .wav) $file;
    done | sort -u > data/dev_raw_mb/wav.scp
fi

if [ $stage -le 2 ]; then
    # split train_raw into train and dev
    # (base) [bin-wu@ahccsclm03 s0]$ wc -l data/*_mb/*
    # 200 data/dev_mb/wav.scp
    # 514 data/dev_raw_mb/wav.scp
    # 514 data/test_mb/wav.scp
    # 4416 data/train_mb/wav.scp
    # 4616 data/train_raw_mb/wav.scp

    grep abiayi data/train_raw_mb/wav.scp | head -n 144 > data/dev_mb/wav.scp
    grep -v -f data/dev_mb/wav.scp data/train_raw_mb/wav.scp | grep abiayi > data/train_mb/wav.scp

    grep kouarata data/train_raw_mb/wav.scp | head -n 48 >> data/dev_mb/wav.scp
    grep -v -f data/dev_mb/wav.scp data/train_raw_mb/wav.scp | grep kouarata >> data/train_mb/wav.scp 

    grep martial data/train_raw_mb/wav.scp | head -n 8 >> data/dev_mb/wav.scp
    grep -v -f data/dev_mb/wav.scp data/train_raw_mb/wav.scp | grep martial >> data/train_mb/wav.scp 

    cp -r data/dev_raw_mb/wav.scp data/test_mb/
fi

if [ $stage -le 3 ]; then
    # cp -r data/train_raw_mb data/train_raw_fr
    for file in data/{train_raw_mb,dev_raw_mb,train_mb,dev_mb,test_mb}; do cp -r $file ${file/mb/fr}; done
fi

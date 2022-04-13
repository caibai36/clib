#!/bin/bash
# see the setting at https://arxiv.org/abs/1809.01431 'Pre-training on high-resource speech recognition improves low-resource speech-to-text translation'
# "Since this corpus does not include a designated test set, we randomly sampled and removed 200 utterances from training to use as a development set, and use the designated development data as a test set."
#
# Stats on new dev/train split avoiding overlap
# dev: source: {'dico': 350, 'part': 164} speakers: {'abiayi': 351, 'kouarata': 126, 'martial': 37}
# train: source: {'dico': 3313, 'part': 1303} speakers: {'abiayi': 3330, 'kouarata': 1108, 'martial': 178}
#
# roughly the ratio of three speakers in trainning set 18:6:1
# We sample 144+48+8 sentences for the three speakers,  remove these 200 utterances from the training set as the development set.
# train_raw - dev = train
# all    4616 - 200 = 4416
# abiayi 3330 -144 = 3186
# kouarata 1108 -48 = 1060
# martial 178 - 8 = 170

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail

stage=1
dataset=dev_mb
corpus=/project/nakamura-lab08/Work/bin-wu/share/data/mboshi/full_corpus_newsplit
dir=train
vad_file=data/segments/align_rmsil_vad.txt
lang=mb
. utils/parse_options.sh || exit 1 # eg. ./run.sh --stage 1

if [ -z $dataset ] || [ -z $corpus ] || [ -z $vad_file ]; then
    echo "./local/mboshi_data_prepare.sh --dataset \$dataset --corpus \$corpus --dir \$dir --vad_file \$vad_file --lang \$lang"
    exit 1
fi

if [ $stage -le 1 ]; then
    echo "Data preparation for dataset ${dataset}..."
    mkdir -p data/${dataset}

    echo "Preparing utt2spk and spk2utt..."
    awk '{print gensub(/^([A-Za-z0-9]+)_([A-Za-z0-9_-]+)$/, "\\0 \\1", "g", $1)}' data/$dataset/wav.scp > data/${dataset}/utt2spk # wav.scp: S002_01 test/S002_01.wav
    utils/utt2spk_to_spk2utt.pl data/${dataset}/utt2spk > data/${dataset}/spk2utt

    echo "Preparing segment..."
    for uttid in $(awk '{print $1}' data/$dataset/wav.scp); do
	awk -v uttid=$uttid '$1==uttid {print $1, $1, $2, $3}' $vad_file
    done | sort -k1,1 -u > data/${dataset}/segments # vad_file: S002_01 1.0800 3.4200
    echo "number of segments: $(wc -l data/$dataset/segments)"

    echo "Preparing text..."
    # abiayi_2015-09-08-11-33-57_samsung-SM-T530_mdw_elicit_Dico18_101.fr.cleaned.noPunct
    if [ $lang = fr ]; then
	postfix=".fr.cleaned.noPunct"
    elif [ $lang = mb ]; then
	postfix=".mb.cleaned"
    else
	echo "$lang should be 'fr' or 'mb', but you are using $lang"
	exit
    fi

    for uttid in $(awk '{print $1}' data/$dataset/wav.scp); do
	echo $uttid $(cat ${corpus}/${dir}/${uttid}${postfix})
    done | sort -k1,1 -u > data/${dataset}/text
    echo "number of text: $(wc -l data/$dataset/text)"
fi

#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail

stage=1
dataset=
corpus=/localwork/asrwork/yonden/wavdata
segmented_recordings=/project/nakamura-lab08/Work/bin-wu/workspace/datasets/yonden/segmented_recordings # directory that outputs segmented recordings according to wav.scp and segments
sampling_rate=16000
dir=data/local/all
data_index="{003..028}*"

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

if [ -z $corpus ] || [ -z $segmented_recordings ]; then
    echo "./local/yonden_data_prep_all.sh --corpus \$corpus --segmented_recordings \$segmented_recordings --sampling_rate \$sampling_rate (default 16000) --dir data/local/all"
    exit 1
fi

# Datasets are numbered by 四電データNAIST管理表.20220906.xlsx
data_dirs=$corpus/kaldi_by_num_date/$data_index
echo "Data preparation for dataset ${dataset}..."

mkdir -p $dir
if [ $stage -le 1 ]; then # use '-1' to skip; use '1' to start.
    echo "Segmenting long recordings from wav.scp and long.scp..."
    date

    # Append all wav.scp and segments files
    # Each line from `$uttid cat $path |' to `$uttid $path'
    eval cat $data_dirs/wav.scp | awk 'NF==4 {print $1, $3}' > $dir/wav.scp # eval for expansion of glob of shell variables
    eval cat $data_dirs/segments > $dir/segments

    mkdir -p $segmented_recordings
    echo $segmented_recordings

    # Segmentation
    ./local/scripts/rec2seg_sox.sh --output_wav_scp $segmented_recordings/segmented_recordings_raw.scp \
				   $dir/wav.scp $dir/segments $segmented_recordings/segmented_recordings_raw

    # Segmentation normalized by the maximum amplitude of each segment (utterance)
    python local/scripts/rec2seg.py --wav_scp $dir/wav.scp \
	   --segments $dir/segments \
	   --output_dir $segmented_recordings/segmented_recordings_normalized \
	   --output_wav_scp $segmented_recordings/segmented_recordings_normalized.scp \
	   --sampling_rate $sampling_rate
    date
fi

if [ $stage -le 2 ]; then    
    echo "Preparing wav.scp..."
    eval cat $data_dirs/wav.scp | awk 'NF==4 {print $1, $3}' > $dir/wav.scp # eval for expansion of glob of shell variables

    echo "Preparing segments..."
    eval cat $data_dirs/segments > $dir/segments
    # additional information
    cp $segmented_recordings/segmented_recordings_normalized.scp $dir/seg_wav_amp_norm.scp
    cp $segmented_recordings/segmented_recordings_raw.scp $dir/seg_wav.scp

    echo "Preparing utt2spk and spk2utt..."

    echo "Preparing text..."
    eval cat $data_dirs/text.addinfo | python local/scripts/replace_str.py --rep_in conf/addinfo2fixed > $dir/text.addinfo
    cat $dir/text.addinfo | awk -v FS='\t' '{print $1, $2}' | python local/scripts/chasen_text2subtokens.py --fields 1 2 3 4 --has_uttid | sed 's/[ \t]*$//' > $dir/text.am.pos
    sed -r -f conf/openjtalk_pos_punctuation_norm.sed -i $dir/text.am.pos # normalize every utterance with period ending 1) =>. 2) ,.=>. 3) ,=>.
    paste -d '\t' <(awk '{print $1}' $dir/text.am.pos) <(cut -d' ' -f2- $dir/text.am.pos) <(cat $dir/text.addinfo | awk -v FS='\t' '{print $3}') > $dir/text.addinfo.bak # fix text.addinfo with text.am.pos
    mv $dir/text.addinfo.bak $dir/text.addinfo

    cat $dir/text.am.pos | python local/scripts/chasen_text2subtokens.py --fields 1 2 --has_uttid | sed 's/[ \t]*$//' > $dir/text.am
    cat $dir/text.am.pos | python local/scripts/chasen_text2subtokens.py --fields 1 --has_uttid | sed 's/[ \t]*$//' > $dir/text # Removing tail spaces: 'sed 's/[ \t]*$//'
    eval cat $data_dirs/text.am.chasen > $dir/text.am.chasen
    cp $dir/text $dir/text.eval
    # additional information
    cp $dir/text $dir/text.char
    cat $dir/text.am.pos | python local/scripts/chasen_text2subtokens.py --fields 2 --has_uttid | \
    	perl local/scripts/kanaseq_splitter.pl -k | sed 's/[ \t]*$//' > $dir/text.kana
    cat $dir/text.am.pos | python local/scripts/chasen_text2subtokens.py --fields 2 --has_uttid | \
    	perl local/scripts/kanaseq2phoneseq.pl -k | sed 's/[ \t]*$//' > $dir/text.phone
fi

if [ $stage -le 3 ]; then
    echo "Creating csv and json summary files..."
    python local/data2info.py --data $dir \
	   --scps kanji:$dir/text.char kana:$dir/text.kana phone:$dir/text.phone seg_wav:$dir/seg_wav.scp seg_wav_amp_norm:$dir/seg_wav_amp_norm.scp problem:conf/problematic_utterances.txt \
	   --outjson $dir/info.json \
	   --outcsv $dir/info.csv \
    	   --time2data conf/time2data # optional from dataset 3 to dataset 26

fi

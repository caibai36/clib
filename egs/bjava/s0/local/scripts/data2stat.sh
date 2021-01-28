#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -uo pipefail

# dir contains datasets
data=data
# dataset name
train=train
dev=dev
test=test
verbose=false

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1
echo "We assume that each line of data/dataset/text follows kaldi format as <uttid> <text> format"
echo "usage: ./local/scripts/data2stat.sh --data data --train train_si284 --dev test_dev93 --test test_eval92"
echo "usage: ./local/scripts/data2stat.sh --data ../../wsj/s5/data --train train_si284 --dev test_dev93 --test test_eval92"
echo "usage: ./local/scripts/data2stat.sh --data ../../csj/s5/data  --train train_nodev --dev train_dev --test eval1"
echo "usage: ./local/scripts/data2stat.sh --data ../../timit/s0/data --verbose true"
echo "usage: ./local/scripts/data2stat.sh --data ../../bjava/s0/data"
echo

for dataset in $train $dev $test; do
    ./local/scripts/data2dur.sh $data/$dataset | grep "Duration"
done; echo

./local/scripts/data_text_overlapped.sh --data $data --train $train --dev $dev --test $test --verbose false; echo

echo -e "data\t#speakers"
for dataset in $train $dev $test; do
    echo $data/$dataset $(cat $data/$dataset/spk2utt | wc -l)
done; echo

./local/scripts/data_speaker_overlapped.sh --data $data --train $train --dev $dev --test $test --verbose false; echo

if [ -f  $data/$dataset/utt2gender ]; then
    echo -e "data\t#gender"
    for dataset in $train $dev $test; do
	echo -n "$dataset "
	grep -f <(cat $data/$dataset/spk2utt | awk '{print $2}') $data/$dataset/utt2gender | \
	    cut -d' ' -f2- | awk '{ !a[$0]++ }; END { for (key in a) print key,a[key] }' | tr '\n' ' '; echo # $2 means pick the first utterance each speaker say.
    done; echo
fi

if [ -f  $data/$dataset/utt2gender ]; then
    echo -e "data\t#utts/gender"
    for dataset in $train $dev $test; do
	echo -n "$dataset "
	cat $data/$dataset/utt2gender | cut -d' ' -f2- | awk '{ !a[$0]++ }; END { for (key in a) print key,a[key] }' | tr '\n' ' '; echo
    done;echo
fi

if $verbose; then
   for dataset in $train $dev $test; do
       wc -l $data/$dataset/{utt2*,wav.scp,spk2utt,text,wav.scp}
       [ ! -f $data/$dataset/segments ] || wc -l $data/$dataset/segments
       echo
   done | sed -r 's/^[ ]+//g' | grep -v "total"
fi

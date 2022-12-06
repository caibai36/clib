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
verbose=true
# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1
if $verbose; then
   echo "We assume that each line of data/dataset/spk2utt follows kaldi format as <speaker> <utt1> <utt2>...format"
   echo "usage ./local/scripts/data_speaker_overlap.sh --data data --train train_si284 --dev test_dev93 --test test_eval92"
   echo
fi

if [ ! -f $data/$train/spk2utt ]; then echo "Error: file $data/$train/spk2utt not found..."; exit 1; fi
if [ ! -f $data/$dev/spk2utt ]; then echo "Error: file $data/$dev/spk2utt not found..."; exit 1; fi
if [ ! -f $data/$test/spk2utt ]; then echo "Error: file $data/$test/spk2utt not found..."; exit 1; fi

echo 
echo -e "data_data\t#comm_speakers"
#echo "number of commom speakers between $dev and $test dataset:"
echo -ne "$dev-$test\t"
comm <(cat $data/$dev/spk2utt | cut -d' ' -f1 | sort) <(cat $data/$test/spk2utt | cut -d' ' -f1 | sort) -12 | wc -l
echo -ne "$train-$dev\t"
comm <(cat $data/$train/spk2utt | cut -d' ' -f1 | sort) <(cat $data/$dev/spk2utt | cut -d' ' -f1 | sort) -12 | wc -l
echo -ne "$train-$test\t"
comm <(cat $data/$train/spk2utt | cut -d' ' -f1 | sort) <(cat $data/$test/spk2utt | cut -d' ' -f1 | sort) -12 | wc -l
echo -ne "$train-$dev+$test\t"
comm <(cat $data/$train/spk2utt | cut -d' ' -f1 | sort) <(cat $data/{$dev,$test}/spk2utt | cut -d' ' -f1 | sort) -12 | wc -l

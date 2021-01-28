#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -uo pipefail

# dataset name
train=train
dev=dev
test=test

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1
echo "We assume that each line of data/dataset/text follows kaldi format as <uttid> <text> format"
echo "usage ./local/scripts/data_text_overlap.sh --train train_si284 --dev test_dev93 --test test_eval92"
echo

if [ ! -f data/$train/text ]; then echo "Error: file data/$train/text not found..."; exit 1; fi
if [ ! -f data/$dev/text ]; then echo "Error: file data/$dev/text not found..."; exit 1; fi
if [ ! -f data/$test/text ]; then echo "Error: file data/$test/text not found..."; exit 1; fi

echo -e "data\t#texts\t#uniq_texts"
echo -e "$train\t$(cat data/$train/text | cut -d' ' -f2- | wc -l)\t$(cat data/$train/text | cut -d' ' -f2- | sort -u | wc -l)"
echo -e "$dev\t$(cat data/$dev/text | cut -d' ' -f2- | wc -l)\t$(cat data/$dev/text | cut -d' ' -f2- | sort -u | wc -l)"
echo -e "$test\t$(cat data/$test/text | cut -d' ' -f2- | wc -l)\t$(cat data/$test/text | cut -d' ' -f2- | sort -u | wc -l)"

echo 
echo -e "data_data\t#comm_texts"
#echo "number of commom texts between $dev and $test dataset:"
echo -ne "$dev-$test\t"
comm <(cat data/$dev/text | cut -d' ' -f2- | sort) <(cat data/$test/text | cut -d' ' -f2- | sort) -12 | wc -l
echo -ne "$train-$dev\t"
comm <(cat data/$train/text | cut -d' ' -f2- | sort) <(cat data/$dev/text | cut -d' ' -f2- | sort) -12 | wc -l
echo -ne "$train-$test\t"
comm <(cat data/$train/text | cut -d' ' -f2- | sort) <(cat data/$test/text | cut -d' ' -f2- | sort) -12 | wc -l
echo -ne "$train-$dev+$test\t"
comm <(cat data/$train/text | cut -d' ' -f2- | sort) <(cat data/{$dev,$test}/text | cut -d' ' -f2- | sort) -12 | wc -l

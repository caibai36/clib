#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -euo pipefail

stage=1
# info=data/local/all_3to28/info.json
info=
dir=data/local/division # the output directory of the datasets (e.g., 'data')

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

if [ -z $info ]; then
    echo "e.g.,  ./local/yonden_data_division.sh --stage 2 --info data/local/all_3to36/info.json --dir data/local/division"
    exit 1
fi

echo "Creating datasets from \'$info\'..."
source local/yonden_data_division_fun.sh

mkdir -p $dir

dataset="all_data3-36"
filter='df["dataset_index"] <= 36'
yonden_baseline "$dir" "$dataset" "$filter"

if [ $stage -le 2 ]; then

    echo "Creating test datasets (for pretrained model)..."

    # dataset="train_data3_5_7to17_baseline"
    # filter='(df["dataset_index"] != 4) & (df["dataset_index"] != 6) & (df["dataset_index"] <= 17)'
    # dataset="dev_data4_baseline"
    # filter='df["dataset_index"] == 4'
    # dataset="all_data3-36_baseline"
    # filter='df["dataset_index"] <= 36'
    # dataset="test_data6_baseline"
    # filter='df["dataset_index"] == 6'
    # yonden_baseline "$dir" "$dataset" "$filter"

    # data4 and data28 merged as dev set
    # data6 or data36 as test set
    filter='df["dataset_index"] == 6'
    dataset="test_data6"
    yonden_baseline "$dir" "$dataset" "$filter"
    dataset="test_data6_ampnorm"
    yonden_baseline_ampnorm "$dir" "$dataset" "$filter"
    # dataset="test_data6_spkinfo"
    # yonden_baseline_spkinfo "$dir" "$dataset" "$filter"
    # dataset="test_data6_ampnorm_spkinfo"
    # yonden_baseline_ampnorm_spkinfo "$dir" "$dataset" "$filter"
    # dataset="test_data6_spkinfo_daily"
    # yonden_baseline_spkinfo_daily "$dir" "$dataset" "$filter"
    # dataset="test_data6_ampnorm_spkinfo_daily"
    # yonden_baseline_ampnorm_spkinfo_daily "$dir" "$dataset" "$filter"

    filter='df["dataset_index"] == 36'	  
    dataset="test_data36"
    yonden_baseline "$dir" "$dataset" "$filter"
    dataset="test_data36_ampnorm"
    yonden_baseline_ampnorm "$dir" "$dataset" "$filter"
    # dataset="test_data36_spkinfo"
    # yonden_baseline_spkinfo "$dir" "$dataset" "$filter"
    # dataset="test_data36_ampnorm_spkinfo"
    # yonden_baseline_ampnorm_spkinfo "$dir" "$dataset" "$filter"
    # dataset="test_data36_spkinfo_daily"
    # yonden_baseline_spkinfo_daily "$dir" "$dataset" "$filter"
    # dataset="test_data36_ampnorm_spkinfo_daily"
    # yonden_baseline_ampnorm_spkinfo_daily "$dir" "$dataset" "$filter"
fi

if [ $stage -le 2 ]; then
    echo "Creating training and development datasets..."

    # data4 and data28 merged as dev set
    # data6 or data36 as test set
    filter='(df["dataset_index"] == 4) | (df["dataset_index"] == 28)'
    dataset="dev_data4_28"
    yonden_baseline "$dir" "$dataset" "$filter"
    dataset="dev_data4_28_ampnorm"
    yonden_baseline_ampnorm "$dir" "$dataset" "$filter"
    # dataset="dev_data4_28_spkinfo"
    # yonden_baseline_spkinfo "$dir" "$dataset" "$filter"
    # dataset="dev_data4_28_ampnorm_spkinfo"
    # yonden_baseline_ampnorm_spkinfo "$dir" "$dataset" "$filter"
    # dataset="dev_data4_28_spkinfo_daily"
    # yonden_baseline_spkinfo_daily "$dir" "$dataset" "$filter"
    # dataset="dev_data4_28_ampnorm_spkinfo_daily"
    # yonden_baseline_ampnorm_spkinfo_daily "$dir" "$dataset" "$filter"

    filter='(df["dataset_index"] != 4) & (df["dataset_index"] != 6) & (df["dataset_index"] != 28) & (df["dataset_index"] != 36) & (df["dataset_index"] <= 36)'
    dataset="train_data3-36_remove_4_6_28_36"
    yonden_baseline "$dir" "$dataset" "$filter"
    dataset="train_data3-36_remove_4_6_28_36_ampnorm"
    yonden_baseline_ampnorm "$dir" "$dataset" "$filter"
    # dataset="train_data3-36_remove_4_6_28_36_spkinfo"
    # yonden_baseline_spkinfo "$dir" "$dataset" "$filter"
    # dataset="train_data3-36_remove_4_6_28_36_ampnorm_spkinfo"
    # yonden_baseline_ampnorm_spkinfo "$dir" "$dataset" "$filter"
    # dataset="train_data3-36_remove_4_6_28_36_spkinfo_daily"
    # yonden_baseline_spkinfo_daily "$dir" "$dataset" "$filter"
    # dataset="train_data3-36_remove_4_6_28_36_ampnorm_spkinfo_daily"
    # yonden_baseline_ampnorm_spkinfo_daily "$dir" "$dataset" "$filter"
fi

# if [ $stage -le 1 ]; then # use '-1' to skip; use '1' to start.
#     echo "Dumping info.json into scp files..."
#     cat $info | python local/dump_info.py --dir $dir/info_scps

#     echo "Converting info.json to original data..."

#     # sed -r 's/(.*) "(.*)"/\1 \2/g' to remove the "". (e.g. id1 "word1 word2" => id1 word1 word2)
#     mkdir -p $dir/all_original
#     field="wav"; python -c "import pandas as pd; df = pd.read_json('$info').T; print(df.set_index('recid')['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/^(.*)\s+(.*)$/\1 cat \2 |/' > $dir/all_original/wav.scp
#     python -c "import pandas as pd; df = pd.read_json('$info').T; print(df[['recid', 'begin_sec', 'end_sec']].to_csv(sep=' ',header=None), end='')" > $dir/all_original/segments
#     field="kanji"; python -c "import pandas as pd; df = pd.read_json('$info').T; print(df['$field'].to_csv(sep=' ',header=None), end='')" |  sed -r 's/(.*) "(.*)"/\1 \2/g' > $dir/all_original/text
#     python -c "import pandas as pd; df = pd.read_json('$info').T; print(('\t' + df['text.am.pos'] + '\t' + df['speaker_label'] + '_' + df['speaker'] + '_' + df['scene']).to_csv(sep=' ',header=None), end='')" |  sed -r -e 's/(.*) "(.*)"/\1 \2/g' -e 's/\s\t/\t/g' > $dir/all_original/text.addinfo
#     for field in "text.am" "text.am.chasen" "text.am.pos" "text.eval"; do
# 	python -c "import pandas as pd; df = pd.read_json('$info').T; print(df['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' >  $dir/all_original/$field
#     done
#     speaker="recid"; python -c "import pandas as pd; df = pd.read_json('$info').T; print(df['$speaker'].to_csv(sep=' ',header=None), end='')" |  sed -r 's/(.*) "(.*)"/\1 \2/g' > $dir/all_original/utt2spk
#     speaker="recid"; python -c "import pandas as pd; df = pd.read_json('$info').T; print(df.groupby('$speaker').apply(lambda x: ' '.join(x['id'].unique())).to_csv(sep=' ',header=None), end='')" |  sed -r 's/(.*) "(.*)"/\1 \2/g' > $dir/all_original/spk2utt
# fi

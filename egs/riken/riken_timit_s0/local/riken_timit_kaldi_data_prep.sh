#!/usr/bin/env bash

# Copyright 2013   (Authors: Bagher BabaAli, Daniel Povey, Arnab Ghoshal)
#           2014   Brno University of Technology (Author: Karel Vesely)
# Apache 2.0.
# Modified script for TIMIT data preparation with phone and word labels

if [ $# -ne 1 ]; then
   echo "Argument should be the Timit directory, see ../run.sh for example."
   exit 1;
fi

timit=$1
datadir=`pwd`/data/kaldi
local=`pwd`/local
utils=`pwd`/utils
conf=`pwd`/conf
tmpdir=$(mktemp -d /tmp/kaldi.XXXX)
trap 'rm -rf "$tmpdir"' EXIT

. ./path.sh # Needed for KALDI_ROOT
sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
   exit 1;
fi

[ -f $conf/test_spk.list ] || { echo "Eval-set speaker list not found."; exit 1; }
[ -f $conf/dev_spk.list ] || { echo "Dev-set speaker list not found."; exit 1; }

# Check directory case
uppercased=false
train_dir=train
test_dir=test
if [ -d $timit/TRAIN ]; then
  uppercased=true
  train_dir=TRAIN
  test_dir=TEST
fi

# Get the list of speakers
if $uppercased; then
  tr '[:lower:]' '[:upper:]' < $conf/dev_spk.list > $tmpdir/dev_spk
  tr '[:lower:]' '[:upper:]' < $conf/test_spk.list > $tmpdir/test_spk
  ls -d "$timit"/TRAIN/DR*/* | sed -e "s:^.*/::" > $tmpdir/train_spk
else
  tr '[:upper:]' '[:lower:]' < $conf/dev_spk.list > $tmpdir/dev_spk
  tr '[:upper:]' '[:lower:]' < $conf/test_spk.list > $tmpdir/test_spk
  ls -d "$timit"/train/dr*/* | sed -e "s:^.*/::" > $tmpdir/train_spk
fi

echo "Preparing train, dev and test data"

# Create phone mapping file for duration format conversion
awk '{print $1 "\t" $3}' $conf/phones.60-48-39.map | sed 's:^q\t$:q\tsil:' > $conf/phones.61-39.map

# Compile the PHN converter
g++ $local/c/timit_normal.cpp -o $local/c/timit_normal || { echo "Failed to compile timit_normal.cpp"; exit 1; }

# Process each dataset
for x in train dev test; do
  echo "Processing $x set..."

  # Define directory variables
  data_dir=$datadir/$x
  time_dir=$datadir/${x}_time
  phn_dir=$time_dir/${x}_phn
  wrd_dir=$time_dir/${x}_wrd
  txt_dir=$time_dir/${x}_txt
  phn_label_dir=$time_dir/${x}_phn_label
  wrd_label_dir=$time_dir/${x}_wrd_label

  # Create directories
  mkdir -p $data_dir $phn_dir $wrd_dir $txt_dir $phn_label_dir $wrd_label_dir

  # Determine source directory based on set type
  if [ "$x" == "train" ]; then
    src_dir=$train_dir
  else
    src_dir=$test_dir
  fi

  # Find audio files (excluding SA* utterances for all sets as per original script)
  find $timit/$src_dir/ -name "*.WAV" -not \( -iname 'SA*' \) | grep -f $tmpdir/${x}_spk > $tmpdir/${x}_sph.flist

  # Create utterance IDs
  sed -e 's:.*/\(.*\)/\(.*\).\(WAV\|wav\)$:\1_\2:' $tmpdir/${x}_sph.flist > $tmpdir/${x}_uttids

  # Create wav.scp - SORTED
  paste $tmpdir/${x}_uttids $tmpdir/${x}_sph.flist | \
    sort -k1,1 | \
    awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' > $data_dir/wav.scp

  # Create utt2spk (extract speaker ID from utterance ID) - SORTED
  cat $tmpdir/${x}_uttids | awk -F'_' '{print $0, $1}' | sort -k1,1 > $data_dir/utt2spk

  # Create spk2utt from utt2spk
  $utils/utt2spk_to_spk2utt.pl < $data_dir/utt2spk > $data_dir/spk2utt

  # Create spk2gender (assuming first letter of speaker ID indicates gender) - SORTED
  cat $data_dir/spk2utt | awk '{print $1}' | \
    perl -ane 'chop; m:^.:; $g = lc($&); print "$_ $g\n";' | sort -k1,1 > $data_dir/spk2gender

  # Process PHN files
  echo "Processing PHN files for $x set..."
  find $timit/$src_dir/ -name "*.PHN" -not \( -iname 'SA*' \) | grep -f $tmpdir/${x}_spk > $tmpdir/${x}_phn.flist

  # Copy PHN files with speaker_utterance naming
  cat $tmpdir/${x}_phn.flist | \
    sed "s:.*/\(.*\)/\(.*\).PHN:cp \0\t$phn_dir/\1_\2.PHN:g" | sh

  # Convert PHN files to label format
  for file in $phn_dir/*; do
    $local/c/timit_normal $file $conf/phones.61-39.map > $phn_label_dir/$(basename $file)
  done

  # Process WRD files
  echo "Processing WRD files for $x set..."
  find $timit/$src_dir/ -name "*.WRD" -not \( -iname 'SA*' \) | grep -f $tmpdir/${x}_spk > $tmpdir/${x}_wrd.flist

  # Copy WRD files with speaker_utterance naming
  cat $tmpdir/${x}_wrd.flist | \
    sed "s:.*/\(.*\)/\(.*\).WRD:cp \0\t$wrd_dir/\1_\2.WRD:g" | sh

  # Convert WRD files to label format
  for file in $wrd_dir/*; do
    $local/c/timit_normal $file > $wrd_label_dir/$(basename $file)
  done

  # Process TXT files for word-level transcriptions
  echo "Processing TXT files for $x set..."
  find $timit/$src_dir/ -name "*.TXT" -not \( -iname 'SA*' \) | grep -f $tmpdir/${x}_spk > $tmpdir/${x}_txt.flist

  # Copy TXT files with speaker_utterance naming
  cat $tmpdir/${x}_txt.flist | \
    sed "s:.*/\(.*\)/\(.*\).TXT:cp \0\t$txt_dir/\1_\2.TXT:g" | sh

  # Extract word-level text for each utterance
  sed -e 's:.*/\(.*\)/\(.*\).\(TXT\|txt\)$:\1_\2:' $tmpdir/${x}_txt.flist > $tmpdir/${x}_txt.uttids

  # Create word-level text file
  while read -r uttid_file; do
    uttid=$(echo "$uttid_file" | awk '{print $1}')
    file=$(echo "$uttid_file" | awk '{print $2}')
    text=$(awk '{$1=""; $2=""; print $0}' "$file" | sed 's/^ *//')
    echo "$uttid $text"
  done < <(paste $tmpdir/${x}_txt.uttids $tmpdir/${x}_txt.flist) | sort -k1,1 > $data_dir/text.word

  # Create phone-level text file (from PHN files)
  sed -e 's:.*/\(.*\)/\(.*\).\(PHN\|phn\)$:\1_\2:' $tmpdir/${x}_phn.flist > $tmpdir/${x}_phn.uttids

  while read line; do
    [ -f $line ] || { echo "Cannot find transcription file '$line'"; exit 1; }
    cut -f3 -d' ' "$line" | tr '\n' ' ' | perl -ape 's: *$:\n:;'
  done < $tmpdir/${x}_phn.flist > $tmpdir/${x}_phn.trans

  paste $tmpdir/${x}_phn.uttids $tmpdir/${x}_phn.trans | sort -k1,1 > $tmpdir/${x}.tmp.trans
  cat $tmpdir/${x}.tmp.trans | $local/timit_norm_trans.pl -i - -m $conf/phones.60-48-39.map -to 48 | sort -k1,1 > $data_dir/text

  # Validate data directory
  $utils/validate_data_dir.sh --no-feats $data_dir || exit 1
done

echo "Data preparation completed successfully"

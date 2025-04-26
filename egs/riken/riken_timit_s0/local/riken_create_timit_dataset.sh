#!/usr/bin/env bash

# timit_dataset_prep.sh
# Script to create a dataset from TIMIT following the standard Kaldi division

if [ $# -lt 2 ]; then
   echo "Usage: $0 <timit_dir> <output_dir> [sph2pipe_path]"
   echo "  timit_dir: Path to TIMIT corpus (e.g., /data/share/bin-wu/data/human/speech/timit/TIMIT/TIMIT)"
   echo "  output_dir: Path to output directory (e.g., /data/share/bin-wu/data/human/speech/timit/processsing/timit)"
   echo "  sph2pipe_path: Path to sph2pipe executable (optional, default: /home/bin-wu/share/tools/kaldi/tools/sph2pipe_v2.5/sph2pipe)"
   exit 1
fi

timit_dir=$1
output_dir=$2
sph2pipe=${3:-/home/bin-wu/share/tools/kaldi/tools/sph2pipe_v2.5/sph2pipe}
local=`pwd`/local
utils=`pwd`/utils
conf=`pwd`/conf

# Check if sph2pipe exists
if [ ! -x "$sph2pipe" ]; then
   echo "Could not find or execute the sph2pipe program at $sph2pipe"
   exit 1
fi

# Check if output directory already contains data
if [ -d "$output_dir" ] && [ "$(ls -A "$output_dir" 2>/dev/null)" ]; then
  echo "Warning: Output directory already contains data. Skipping..."
  exit 0
fi

# Create necessary directories
mkdir -p "$output_dir/phone"
mkdir -p "$output_dir/word"
mkdir -p "$output_dir/text"
mkdir -p "$output_dir/wav"
mkdir -p "$output_dir/conf"

# Check for required files
[ -f $conf/test_spk.list ] || { echo "Eval-set speaker list not found."; exit 1; }
[ -f $conf/dev_spk.list ] || { echo "Dev-set speaker list not found."; exit 1; }
cp $conf/test_spk.list $conf/dev_spk.list "$output_dir/conf/"

# Copy phone map and create derived map
if [ -f $conf/phones.60-48-39.map ]; then
  cp $conf/phones.60-48-39.map "$output_dir/conf/"
  # Create phone mapping for labels
  awk '{print $1 "\t" $3}' $conf/phones.60-48-39.map | sed 's:^q\t$:q\tsil:' > "$output_dir/conf/phones.61-39.map"
else
  echo "Warning: phones.60-48-39.map not found."
  exit 1
fi

# Check directory case
uppercased=false
train_dir=train
test_dir=test
if [ -d "$timit_dir/TRAIN" ]; then
  uppercased=true
  train_dir=TRAIN
  test_dir=TEST
fi

# Create temporary directory
tmpdir=$(mktemp -d /tmp/kaldi_timit.XXXX)
trap 'rm -rf "$tmpdir"' EXIT

# Get the list of speakers for each set
if $uppercased; then
  tr '[:lower:]' '[:upper:]' < $conf/dev_spk.list > $tmpdir/dev_spk
  tr '[:lower:]' '[:upper:]' < $conf/test_spk.list > $tmpdir/test_spk
  ls -d "$timit_dir"/TRAIN/DR*/* | sed -e "s:^.*/::" > $tmpdir/all_train_spk
else
  tr '[:upper:]' '[:lower:]' < $conf/dev_spk.list > $tmpdir/dev_spk
  tr '[:upper:]' '[:lower:]' < $conf/test_spk.list > $tmpdir/test_spk
  ls -d "$timit_dir"/train/dr*/* | sed -e "s:^.*/::" > $tmpdir/all_train_spk
fi

# Get actual train speakers by removing dev and test speakers
grep -v -f $tmpdir/dev_spk -f $tmpdir/test_spk $tmpdir/all_train_spk > $tmpdir/train_spk

echo "Preparing train, dev and test data"

# Compile timit_normal if available
if [ -f $local/c/timit_normal.cpp ]; then
  g++ $local/c/timit_normal.cpp -o $local/c/timit_normal || { echo "Failed to compile timit_normal.cpp"; exit 1; }
  has_converter=true
else
  echo "Warning: timit_normal.cpp not found. Will use simple file copying."
  has_converter=false
fi

# Initialize YAML file sections
> "$tmpdir/train_yaml"
> "$tmpdir/dev_yaml"
> "$tmpdir/test_yaml"

# Process each dataset
for x in train dev test; do
  echo "Processing $x set..."

  # Determine source directory based on set type
  if [ "$x" == "train" ]; then
    src_dir=$train_dir
    spk_list="$tmpdir/train_spk"
    yaml_section="$tmpdir/train_yaml"
  elif [ "$x" == "dev" ]; then
    src_dir=$test_dir
    spk_list="$tmpdir/dev_spk"
    yaml_section="$tmpdir/dev_yaml"
  else # test
    src_dir=$test_dir
    spk_list="$tmpdir/test_spk"
    yaml_section="$tmpdir/test_yaml"
  fi

  # Find all files except SA utterances
  find "$timit_dir/$src_dir/" -name "*.WAV" -not \( -iname 'SA*' \) | grep -f "$spk_list" > "$tmpdir/${x}_wav.flist"
  find "$timit_dir/$src_dir/" -name "*.PHN" -not \( -iname 'SA*' \) | grep -f "$spk_list" > "$tmpdir/${x}_phn.flist"
  find "$timit_dir/$src_dir/" -name "*.WRD" -not \( -iname 'SA*' \) | grep -f "$spk_list" > "$tmpdir/${x}_wrd.flist"
  find "$timit_dir/$src_dir/" -name "*.TXT" -not \( -iname 'SA*' \) | grep -f "$spk_list" > "$tmpdir/${x}_txt.flist"

  # Create utterance IDs
  sed -e 's:.*/\(.*\)/\(.*\).\(WAV\|wav\)$:\1_\2:' "$tmpdir/${x}_wav.flist" > "$tmpdir/${x}_wav_uttids"
  sed -e 's:.*/\(.*\)/\(.*\).\(PHN\|phn\)$:\1_\2:' "$tmpdir/${x}_phn.flist" > "$tmpdir/${x}_phn_uttids"
  sed -e 's:.*/\(.*\)/\(.*\).\(WRD\|wrd\)$:\1_\2:' "$tmpdir/${x}_wrd.flist" > "$tmpdir/${x}_wrd_uttids"
  sed -e 's:.*/\(.*\)/\(.*\).\(TXT\|txt\)$:\1_\2:' "$tmpdir/${x}_txt.flist" > "$tmpdir/${x}_txt_uttids"

  # Process WAV files - create wav.scp and convert all files
  echo "Converting WAV files for $x set..."
  paste "$tmpdir/${x}_wav_uttids" "$tmpdir/${x}_wav.flist" | \
    sort -k1,1 | \
    while read -r uttid wav_file; do
      "$sph2pipe" -f wav "$wav_file" > "$output_dir/wav/${uttid}.wav"
      echo "$uttid" >> "$yaml_section"
    done

  # Create wav.scp file (for reference)
  paste "$tmpdir/${x}_wav_uttids" "$tmpdir/${x}_wav.flist" | \
    sort -k1,1 | \
    awk '{printf("%s %s -f wav %s |\n", $1, "'$sph2pipe'", $2);}' > "$output_dir/wav.scp.${x}"

  # Process PHN files
  echo "Processing PHN files for $x set..."
  paste "$tmpdir/${x}_phn_uttids" "$tmpdir/${x}_phn.flist" | \
    sort -k1,1 | \
    while read -r uttid phn_file; do
      if [ "$has_converter" = true ]; then
        $local/c/timit_normal "$phn_file" "$output_dir/conf/phones.61-39.map" > "$output_dir/phone/${uttid}.PHN"
      else
        # Simple copy, preserving format
        cp "$phn_file" "$output_dir/phone/${uttid}.PHN"
      fi
    done

  # Process WRD files
  echo "Processing WRD files for $x set..."
  paste "$tmpdir/${x}_wrd_uttids" "$tmpdir/${x}_wrd.flist" | \
    sort -k1,1 | \
    while read -r uttid wrd_file; do
      if [ "$has_converter" = true ]; then
        $local/c/timit_normal "$wrd_file" > "$output_dir/word/${uttid}.WRD"
      else
        # Simple copy, preserving format
        cp "$wrd_file" "$output_dir/word/${uttid}.WRD"
      fi
    done

  # Process TXT files
  echo "Processing TXT files for $x set..."
  paste "$tmpdir/${x}_txt_uttids" "$tmpdir/${x}_txt.flist" | \
    sort -k1,1 | \
    while read -r uttid txt_file; do
      # Keep the original format but strip the first two fields
      awk '{$1=""; $2=""; print substr($0,3)}' "$txt_file" > "$output_dir/text/${uttid}.TXT"
    done

  echo "Completed processing $x set"
done

# Sort and format YAML entries
sort "$tmpdir/train_yaml" | uniq > "$tmpdir/train_yaml.sorted"
sort "$tmpdir/dev_yaml" | uniq > "$tmpdir/dev_yaml.sorted"
sort "$tmpdir/test_yaml" | uniq > "$tmpdir/test_yaml.sorted"

# Create final YAML file with proper formatting
yaml_file="$output_dir/division_timit.yaml"
echo "train:" > "$yaml_file"
cat "$tmpdir/train_yaml.sorted" | sed 's/^/- /' >> "$yaml_file"
echo "dev:" >> "$yaml_file"
cat "$tmpdir/dev_yaml.sorted" | sed 's/^/- /' >> "$yaml_file"
echo "test:" >> "$yaml_file"
cat "$tmpdir/test_yaml.sorted" | sed 's/^/- /' >> "$yaml_file"

# Report statistics
train_count=$(wc -l < "$tmpdir/train_yaml.sorted")
dev_count=$(wc -l < "$tmpdir/dev_yaml.sorted")
test_count=$(wc -l < "$tmpdir/test_yaml.sorted")
total_count=$((train_count + dev_count + test_count))
wav_count=$(find "$output_dir/wav/" -name "*.wav" | wc -l)
phn_count=$(find "$output_dir/phone/" -name "*.PHN" | wc -l)
wrd_count=$(find "$output_dir/word/" -name "*.WRD" | wc -l)
txt_count=$(find "$output_dir/text/" -name "*.TXT" | wc -l)

echo "Data preparation completed successfully"
echo "Data is available in $output_dir"
echo "Data division is available in $yaml_file"
echo "Statistics:"
echo "  Train set: $train_count files"
echo "  Dev set: $dev_count files"
echo "  Test set: $test_count files"
echo "  Total in YAML: $total_count files"
echo "  WAV files: $wav_count files"
echo "  PHN files: $phn_count files"
echo "  WRD files: $wrd_count files"
echo "  TXT files: $txt_count files"

# Check for mismatches
if [ "$total_count" -ne "$wav_count" ]; then
  echo "WARNING: Mismatch between YAML entries ($total_count) and WAV files ($wav_count)"
fi
if [ "$wav_count" -ne "$phn_count" ]; then
  echo "WARNING: Mismatch between WAV files ($wav_count) and PHN files ($phn_count)"
fi
if [ "$wav_count" -ne "$wrd_count" ]; then
  echo "WARNING: Mismatch between WAV files ($wav_count) and WRD files ($wrd_count)"
fi
if [ "$wav_count" -ne "$txt_count" ]; then
  echo "WARNING: Mismatch between WAV files ($wav_count) and TXT files ($txt_count)"
fi

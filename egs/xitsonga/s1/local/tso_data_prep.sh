tso=$1
test_dir=$2

wav=$test_dir/xitsonga_files.txt 
vad=$test_dir/xitsonga_vad.txt 

mkdir -p data/test

# Prepare the wav.scp utt2spk spk2utt and segments file for xitsonga.
for file in $(cat $wav); do
    find $tso -name $file 
done | sed -r "s:^.*_([a-z0-9]+)_([0-9]+).wav:\1_\2 &:" > data/test/wav.scp
cat data/test/wav.scp | awk '{print gensub(/(.*)_(.*)/, "\\0 \\1", "g", $1)}' > data/test/utt2spk
utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt
cat $vad | awk '{print gensub(/.*_.*_(.*_.*)/, "\\1 \\1", "g", $1), $2, $3}' > data/test/segments

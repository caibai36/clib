stage=1
dataset=
audio_dir=
vad_file=
. utils/parse_options.sh || exit 1 # eg. ./run.sh --stage 1

if [ -z $dataset ] || [ -z $audio_dir ] || [ -z $vad_file ]; then
    echo "./local/zr19_data_prepare.sh --dataset \$dataset --audio_dir \$audio_dir --vad_file \$vad_file"
    exit 1
fi

if [ $stage -le 1 ]; then
    echo "Data preparation for dataset ${dataset}..."
    mkdir -p data/${dataset}
    echo "Preparing wav.scp..."
    for file in $audio_dir/*; do uttid=$(basename $file .wav); echo $uttid $file; done | sort -u > data/$dataset/wav.scp
    echo "number of utterances: $(wc -l data/$dataset/wav.scp): "

    echo "Preparing utt2spk and spk2utt..."
    awk '{print gensub(/^([A-Za-z0-9]+)_([A_Za-z0-9]+)$/, "\\0 \\1", "g", $1)}' data/$dataset/wav.scp > data/${dataset}/utt2spk # wav.scp: S002_01 test/S002_01.wav
    utils/utt2spk_to_spk2utt.pl data/${dataset}/utt2spk > data/${dataset}/spk2utt

    echo "Preparing segment..."
    awk '{print $1}' data/$dataset/wav.scp | grep -f - $vad_file | awk '{print $1, $1, $2, $3}' | sort -k1,1 -u > data/${dataset}/segments # vad_file: S002_01 1.0800 3.4200 
    echo "number of segments: $(wc -l data/$dataset/segments): "
fi

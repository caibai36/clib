# (mlp4) [bin-wu@ahctitan05 s0]$(wsj) head data/train/wav.scp
# BABEL_OP3_402_19703_20141102_191903_C6_scripted /project/nakamura-lab08/Work/bin-wu/share/tools/kaldi/tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 1 /project/nakamura-lab09/Share/Corpora/Speech/multi/Additional_OpenASR2020/javanese/IARPA_BABEL_OP3_402_LDC2020S07/package/IARPA_BABEL_OP3_402/scripted/training/audio/BABEL_OP3_402_19703_20141102_191903_C6_scripted.sph |
# BABEL_OP3_402_19703_20141102_191911_T1_scripted /project/nakamura-lab08/Work/bin-wu/share/tools/kaldi/tools/sph2pipe_v2.5/sph2pipe -f wav -p -c 1 /project/nakamura-lab09/Share/Corpora/Speech/multi/Additional_OpenASR2020/javanese/IARPA_BABEL_OP3_402_LDC2020S07/package/IARPA_BABEL_OP3_402/scripted/training/audio/BABEL_OP3_402_19703_20141102_191911_T1_scripted.sph |
    
mkdir -p exp/data/train exp/data/dev exp/data/test local/sandbox

cat data/train/wav.scp | sed -r "s:(\w+) (.*) \|:\2 > exp/data/train/\1.wav:g" > local/sandbox/extract_wav.sh
cat data/dev/wav.scp | sed -r "s:(\w+) (.*) \|:\2 > exp/data/dev/\1.wav:g" >> local/sandbox/extract_wav.sh
cat data/test/wav.scp | sed -r "s:(\w+) (.*) \|:\2 > exp/data/test/\1.wav:g" >> local/sandbox/extract_wav.sh

chmod +x local/sandbox/extract_wav.sh
./local/sandbox/extract_wav.sh

# cp -r exp/data/ /project/nakamura-lab08/Work/bin-wu/share/data/javanese
# mkdir -p exp/data/train exp/data/dev exp/data/test
# head data/train/wav.scp | sed -r "s:(\w+) (.*) \|:\2 > exp/data/train/\1.wav:g" > local/sandbox/extract_wav.sh
# head data/dev/wav.scp | sed -r "s:(\w+) (.*) \|:\2 > exp/data/dev/\1.wav:g" >> local/sandbox/extract_wav.sh
# head data/test/wav.scp | sed -r "s:(\w+) (.*) \|:\2 > exp/data/test/\1.wav:g" >> local/sandbox/extract_wav.sh
# chmod +x local/sandbox/extract_wav.sh
# ./local/sandbox/extract_wav.sh

# (mlp4) [bin-wu@ahctitan05 s0]$(wsj) cp data/train/segments /project/nakamura-lab08/Work/bin-wu/share/data/javanese/segments/train/
# (mlp4) [bin-wu@ahctitan05 s0]$(wsj) cp data/dev/segments /project/nakamura-lab08/Work/bin-wu/share/data/javanese/segments/dev
# (mlp4) [bin-wu@ahctitan05 s0]$(wsj) cp data/test/segments /project/nakamura-lab08/Work/bin-wu/share/data/javanese/segments/test

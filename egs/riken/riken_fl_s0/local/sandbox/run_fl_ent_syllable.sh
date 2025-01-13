# run_fl_ent.sh
# For sample_tv07_phoneme.txt
./local/sandbox/fl_ent.sh --phoneme_file conf/data/Initials.txt --corpus_file exp/data/sample_tv07_phoneme.txt --output_file exp/fl_ent_syllable/fl_sample_tv07_phoneme.csv --syllable_fl true

# For tv07_phoneme.txt
./local/sandbox/fl_ent.sh --phoneme_file conf/data/Initials.txt --corpus_file exp/data/tv07_phoneme.txt --output_file exp/fl_ent_syllable/fl_tv07_phoneme.csv --syllable_fl true

# For pd99_phoneme.txt
./local/sandbox/fl_ent.sh --phoneme_file conf/data/Initials.txt --corpus_file exp/data/pd99_phoneme.txt --output_file exp/fl_ent_syllable/fl_pd99_phoneme.csv --syllable_fl true

# For tv07_phoneme_tone.txt
./local/sandbox/fl_ent.sh --phoneme_file conf/data/Initials.txt --corpus_file exp/data/tv07_phoneme_tone.txt --output_file exp/fl_ent_syllable/fl_tv07_phoneme_tone.csv --syllable_fl true

# For pd99_phoneme_tone.txt
./local/sandbox/fl_ent.sh --phoneme_file conf/data/Initials.txt --corpus_file exp/data/pd99_phoneme_tone.txt --output_file exp/fl_ent_syllable/fl_pd99_phoneme_tone.csv --syllable_fl true

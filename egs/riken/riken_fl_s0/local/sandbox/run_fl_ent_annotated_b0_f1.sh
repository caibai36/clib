# Base variables
phoneme_file="conf/dict/marmoset_labels.txt"
corpus_file="exp/marmoset_data/b0_f1_compound_calls.txt"

# Bigram for single calls
ngram=2
./local/sandbox/fl_ent.sh --phoneme_file ${phoneme_file} --corpus_file ${corpus_file} --ngram ${ngram} --output_file exp/fl_ent_marmoset/b0_f1_single_calls_${ngram}gram.csv --syllable_fl false

# Bigram for compound calls
./local/sandbox/fl_ent.sh --phoneme_file ${phoneme_file} --corpus_file ${corpus_file} --ngram ${ngram} --output_file exp/fl_ent_marmoset/b0_f1_compound_calls_${ngram}gram.csv --syllable_fl true

# Trigram for single calls
ngram=3
./local/sandbox/fl_ent.sh --phoneme_file ${phoneme_file} --corpus_file ${corpus_file} --ngram ${ngram} --output_file exp/fl_ent_marmoset/b0_f1_single_calls_${ngram}gram.csv --syllable_fl false

# Trigram for compound calls
./local/sandbox/fl_ent.sh --phoneme_file ${phoneme_file} --corpus_file ${corpus_file} --ngram ${ngram} --output_file exp/fl_ent_marmoset/b0_f1_compound_calls_${ngram}gram.csv --syllable_fl true

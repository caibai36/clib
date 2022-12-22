# This model has order 7 but KenLM was compiled to support up to 6.
./run.sh --stage 7 --file_lm data/train/lm/train.char_4gram.arpa --type_lm "char_ngram"
./run.sh --stage 7 --file_lm data/train/lm/train.char_5gram.arpa --type_lm "char_ngram"
./run.sh --stage 7 --file_lm data/train/lm/train.char_6gram.arpa --type_lm "char_ngram"

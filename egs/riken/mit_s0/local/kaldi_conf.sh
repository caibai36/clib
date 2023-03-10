###################################################################################
# File: kaldi_conf.sh
# Implemented at 22:30 on 08 January 2018 by bin-wu.
# Script to create the basic directory structure
###################################################################################
# Please set custom kaldi root.
KALDI_ROOT=/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi  # CHECKME

# Set path.sh: contains the path to the Kaldi source directory
#      replace the first line of path.sh with your custom kaldi root.
#      append at the end of file with the main root and related directories to PATH
sed -e "1 c export KALDI_ROOT=$KALDI_ROOT" $KALDI_ROOT/egs/wsj/s5/path.sh > path.sh

# Set cmd.sh: contains different commands.
echo -e 'export train_cmd="run.pl --mem 4G"\nexport decode_cmd="run.pl --mem 4G"\nexport cuda_cmd="run.pl --gpu 1"' > cmd.sh

# Create symbolic links to essential scripts to build kaldi system.
# steps: contains essential scripts for creating an ASR system
# utils: contains scripts to modify Kaldi files in certain ways
ln -sf $KALDI_ROOT/egs/wsj/s5/steps/ steps
ln -sf $KALDI_ROOT/egs/wsj/s5/utils/ utils

# configuration files for mfcc and etc.
[ -d conf ] || mkdir conf
cp $KALDI_ROOT/egs/wsj/s5/conf/* conf/
sed 's/40 /80 /g' conf/mfcc_hires.conf > conf/mfcc_hires80.conf
# Task dependent. You have to check non-linguistic symbols used in the corpus.
# These operations have priorities. First replacing string, then deleting characters and then replacing characters.
[ -f conf/str_rep.txt ] || touch conf/str_rep.txt      # replace the special strings in original text
[ -f conf/chars_del.txt ] || touch conf/chars_del.txt  # the characters (including non_lang_syms} to be deleted
[ -f conf/chars_rep.txt ] || touch conf/chars_rep.txt  # the characters (including non_lang_syms} to be replaced

# Get some scripts for data preparation.
cp $KALDI_ROOT/egs/wsj/s5/local/cstr_wsj_data_prep.sh local
cp $KALDI_ROOT/egs/wsj/s5/local/wsj_format_data.sh local
cp $KALDI_ROOT/egs/wsj/s5/local/cstr_ndx2flist.pl local
cp $KALDI_ROOT/egs/wsj/s5/local/normalize_transcript.pl local
cp $KALDI_ROOT/egs/wsj/s5/local/flist2scp.pl local
cp $KALDI_ROOT/egs/wsj/s5/local/find_transcripts.pl local
# remove the langauge model part of formatting the data
sed -i -e '/# Next, for each type/, /done/ d' -e'/tmp/ d' -e '/lm/ d' local/wsj_format_data.sh
# only retain the the necessary datasets --- 'test_eval92 train_si284 train_si84 test_dev93' --- for experiments
sed -i 's/train_si284 test_eval92 test_eval93 test_dev93 test_eval92_5k test_eval93_5k test_dev93_5k dev_dt_05 dev_dt_20/test_eval92 train_si284 train_si84 test_dev93/g' local/wsj_format_data.sh

# make local directories for private scripts
# mkdir -p local/{script,src,bin}
mkdir -p local/scripts

##################################################################################
# Other options

# # the directory of corpora
# timit=/project/nakamura-lab01/Share/Corpora/Speech/en/TIMIT/TIMIT  # CHECKME 
# wsj0=/project/nakamura-lab01/Share/Corpora/Speech/en/WSJ/wsj0  # CHECKME
# wsj1=/project/nakamura-lab01/Share/Corpora/Speech/en/WSJ/wsj1  # CHECKME

# # Create symbolic links to essential scripts for VTLN
# # sid: contains the speaker recognition (sid) scripts
# # lid: contains language identification (lid) scripts
# ln -sf $KALDI_ROOT/egs/lre/v1/lid lid    # optional
# ln -sf $KALDI_ROOT/egs/sre08/v1/sid/ sid # optional

# Create configuration files
# # configuration files for mfcc and etc.
# [ -d conf ] || mkdir conf
# cp $KALDI_ROOT/egs/lre/v1/conf/vad.conf conf/ # optional
# # Load kaldi configure files0
# cp $KALDI_ROOT/egs/timit/s5/conf/* conf/
# # We need all the utterances for test set.
# find $timit/TEST -name *PHN -not \( -iname 'SA*' \) | sed  's:.*/\(\w*\)/\w*.PHN:\1:' | tr   '[:upper:]' '[:lower:]' | sort -u > conf/test_spk.list

# # Get some scripts for data preparation.
# cp $KALDI_ROOT/egs/timit/s5/local/timit* local

# # Get the score tools for decoding
# cp steps/score_kaldi.sh local/score.sh

# # Set path.sh: contains the path to the Kaldi source directory
# #      replace the first line of path.sh with your custom kaldi root.
# #      append at the end of file with the main root and related directories to PATH
# sed -e "1 c export KALDI_ROOT=$KALDI_ROOT" \
#     -e "$ a \ " \
#     -e "$ a export MAIN_ROOT=\$PWD/../../.." \
#     -e "$ a export PATH=\${MAIN_ROOT}/utils:\$PATH" $KALDI_ROOT/egs/wsj/s5/path.sh > path.sh

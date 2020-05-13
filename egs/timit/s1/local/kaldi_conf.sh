###################################################################################
# File: kaldi_conf.sh
# Implemented at 22:30 on 08 January 2018 by bin-wu.
# Modified at 15:45 on 09 January 2018 by bin-wu
#          Do experiments on all the utterances of test set instead of training set
# Script to create the basic directory structure needed of running the timit example
####################################################################################

# Please set custom kaldi root.
KALDI_ROOT=/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi # CHECKME
# the directory of timit
timit=/project/nakamura-lab01/Share/Corpora/Speech/en/TIMIT/TIMIT # CHECKME 

# Set path.sh: replace the first line of path.sh with your custom kaldi root.
# path.sh: contains the path to the Kaldi source directory
sed -e "1 c export KALDI_ROOT=$KALDI_ROOT" $KALDI_ROOT/egs/wsj/s5/path.sh > path.sh

# Create symbolic links to essential scripts to build kaldi system.
# steps: contains essential scripts for creating an ASR system
# utils: contains scripts to modify Kaldi files in certain ways
ln -sf $KALDI_ROOT/egs/wsj/s5/steps/ steps
ln -sf $KALDI_ROOT/egs/wsj/s5/utils/ utils

# # Create symbolic links to essential scripts for VTLN
# # sid: contains the speaker recognition (sid) scripts
# # lid: contains language identification (lid) scripts
# ln -sf $KALDI_ROOT/egs/lre/v1/lid lid    # optional
# ln -sf $KALDI_ROOT/egs/sre08/v1/sid/ sid # optional

# Create configuration files
# cmd.sh: contains different commands.
echo -e "export train_cmd=\"run.pl --mem 4G\"\nexport decode_cmd=\"run.pl --mem 4G\"\nexport cuda_cmd=\"run.pl --gpu 1\"" > cmd.sh
# configuration files for mfcc and etc.
[ -d conf ] || mkdir conf
# cp $KALDI_ROOT/egs/lre/v1/conf/vad.conf conf/ # optional
# Load kaldi configure files
cp $KALDI_ROOT/egs/timit/s5/conf/* conf/
cp $KALDI_ROOT//egs/wsj/s5/conf/mfcc_hires.conf conf/
# We need all the utterances for test set.
find $timit/TEST -name *PHN -not \( -iname 'SA*' \) | sed  's:.*/\(\w*\)/\w*.PHN:\1:' | tr   '[:upper:]' '[:lower:]' | sort -u > conf/test_spk.list

# Get some scripts for data preparation.
cp $KALDI_ROOT/egs/timit/s5/local/timit* local

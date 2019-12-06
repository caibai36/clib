#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh

# Get the num_frames.scp, feat_dim.scp, num_tokens.scp, tokenid.scp, vocab_size.scp, feat.scp, token.scp and etc.
# then merge them into a json file indexed by the utterance id.
# If you want to add more information, just create more scp files.

nlsyms=""
lang=""
feat="" # feat.scp
oov="<unk>"
bpecode=""
verbose=1
output_json=""  # the format of espnet
output_dir_of_scps=""
output_utts_json=""  # the json of attributes of utterances

. utils/parse_options.sh

if [ $# != 2 ]; then
    echo "Usage: $0 <data-dir> <dict>";
    echo "Options: [--output-json <json_file>] [--output_dir_of_scps <scps_dir>] [output_utts_json <utts_json_file>]"
    exit 1;
fi

dir=$1
dic=$2
tmpdir=`mktemp -d ${dir}/tmp-XXXXX`
trap 'rm -rf ${tmpdir}' EXIT
rm -f ${tmpdir}/*.scp

# These operations have priorities. First replacing string, then deleting characters and then replacing characters.
mkdir -p conf
[ -f conf/str_rep.txt ] || touch conf/str_rep.txt      # replace the special strings in original text
[ -f conf/chars_del.txt ] || touch conf/chars_del.txt  # the characters (including non_lang_syms} to be deleted
[ -f conf/chars_rep.txt ] || touch conf/chars_rep.txt  # the characters (including non_lang_syms} to be replaced

# Get num_frames.scp, feat_dim.scp
# input, which is not necessary for decoding mode, and make it as an option
if [ ! -z ${feat} ]; then
    if [ ${verbose} -eq 0 ]; then
        [ -f ${dir}/utt2num_frames ] || utils/data/get_utt2num_frames.sh ${dir} &>/dev/null
        cp ${dir}/utt2num_frames ${tmpdir}/num_frames.scp
        feat-to-dim scp:${feat} ark,t:${tmpdir}/feat_dim.scp &>/dev/null
    else
	# Redirecting stdout to stderr to prevent printing warning or error message to merged jsons.
        [ -f ${dir}/utt2num_frames ] || utils/data/get_utt2num_frames.sh ${dir} 1>&2
        cp ${dir}/utt2num_frames ${tmpdir}/num_frames.scp
        feat-to-dim scp:${feat} ark,t:${tmpdir}/feat_dim.scp 1>&2
    fi
fi

# Get token.scp
# We will remove the uttid (-s 1), and split every sentence as a sequence of characters (-n 1),
# preserving the non-linguistic symbols if exist;
# add <sos> and <eos> at the beginning and the ending of each sentence,
if [ ! -z ${bpecode} ]; then
    paste -d " " <(awk '{print $1}' ${dir}/text) <(cut -f 2- -d" " ${dir}/text | spm_encode --model=${bpecode} --output_format=piece) > ${tmpdir}/token.scp
elif [ ! -z ${nlsyms} ]; then
    cat ${dir}/text | \
	tr [A-Z] [a-z] | \
	python cutils/replace_str.py --rep_in=conf/str_rep.txt --sep='#' | \
	cutils/text2token.py -s 1 -n 1 -l ${nlsyms} --chars-delete=conf/chars_del.txt --chars-replace=conf/chars_rep.txt | \
	sed -r -e 's/^(\w*) /\1 <sos> /' -e 's/$/ <eos>/' | \
	tr [A-Z] [a-z] > ${tmpdir}/token.scp
else
    cat ${dir}/text | \
	tr [A-Z] [a-z] | \
	python cutils/replace_str.py --rep_in=conf/str_rep.txt --sep='#' | \
	cutils/text2token.py -s 1 -n 1 --chars-delete=conf/chars_del.txt --chars-replace=conf/chars_rep.txt | \
	sed -r -e 's/^(\w*) /\1 <sos> /' -e 's/$/ <eos>/' | \
	tr [A-Z] [a-z] > ${tmpdir}/token.scp
fi

# Get tokenid.scp, num_tokens.scp
# Count how many tokens for each sentence (num_tokens.scp)
# Convert token sequence to tokenid sequence (tokenid.scp)
# Get the size of vocabulary (vocab_size.scp):
#     $ wc -l data/lang_1char/train_si284_units.txt 
#     34 data/lang_1char/train_si284_units.txt             # the size of the vocabulary: 34
#     $ tail -n 1 data/lang_1char/train_si284_units.txt
#     z 33                                                 # output dimension = the size of the vocabulary: 33 + 1 
cat ${tmpdir}/token.scp | utils/sym2int.pl --map-oov ${oov} -f 2- ${dic} > ${tmpdir}/tokenid.scp
cat ${tmpdir}/tokenid.scp | awk '{print $1 " " NF-1}' > ${tmpdir}/num_tokens.scp # -1 for uncounting first field: the uttid
vocsize=`tail -n 1 ${dic} | awk '{print $2}'`
odim=`echo "$vocsize + 1" | bc`
awk -v odim=${odim} '{print $1 " " odim}' ${dir}/text > ${tmpdir}/vocab_size.scp

# others
if [ ! -z ${lang} ]; then
    awk -v lang=${lang} '{print $1 " " lang}' ${dir}/text > ${tmpdir}/lang.scp
fi
# feats
cat ${feat} > ${tmpdir}/feat.scp

rm -f ${tmpdir}/*.json
for x in ${dir}/text ${dir}/utt2spk ${tmpdir}/*.scp; do
    k=`basename ${x} .scp`
    cat ${x} | cutils/scp2json.py --key ${k} > ${tmpdir}/${k}.json
done

if [ ! -z ${output_json} ] && [ ! -z ${output_utts_json} ]; then
    cutils/mergejson.py --verbose ${verbose} --output-json ${output_json} --output-utts-json ${output_utts_json} ${tmpdir}/*.json  
elif [ ! -z ${output_utts_json} ]; then
    cutils/mergejson.py --verbose ${verbose} --output-utts-json ${output_utts_json} ${tmpdir}/*.json
else
    cutils/mergejson.py --verbose ${verbose} ${tmpdir}/*.json
fi

if [ ! -z ${output_dir_of_scps} ]; then
    rm -rf ${output_dir_of_scps}
    mv ${tmpdir} ${output_dir_of_scps}
else
    rm -fr ${tmpdir}
fi

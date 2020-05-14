#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Last modified at 18:16 on 06 December 2019 by WuBin.
. ./path.sh

# Get the num_frames.scp, feat_dim.scp, num_tokens.scp, tokenid.scp, vocab_size.scp, feat.scp, token.scp and etc.
# then merge them into a json file indexed by the utterance id.
# If you want to add more information, just create more scp files.
feat="" # feat.scp
non_ling_syms=""
output_dir_of_scps=""
output_utts_json=""
oov="<unk>"

. utils/parse_options.sh

if [[ $# != 1 && $# != 2 ]]; then
    echo "Usage: $0 <data-dir> <dict> [--output_dir_of_scps <scps_dir>]"
    exit 1;
fi

dir=$1
[ $# == 2 ] && dict=$2

tmpdir=`mktemp -d ${dir}/tmp-XXXXX`
trap 'rm -rf ${tmpdir}' EXIT
rm -f ${tmpdir}/*.scp

# Get feat.scp, num_frames.scp and feat_dim.scp if feat.scp exists.
if [ ! -z ${feat} ]; then
    cat ${feat} > ${tmpdir}/feat.scp
    # Redirecting stdout to stderr to prevent printing warning or error message to merged jsons.
    [ -f ${dir}/utt2num_frames ] || utils/data/get_utt2num_frames.sh ${dir} 1>&2
    cp ${dir}/utt2num_frames ${tmpdir}/num_frames.scp
    feat-to-dim scp:${feat} ark,t:${tmpdir}/feat_dim.scp 1>&2
fi

# Get token.scp, tokenid.scp, num_tokens.scp and vocab_size.scp
# Add <sos> and <eos> at the beginning and ending of the sentences (token.scp)
# Count how many tokens for each sentence (num_tokens.scp)
# Convert token sequence to tokenid sequence (tokenid.scp)
if [ -f $dir/text ]; then
    local/scripts/text2token.py --text ${dir}/text \
			       --strs-replace-in=conf/str_rep.txt \
			       --strs-replace-sep='#' \
			       --chars-delete=conf/chars_del.txt \
			       --chars-replace=conf/chars_rep.txt \
			       --non-ling-syms=${non_ling_syms} \
			       --skip-ncols=1 \
			       --str2lower | sed -r -e 's/^(\S*) /\1 <sos> /' -e 's/$/ <eos>/' > ${tmpdir}/token.scp
    cat ${tmpdir}/token.scp | utils/sym2int.pl --map-oov ${oov} -f 2- ${dict} > ${tmpdir}/tokenid.scp
    cat ${tmpdir}/tokenid.scp | awk '{print $1 " " NF-1}' > ${tmpdir}/num_tokens.scp # -1 for uncounting first field: the uttid
    vocsize=`tail -n 1 ${dict} | awk '{print $2}'` # Get the index of the last word, assuming the largest index starting from 0 (vocsize=$vocab_size-1).
    odim=`echo "$vocsize + 1" | bc`
    awk -v odim=${odim} '{print $1 " " odim}' ${dir}/text > ${tmpdir}/vocab_size.scp
    cp ${dir}/text ${tmpdir}/text.scp
fi

if [ -f $dir/utt2spk ]; then
    cp ${dir}/utt2spk ${tmpdir}/utt2spk.scp
fi

# Merge all the information of scp files to create utts_json file.
rm -f ${tmpdir}/*.json
for x in ${tmpdir}/*.scp; do
    k=`basename ${x} .scp`
    cat ${x} | local/scripts/scp2json.py --key ${k} > ${tmpdir}/${k}.json
done
local/scripts/mergejson.py --output-utts-json=${output_utts_json} ${tmpdir}/*.json

if [ ! -z ${output_dir_of_scps} ]; then
    rm -rf ${output_dir_of_scps}
    mv ${tmpdir} ${output_dir_of_scps}
else
    rm -fr ${tmpdir}
fi

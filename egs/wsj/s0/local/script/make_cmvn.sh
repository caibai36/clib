#!/bin/bash

# Copyright 2012-2014  Brno University of Technology (Author: Karel Vesely),
#                 
# Apache 2.0.
#
# This script dumps features after CMVN in a new data directory.

# Begin configuration section.
delta_order=0
nj=4
cmd=run.pl
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
   echo "Usage: $0 [options] <data-dir> <feat-dir>"
   echo "e.g.: $0 data feats"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --delta_order <delta_order>                      # apply delta after cmvn"
   exit 1;
fi

datadir=$1
featdir=$2

if [ -f $datadir/raw.scp ]; then
    cp $datadir/raw.scp $datadir/feats.scp # make sure feats.scp store paths of the raw mfcc ark.
else
    cp $datadir/feats.scp $datadir/raw.scp # store the script file of the raw mfcc before CMVN;
fi


sdata=$datadir/split$nj;
cmvn_opts="";#`cat $gmmdir/cmvn_opts 2>/dev/null`

mkdir -p $featdir
[[ -d $sdata && $datadir/feats.scp -ot $sdata ]] || split_data.sh $datadir $nj || exit 1;

# Check files exist,
for f in $sdata/1/feats.scp $sdata/1/cmvn.scp; do
  [ ! -f $f ] && echo "$0: Missing $f" && exit 1;
done

# feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas --delta-order=$delta_order ark:- ark:- |"

# Prepare the output dir,
#utils/copy_data_dir.sh $srcdata $data; rm $data/{feats,cmvn}.scp 2>/dev/null
# Make $feadir an absolute pathname,
[ '/' != ${featdir:0:1} ] && featdir=$PWD/$featdir

# Store the output-features,
$cmd JOB=1:$nj $featdir/make_cmvn.JOB.log \
  copy-feats "$feats" \
  ark,scp:$featdir/feats_cmvn.JOB.ark,$featdir/feats_cmvn.JOB.scp || exit 1;
   
# Merge the scp,
for n in $(seq 1 $nj); do
  cat $featdir/feats_cmvn.$n.scp 
done > $datadir/feats.scp

echo "$0: Done!"

exit 0;

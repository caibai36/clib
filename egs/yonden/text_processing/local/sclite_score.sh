#!/bin/bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -o pipefail # without -u here for conda setting

tag=default

# The "ref", "hyp", and "text" share the identical uttids. For example: an error utterance in the error analysis file "$out_dir/analysis_err" can be
# 210914_1029_田中班_無線機_01740_0145012_0145091,err:A 1 S 1 I 0 D 0 all_tokens 8 corr_tokens 7
# 210914_1029_田中班_無線機_01740_0145012_0145091,utt:保護具はきます。                                <= text
# 210914_1029_田中班_無線機_01740_0145012_0145091,ref:ホ ゴ グ ハ キ マ ス <period>                   <= ref
# 210914_1029_田中班_無線機_01740_0145012_0145091,hyp:ホ ゴ グ ワ キ マ ス <period>                   <= hyp
# 210914_1029_田中班_無線機_01740_0145012_0145091,evl:            S                        
#
ref=../s1/exp/tmp/timit/default/EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp-run0/mfcc39_batchsize32_cutoff1600_labelsmoothing0.05_lr0.001_gradclip5_factor0.5_patience3/eval/beamsize10/ref_word.txt
hyp=../s1/exp/tmp/timit/default/EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp-run0/mfcc39_batchsize32_cutoff1600_labelsmoothing0.05_lr0.001_gradclip5_factor0.5_patience3/eval/beamsize10/hypo_word.txt
text= # extra reference text with same uttid as ref and hypo for easy error analysis

sclite=/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi/tools/sctk/bin/sclite # CHECKME

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1
out_dir=exp/scores/$tag
mkdir -p $out_dir

shyp=/tmp/shyp
sref=/tmp/sref

sed 's/\([^ ]*\) *\(.*\)/\2 (\1)/'  $hyp > $shyp
sed 's/\([^ ]*\) *\(.*\)/\2 (\1)/'  $ref > $sref

# $sclite -r $sref -h $shyp -i rm -o prf -s -O $out_dir -n result
$sclite -r $sref -h $shyp -i rm -o all -s -e utf-8 -O $out_dir -n result > /dev/null
cat $out_dir/result.pra | grep -E "id:|Scores:|Attributes:|REF:|HYP:|Eval:" > $out_dir/result

python local/grep_sclite_error.py --input $out_dir/result --text "$text" > $out_dir/analysis_err
python local/grep_sclite_error.py --input $out_dir/result --text "$text" --print_all > $out_dir/analysis_all

echo "result at: $out_dir/result" 

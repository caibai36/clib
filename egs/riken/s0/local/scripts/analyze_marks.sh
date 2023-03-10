text=data/train_si284/text
result=exp/marks

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. utils/parse_options.sh || exit 1

mkdir -p $result
text_trans=$result/text_trans
mark=$result/marks
text_analysis=$result/text_analysis.txt

cat $text | python local/scripts/text2token.py | tr ' ' '\n' | sort -u | grep -v [a-zA-Z0-9] | sed '/^$/ d'> $mark
cat $text | tr [A-Z] [a-z] > $text_trans
python local/scripts/find_mark.py --in_text="$text_trans" --in_mark="$mark" | tee $text_analysis

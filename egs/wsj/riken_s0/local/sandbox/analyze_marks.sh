text=data/train_si284/text
text_trans=exp/sandbox/analysis/analyze_marks/text_trans
mark=exp/sandbox/analysis/analyze_marks/marks
text_analysis=exp/sandbox/analysis/analyze_marks/text_analysis.txt
cat $text | python cutils/text2token.py | tr ' ' '\n' | sort -u | grep -v [a-zA-Z0-9] | sed '/^$/ d'> $mark
cat $text | tr [A-Z] [a-z] > $text_trans
python cutils/find_mark.py --in_text="$text_trans" --in_mark="$mark" | tee $text_analysis 

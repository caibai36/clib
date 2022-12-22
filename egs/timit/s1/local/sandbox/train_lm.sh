dir=data/train
mkdir $dir/lm
cat $dir/scps/token.scp | cut -d' ' -f2- | sed -e 's/^<sos> //g' -e 's/ <eos>//g' > $dir/lm/$(basename $dir).char.txt
cd $dir/lm/
order=3;file=$(basename $dir).char.txt;ngram-count -text $file -order $order -lm $(basename $dir).char_${order}gram.arpa
cd -

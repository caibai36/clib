hyp=exp/yonden/text_chasen.kana
ref=exp/yonden/text.kana
tag=KER_$(basename $ref)_$(basename $hyp)
./local/sclite_score.sh  --tag $tag --ref $ref --hyp $hyp

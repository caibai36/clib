date
feat=data/test_en/feats.scp
result=eval/abx/embedding/exp/feat/mfcc39
echo "feat: $feat"
echo "result: $result"
python local/feat2embedding.py --feat=$feat --result=$result
date

date
feat=data/test_en_vtln/feats.scp
result=eval/abx/embedding/exp/feat/mfcc39_vtln
echo "feat: $feat"
echo "result: $result"
python local/feat2embedding.py --feat=$feat --result=$result
date

date
feat=data/test_en_hires/feats.scp
result=eval/abx/embedding/exp/feat/mfcc40
echo "feat: $feat"
echo "result: $result"
python local/feat2embedding.py --feat=$feat --result=$result
date

date
feat=data/test_en_hires80/feats.scp
result=eval/abx/embedding/exp/feat/mfcc80
echo "feat: $feat"
echo "result: $result"
python local/feat2embedding.py --feat=$feat --result=$result
date

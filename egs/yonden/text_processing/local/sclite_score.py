# result file 'result.pra' from the command of '$sclite -r $sref -h $shyp -i rm -o all -s -O -e utf-8 $out_dir -n result'
# Eval:        D   S                                        S                                                 D               S      

# id: (MWVW0_SX396)
# Scores: (#C #S #D #I) 38 10 3 1
# Attributes: Case_sensitve 
# REF:  sil dh ix f ih sh vcl b iy vcl g ae n cl t  l iy cl f r ae n cl  t  ix cl k l iy ax n dh ax s er f ix s ax v dh ax s epi m ao * l ey cl k sil 
# HYP:  sil dh ax f ih sh cl  p ix vcl g ae n cl ch l iy cl f r ae n vcl jh ix cl k l iy ** * ah m  s er f ix s ax v ** ax s epi m ao l l iy cl k sil 
# Eval:        S          S   S S                S                   S   S               D  D S  S                   D                I   S           

# input_file = "exp/scores/default/result.pra"
from pprint import pprint
input_file = "exp/scores/test.txt"
def get_utt2inf(input_file):
    utt2inf = {}
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip() and ("Scores:" in line or "id:" in line or "REF:" in line or "HYP:" in line or "Eval:" in line or "Attributes:" in line) : # not empty line
                if "Eval" not in line: line = line.strip()
                if "Eval" not in line:
                    key, value = line.split(maxsplit=1)
                else:
                    key, value = line.split(" ", maxsplit=1)
                if (key == "id:"):
                    uttid = value[1:len(value)-1] # (MWVW0_SX396)
                    utt2inf[uttid] = {}
                    utt2inf[uttid]['uttid'] = uttid
                if (key == "Scores:"):
                    _, _, _, _, count, substitution, deletion, insertion = value.split()
                    utt2inf[uttid]['count'] = int(count)
                    utt2inf[uttid]['sub'] = int(substitution)
                    utt2inf[uttid]['del'] = int(deletion)
                    utt2inf[uttid]['ins'] = int(insertion)
                    utt2inf[uttid]['all'] = int(substitution) + int(deletion) + int(insertion)
                if (key ==  "REF:"):
                    utt2inf[uttid]['ref'] = value
                if (key == "HYP:"):
                    utt2inf[uttid]['hyp'] = value
                if (key == "Eval:"):
                    utt2inf[uttid]['evl'] = value[:-1]
                if (key == "Attributes:"):
                    utt2inf[uttid]['att'] = value
    return utt2inf
pprint(get_utt2inf(input_file), width=200)

# {'MWVW0_SX36': {'all': 5,
#                 'att': 'Case_sensitve',
#                 'count': 41,
#                 'del': 2,
#                 'evl': '       D   S                                        S                                                 D               S      ',
#                 'hyp': 'sil ow * l ey dh ax m ow s cl t ix cl k aa m cl p l ix sh aa r dx ih s sil ax vcl t ey n cl p aa cl p * ix l eh er dx iy sil',
#                 'ins': 0,
#                 'ref': 'sil ow n l ix dh ax m ow s cl t ix cl k aa m cl p l ih sh aa r dx ih s sil ax vcl t ey n cl p aa cl p y ix l eh er dx ix sil',
#                 'sub': 3},
#  'MWVW0_SX396': {'all': 14,
#                  'att': 'Case_sensitve',
#                  'count': 38,
#                  'del': 3,
#                  'evl': '       S          S   S S                S                   S   S               D  D S  S                   D                I   S           ',
#                  'hyp': 'sil dh ax f ih sh cl  p ix vcl g ae n cl ch l iy cl f r ae n vcl jh ix cl k l iy ** * ah m  s er f ix s ax v ** ax s epi m ao l l iy cl k sil',
#                  'ins': 1,
#                  'ref': 'sil dh ix f ih sh vcl b iy vcl g ae n cl t  l iy cl f r ae n cl  t  ix cl k l iy ax n dh ax s er f ix s ax v dh ax s epi m ao * l ey cl k sil',
#                  'sub': 10}}

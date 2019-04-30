. path.sh
set -v
chars_del=cutils/tests/data/char_del.txt
chars_rep=cutils/tests/data/char_rep.txt

cat cutils/tests/data/dump/text.scp

cutils/text2token.py -n 1 -s 1 cutils/tests/data/dump/text.scp -d $chars_del -r $chars_rep | tee cutils/tests/data/dump/token.scp

cat cutils/tests/data/dump/text.scp | cutils/text2token.py -n 1 -s 1 -d $chars_del -r $chars_rep | tee cutils/tests/data/dump/token.scp

cat cutils/tests/data/dump/token.scp | tr -s ' ' '\n' | sort -u  | sed '1,2 d' | awk '{print $0 " " NR}' > cutils/tests/data/dump/dict

cat cutils/tests/data/dump/token.scp | perl utils/sym2int.pl --map-oov '<space>' -f 2-  cutils/tests/data/dump/dict > cutils/tests/data/dump/tokenid.scp

# cat cutils/tests/data/dump/token.scp | scp2json.py -k token | tee cutils/tests/data/dump/token.json

# cat cutils/tests/data/dump/text.scp | scp2json.py -k text | tee cutils/tests/data/dump/text.json
# cat cutils/tests/data/dump/tokenid.scp | scp2json.py -k tokenid | tee cutils/tests/data/dump/tokenid.json

# # mergejson.py cutils/tests/data/dump/*json --output-json cutils/tests/data/data.json
# mergejson.py cutils/tests/data/dump/*json --output-utts-json cutils/tests/data/utts.json --output-json cutils/tests/data/data.json

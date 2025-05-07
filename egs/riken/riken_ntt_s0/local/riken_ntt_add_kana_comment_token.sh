# (mlp) [bin-wu@s186 riken_ntt_s0]$(master *) head data/local/all_kana_comment_token.csv
# session_id,session,subject,month,session_begin_sec,session_end_sec,text,kana,comment,kana_token,kana_comment_token
# kk001_1,kk001_1_0001,kk,00,11.401,13.647,アクモッチャップ＠ 。,アクモッチャップ＠ 。,＊ マーク不正確,ア ク  モ ッ  チャ ッ プ,ア ク  モ ッ  チャ ッ プ
# kk001_1,kk001_1_0002,kk,00,19.429,19.906,どっこいしょ 。,ドッコイショ 。,＊,ド  ッ  コ イ  ショ,ド  ッ  コ イ  ショ
# kk001_1,kk001_1_0003,kk,00,30.070,30.891,ショ クンクン 。,ショ クンクン 。,＊,ショ  ク ン ク ン,ショ  ク ン ク ン
# kk001_1,kk001_1_0004,kk,00,31.760,33.121,ショ クンクン どうしたの ？,ショ クンクン ドーシタノ ？,＊,ショ  ク ン ク ン  ド  ー シ タ ノ,ショ  ク ン ク ン  ド  ー シ タ ノ
# kk001_1,kk001_1_0005,kk,00,35.539,36.082,,,発声,,<vocalization>
# kk001_1,kk001_1_0006,kk,00,37.627,38.439,うんち 臭い ？,ウンチ クサイ ？,,ウ ン チ  ク サ イ,ウ ン チ  ク サ イ
# paste -d, <(cat data/local/all.csv) <(cat data/local/all.csv | sed 1d | awk -F, '{print $8}' | python local/scripts/replace_str.py --rep_in conf/str_rep.txt | python local/scripts/kanaseq2phoneseq.py --print_kana 2>/dev/null | sed 's/ <space>//g' | sed '1i kana_token') > data/local/all_kana_token.csv

# Merge the original csv file with its kana token transcription
paste -d, \
    <(cat data/local/all.csv) \
    <(cat data/local/all.csv |
        # Skip header line
        sed 1d |
        # Extract the 8th column (presumably text content)
        awk -F, '{print $8}' |
        # Replace strings according to the replacement rules
        python local/scripts/replace_str.py --rep_in conf/str_rep.txt |
        # Convert kana sequence to phone sequence, with kana output flag
        python local/scripts/kanaseq2phoneseq.py --print_kana 2>/dev/null |
        # Remove space tokens
        sed 's/ <space>//g' |
        # Add column header
        sed '1i kana_token'
    ) > data/local/all_kana_token.csv

# Process the combined file to add kana comments/tokens
python local/riken_ntt_add_kana_comment_token.py

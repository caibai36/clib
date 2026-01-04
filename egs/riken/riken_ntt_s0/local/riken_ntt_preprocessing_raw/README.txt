# Extract information for each subject:
Session, subject, month, session_begin_sec, session_end_sec, text, kana, comment.

# Loop through each infant (sa, kk, ma, mk, sk)
mkdir -p info
for infant in sa kk ma mk sk; do
    # Create header for the output file
    echo "session_id,session,subject,month,session_begin_sec,session_end_sec,text,kana,comment" | tee "info/$infant.txt"
    # Process each text file for the current infant
    find "/a/data/NTT_Infant/data/item/$infant/" -name "*.txt" | while read -r file; do
        # Extract subject and month from the file path
        subject=$(basename "$(dirname "$(dirname "$file")")")
        month=$(basename "$(dirname "$file")")
        # Extract filename without extension
        filename=$(basename "$file" .txt)
        # Get directory of the current file
        dir=$(dirname "$file")
        # Extract session information from .lbl file, convert to UTF-8 and ensure only one comma
        session_info=$(iconv -f SHIFT_JIS -t UTF-8 "$dir/$filename.lbl" | sed -e 's/ *$//' -e 's/D//g' -e 's/ /,/g' | awk -F',' '{print $1 "," $2}')
        # Extract text, kana, and comment, convert to UTF-8 and remove commas
        text=$(iconv -f SHIFT_JIS -t UTF-8 "$file" | awk '{print $1}' | tr -s '\n' ' ' | sed 's/ *$//' | tr -d ',')
        kana=$(iconv -f SHIFT_JIS -t UTF-8 "$file" | awk '{print $2}' | tr -s '\n' ' ' | sed 's/ *$//' | tr -d ',')
        comment=$(iconv -f SHIFT_JIS -t UTF-8 "$dir/$filename.cmt" | tr -s '\n' ' ' | sed 's/ *$//' | tr -d ',')

        # Extract session_id based on the format
        if [[ $filename =~ ^[a-z]{2}-[0-9]{2}_[1-9] ]]; then
            # For format like sk-05_1_0001
            session_id=$(echo $filename | cut -d'_' -f1,2)
        elif [[ $filename =~ ^[a-z]{2}[0-9]{3}_[1-9] ]]; then
            # For format like sa001_2_0310 or sk000_1_0357
            session_id=$(echo $filename | cut -d'_' -f1,2)
        else
            # Default case: take first 7 characters
            session_id=$(echo $filename | cut -c1-7)
        fi

        # Output the compiled information
        echo "$session_id,$filename,$subject,$month,$session_info,$text,$kana,$comment"
    done | tee -a "info/$infant.txt"
done

# others
for infant in sa kk ma mk sk; do echo "session,subject,month,session_begin_sec,session_end_sec,text,kana,comment"| tee $infant.txt;for file in $(find /a/data/NTT_Infant/data/item/$infant/ -name "*.txt"); do subject=$(basename $(dirname $(dirname $file)));month=$(basename $(dirname $file)); echo -ne $(basename $file .txt)",$subject,$month,$(dir=$(dirname $file); cat $dir/$(basename $file .txt).lbl | sed -e 's/ *$//' -e 's/D//g' | sed 's/ /,/g')",""$(cat $file | awk '{print $1}' | tr -s '\n' ' ' | sed 's/ *$//')","$(cat $file | awk '{print $2}' | tr -s '\n' ' ' |  sed 's/ *$//')","$(dir=$(dirname $file); cat $dir/$(basename $file .txt).cmt | tr -s '\n' ' ' | sed 's/ *$//')"\n"; done | tee -a $infant.txt; done

for infant in sa kk ma mk sk; do infant=sa;echo "session,subject,month,session_begin_sec,session_end_sec,text,kana,comment"| tee $infant.txt;for file in $(find /a/data/NTT_Infant/data/item/$infant/ -name "*.txt"); do subject=$(basename $(dirname $(dirname $file)));month=$(basename $(dirname $file)); echo -ne $(basename $file .txt)",$subject,$month,$(dir=$(dirname $file); cat $dir/$(basename $file .txt).lbl | sed -e 's/ *$//' -e 's/D//g' | sed 's/ /,/g')",""$(cat $file | awk '{print $1}' | tr -s '\n' ' ' | sed 's/ *$//')","$(cat $file | awk '{print $2}' | tr -s '\n' ' ' |  sed 's/ *$//')","$(dir=$(dirname $file); cat $dir/$(basename $file .txt).cmt | tr -s '\n' ' ' | sed 's/ *$//')"\n"; done | tee -a $infant.txt; done


# cat *txt > all.csv remove repeated header lines
# dos2unix all.csv

##################################################################
# Add token
# (mlp) [bin-wu@s186 riken_ntt_s0]$(master *) head data/local/all_kana_comment_token.csv
# session_id,session,subject,month,session_begin_sec,session_end_sec,text,kana,comment,kana_token,kana_comment_token
# kk001_1,kk001_1_0001,kk,00,11.401,13.647,アクモッチャップ＠ 。,アクモッチャップ＠ 。,＊ マーク不正確,ア ク  モ ッ  チャ ッ プ,ア ク  モ ッ  チャ ッ プ
# kk001_1,kk001_1_0002,kk,00,19.429,19.906,どっこいしょ 。,ドッコイショ 。,＊,ド  ッ  コ イ  ショ,ド  ッ  コ イ  ショ
# kk001_1,kk001_1_0003,kk,00,30.070,30.891,ショ クンクン 。,ショ クンクン 。,＊,ショ  ク ン ク ン,ショ  ク ン ク ン
# kk001_1,kk001_1_0004,kk,00,31.760,33.121,ショ クンクン どうしたの ？,ショ クンクン ドーシタノ ？,＊,ショ  ク ン ク ン  ド  ー シ タ ノ,ショ  ク ン ク ン  ド  ー シ タ ノ
# kk001_1,kk001_1_0005,kk,00,35.539,36.082,,,発声,,<vocalization>
# kk001_1,kk001_1_0006,kk,00,37.627,38.439,うんち 臭い ？,ウンチ クサイ ？,,ウ ン チ  ク サ イ,ウ ン チ  ク サ イ
paste -d, <(cat data/local/all.csv) <(cat data/local/all.csv | sed 1d | awk -F, '{print $8}' | python local/scripts/replace_str.py --rep_in conf/str_rep.txt | python local/scripts/kanaseq2phoneseq.py --print_kana 2>/dev/null | sed 's/ <space>//g' | sed '1i kana_token') > data/local/all_kana_token.csv
python local/riken_ntt_add_kana_comment_token.py
##################################################################
# Create speaker
[bin-wu@s186 riken_ntt_s0]$(master) python local/riken_ntt_preprocessing_raw/extract_speaker_linux.py |& tee logs/extract_speaker.log


[bin-wu@s186 riken_ntt_s0]$(master) python local/extract_speaker_linux.py |& tee logs/extract_speaker.log
INFO: Item path: /data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/data/item
INFO: Output directory: data/local
INFO: Processing infants: ['sa', 'kk', 'ma', 'mk', 'sk']
INFO:
============================================================
INFO: Processing infant: sa
INFO: ============================================================
INFO: Processing 82916 tag files for infant: sa
INFO: Found speaker info for 82916 files in 225 sessions for sa
INFO:
============================================================
INFO: Processing infant: kk
INFO: ============================================================
INFO: Processing 22040 tag files for infant: kk
INFO: Found speaker info for 22040 files in 90 sessions for kk
INFO:
============================================================
INFO: Processing infant: ma
INFO: ============================================================
INFO: Processing 68910 tag files for infant: ma
INFO: Found speaker info for 68910 files in 204 sessions for ma
INFO:
============================================================
INFO: Processing infant: mk
INFO: ============================================================
INFO: Processing 33209 tag files for infant: mk
INFO: Found speaker info for 33209 files in 159 sessions for mk
INFO:
============================================================
INFO: Processing infant: sk
INFO: ============================================================
INFO: Processing 62392 tag files for infant: sk
WARNING: Unknown speaker code: Z1 in /data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/data/item/sk/18/sk035_3_0060.tag
WARNING: Unknown speaker code: Z2 in /data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/data/item/sk/18/sk035_3_0061.tag
INFO: Found speaker info for 62392 files in 258 sessions for sk
INFO:
============================================================
INFO: Created data/local/speaker.csv with 269467 entries
INFO: sa: 225 unique sessions
INFO: kk: 90 unique sessions
INFO: ma: 204 unique sessions
INFO: mk: 159 unique sessions
INFO: sk: 258 unique sessions
INFO:
Done! Speaker information saved to data/local/speaker.csv

Administrator@DESKTOP-R7C0RP9 MINGW64 /a/data/NTT_Infant/processing/2_create_info_txt/info
$ wc -l all_kana_comment_token.csv
269468 all_kana_comment_token.csv

[bin-wu@s186 riken_ntt_s0]$(master) wc -l data/local/speaker.csv
269468 data/local/speaker.csv

# Merge speaker to csv
[bin-wu@s186 riken_ntt_s0]$(master) python local/riken_ntt_preprocessing_raw/merge_speaker_to_info_linux.py |& tee logs/merge_speaker_to_info.log
INFO: Main CSV: /data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/processed/all_kana_comment_token.csv
INFO: Speaker CSV: data/local/speaker.csv
INFO: Output CSV: data/local/all_kana_comment_token_speaker.csv
INFO: Reading /data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/processed/all_kana_comment_token.csv...
INFO: Reading data/local/speaker.csv...
INFO: Main data: 269467 rows
INFO: Speaker data: 269467 rows
INFO: ✓ All sessions have speaker information!
INFO:
============================================================
INFO: Merged file saved to data/local/all_kana_comment_token_speaker.csv
INFO: Total rows: 269467
INFO: ============================================================
INFO:
Speaker distribution:
INFO:
Noise distribution:
INFO:
Direction distribution:
INFO:
Loudness distribution:
speaker
child           146008
mother           60031
father           54469
other_child       4953
other_person      3081
unknown            925
Name: count, dtype: int64
noise
quiet    267173
noisy      2294
Name: count, dtype: int64
direction
NaN               172825
adult_to_child     92868
adult_to_adult      3774
Name: count, dtype: int64
loudness
NaN         250567
too_low      14120
too_high      4780
Name: count, dtype: int64

the total count, just sum all the categories:
	• child: 146,008
	• mother: 60,031
	• father: 54,469
	• other_child: 4,953
	• other_person: 3,081
	• unknown: 925
Total
146,008+60,031+54,469+4,953+3,081+925=269,467146{,}008 + 60{,}031 + 54{,}469 + 4{,}953 + 3{,}081 + 925 = 269{,}467146,008+60,031+54,469+4,953+3,081+925=269,467
Total = 269,467


[bin-wu@s186 riken_ntt_s0]$(master) head data/local/all_kana_comment_token_speaker.csv
session_id,session,subject,month,session_begin_sec,session_end_sec,text,kana,comment,kana_token,kana_comment_token,speaker,noise,loudness,direction,original_tag
kk001_1,kk001_1_0001,kk,0,11.401,13.647,アクモッチャップ＠ 。,アクモッチャップ＠ 。,＊ マーク不正確,ア ク モ ッ チャ ッ プ,ア ク モ ッ チャ ッ プ,mother,quiet,too_low,,M I L
kk001_1,kk001_1_0002,kk,0,19.429,19.906,どっこいしょ 。,ドッコイショ 。,＊,ド ッ コ イ ショ,ド ッ コ イ ショ,mother,quiet,too_low,,M I L
kk001_1,kk001_1_0003,kk,0,30.07,30.891,ショ クンクン 。,ショ クンクン 。,＊,ショ ク ン ク ン,ショ ク ン ク ン,mother,quiet,,adult_to_child,M I B
kk001_1,kk001_1_0004,kk,0,31.76,33.121,ショ クンクン どうしたの ？,ショ クンクン ドーシタノ ？,＊,ショ ク ン ク ン ドー シ タ ノ,ショ ク ン ク ン ドー シ タ ノ,mother,quiet,,adult_to_child,M I B

import sys
import os
import warnings

import re
import argparse
from typing import Union, List, Set

import numpy as np
import pandas as pd

def read_segments(path: str = ""):
    """
    Read the Kaldi segments file into a dict.
    The Kaldi segments file has line format of <uttid> <recid> <begin_sec> <end_sec>.
    The dict maps <uttid> to {"uttid": <uttid>, "recid": <recid>, "begin_sec": <begin_sec>, "end_sec": <end_sec>}.
    """
    segments_dict = {}

    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            uttid, recid, begin_sec, end_sec = re.split('\s+', line)
            segments_dict[uttid] = {"uttid": uttid, "recid": recid, "begin_sec": float(begin_sec), "end_sec": float(end_sec)}

    return segments_dict

def read_scp(path: str = "", value_type: str = ""):
    """
    Read the Kaldi script file
    The Kaldi segments file has line format of <uttid> <content>.
    The dict maps <uttid> to content
    """
    scp_dict = {}

    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            fields = re.split('\s+', line)
            if (len(fields) == 0 or fields[0] == ""):
                continue
            elif (len(fields) == 1):
                scp_dict[fields[0]] = ""
            else:
                uttid, content = re.split('\s+', line, maxsplit=1)
                if value_type == "int":
                    scp_dict[uttid] = int(content)
                elif value_type == "float":
                    scp_dict[uttid] = float(content)
                else:
                    scp_dict[uttid] = content

    return scp_dict

def read_yonden_addinfo(path: str = ""):
    """
    Read the yonden addinfo file into a dict.
    The yonden addinfo file has line format of 発話ID\tPOS付きopenjtalk形態素解析結果\t話者記号_話者名_シーン番号.シーン名. or <uttid>\t<text>\t<speaker_label>_<speaker>_<sceneid>.<scene>
    (e.g., "210907_0808_平岡班_無線機_00020_0063913_0064079\t作業者|サギョーシャ|名詞|一般 マツナガ|マツナガ|名詞|固有名詞\tC_マツナガ_0-0.自己紹介")

    The dict maps <uttid> to {"uttid": <uttid>, "text": <text>, "speaker_label": <speaker_label>, "speaker": <speaker>, "sceneid": <sceneid>, "scene": <sceneid> + "." + <scene>}.
    """
    addinfo_dict = {}

    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()

            # Fix a problem sentence
            if (line == "220620_1330_中讃_増田班_00640_0210700_0210830\t了解|リョーカイ|名詞|サ変接続 。|。|記号|*\t指_作業指揮者_"):
                warnings.warn("Fixing '指_作業指揮者'_ => '指_作業指揮者_3-3.(TR)送電作業' for utternace: '{}'\n".format(line))
                line = "220620_1330_中讃_増田班_00640_0210700_0210830\t了解|リョーカイ|名詞|サ変接続 。|。|記号|*\t指_作業指揮者_3-3.(TR)送電作業"

            uttid = text = speaker_label = speaker = sceneid = scene = "NA"
            format1 = re.findall("^([^\t]+)\t([^\t]+)\t(.*)_(.*)_(.*)\.(.*)$", line) # 指_作業指揮者_3-3.(TR)送電作業
            format2 = re.findall("^([^\t]+)\t([^\t]+)\t(\d.*)\.(.*)$", line) # 3-3.(TR)送電作業
            assert len(format1) == 1 or len(format2) == 1, "The addinfo line format problem with line:\n{}\n\n".format(line.replace('\t','\\t')) +  r"Addinfo line format should be 発話ID\tPOS付きopenjtalk形態素解析結果\t話者記号_話者名_シーン番号.シーン名." +  "\n" + r"or <uttid>\t<text>\t<speaker_label>_<speaker>_<sceneid>.<scene>"
            if format1: uttid, text, speaker_label, speaker, sceneid, scene = format1[0]
            if format2: uttid, text, sceneid, scene = format2[0]
            if speaker == "#N/A": speaker = "NA"

            addinfo_dict[uttid] = {"uttid": uttid, "text": text, "speaker_label": speaker_label, "speaker": speaker, "sceneid": sceneid, "scene": sceneid+ "." + scene}

    return addinfo_dict

def read_yonden_uttid_to_df_uttinfo(uttids: Union[List, Set] = set()) -> pd.DataFrame:
    """
    Read information from the uttid in yonden format (e.g. 210907_0808_平岡班_無線機_00020_0063913_0064079 for date_starttime_group_device_diagindex_begintime_endtime)
    Dataset information such 'data1' and 'data2' according to 四電データNAIST管理表.20220627 will be added.

    return a data frame with columns of  ['data', 'work_date', 'work_start_time', 'group', 'dialog_index']
    """
    uttinfo_dict = {}
    data_dict = {}
    # create other information from uttid information
    for uttid in uttids:
        if len(re.split("_", uttid)) == 7:
            work_date, work_start_time, group, device, dialog_index, audio_begin_time, audio_end_time =  re.findall('^(\d+)_(\d+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)', uttid)[0]

            # Extract dataset information from a map from time to data index
            time = work_date + "_" + work_start_time
            if time in time2data:
                data_dict[uttid] = int(time2data[time])
            else:
                print("Warning: the dataset for data_starttime '{}' is not available: the data column of df_info with uttid '{}' is set to be 'NA' (not available).".format(time, uttid))
                data_dict[uttid] = np.NaN
        else:
            print("Warning: invalid uttid '{}' with more then 7 items, all information for this uttid set to be 'NA'".format(uttid))
            work_date = work_start_time = group = device = dialog_index = audio_begin_time = audio_end_time = "NA"
            data_dict[uttid] = np.NaN

        # data1-data17 220517_0620_平岡班_無線機_00010_0000476_0000641
        # data18-data28 220628_0834_高松_平岡班_00010_0000135_0000283
        if (data_dict[uttid] <= 17):
            location = "NA"
        else:
            location = group
            group = device
            device = "NA"

        uttinfo_dict[uttid] = [data_dict[uttid], work_date, work_start_time, group, location, dialog_index]

    df_uttinfo = pd.DataFrame.from_dict(uttinfo_dict, orient='index', columns = ['dataset_index', 'work_date', 'work_start_time', 'group', 'location', 'dialog_index'])
    return df_uttinfo


parser = argparse.ArgumentParser(description="Create json or csv format file including data information.\nThe wav_scp file is paired with the segments file by the 'recid' column if both are available.\n\nExample:\n    python local/data2info.py --data $timit/s0/data/test\n    python local/data2info.py --data $dir kana:$dir/text.kana phone:$dir/text.phone --outjson $dir/info.json", formatter_class=argparse.RawTextHelpFormatter)
# Directory options
data = parser.add_argument("--data", type=str, default="", help="kaldi data directory, less priority than --wav_scp, --segments and --addinfo options. 'spk2utt' not included.")
# File options
scps = parser.add_argument('--scps', type=str, nargs="+", default=[], help="a list of kaldi script files with line format <utterance_id> <content>,\nllsted as '<column_name>:<path_to_scp> <column_name2>:<path_to_scp2> ...'\n(e.g., --scps kana:exp/del/text.kana phone:exp/del/text.phone) ")
json = parser.add_argument('--outjson', type=str, default="", help="output information in json format. e.g., --outjson ./info.json")
csv = parser.add_argument('--outcsv', type=str, default="", help="output information in csv format. e.g., --outcsv ./info.csv")
# Yonden data
time2data = parser.add_argument('--time2data', type=str, default="", help="time2data file for yonden data with each line in format of date_starttime:dataset_index (e.g., 210907_0808:3) (Optional from data3 to data36)")
normalize_yonden_data = parser.add_argument('--normalize_yonden_data', type=bool, default=True, help="Normalize the yonden data (default True) (e.g., '作業指揮者’ => ‘指揮者マスダ’ if in 増田班)")
addinfo = parser.add_argument('--addinfo', type=str, default="", help="the yonden addinfo file with line format of <utterance_id>\\t<text>\\t<speaker_label>_<speaker>_<sceneid>.<scene> (optional)")
# Alternative options
wav_scp = parser.add_argument('--wav_scp', type=str, default="", help="the kaldi 'wav.scp' file with line format of <recording_id> <recording_path> (optional)")
segments = parser.add_argument('--segments', type=str, default="", help="the kaldi 'segments' file with line format of <utterance_id> <recording_id> <begin_sec> <end_sec> (optional)")


args = parser.parse_args()

df_info = None

# Processing wav.scp and segments files that share the recid field
wav_scp = args.wav_scp
segments = args.segments
addinfo = args.addinfo
scps = args.scps
json_file = args.outjson
csv_file = args.outcsv
data = args.data
normalize_yonden_data = args.normalize_yonden_data

if data:
    if not os.path.exists(data):
        print("ERROR: data directory of '{}' not found".format(data))
        sys.exit(1)

    # Kaldi typical files
    kaldi_wav = os.path.join(data, "wav.scp")
    kaldi_segments = os.path.join(data, "segments")

    if not wav_scp and os.path.exists(kaldi_wav):
        wav_scp = kaldi_wav
    if not segments and os.path.exists(kaldi_segments):
        segments = kaldi_segments

    # yonden addinfo file
    yonden_addinfo = os.path.join(data, "text.addinfo")
    if not addinfo and os.path.exists(yonden_addinfo):
        addinfo = yonden_addinfo

if args.time2data:
    time2data = {}
    with open(args.time2data) as f:
        for line in f:
            line = line.strip()
            time, dataset_index = re.split(":", line)
            time2data[time] = int(dataset_index)
else:
    # According to 四電データNAIST管理表.20220906
    time2data = {"210907_0808": 3,
                 "210909_1249": 4,
                 "210914_1029": 5,
                 "210920_1358": 6,
                 "210924_1114": 7,
                 "220210_1351": 8,
                 "220216_1347": 9,
                 "220221_1131": 10,
                 "220311_0920": 11,
                 "220314_1431": 12,
                 "220413_1402": 13,
                 "220418_1154": 14,
                 "220509_0929": 15,
                 "220517_0620": 16,
                 "220526_0919": 17,
                 "220620_1026": 18,
                 "220620_1330": 19,
                 "220627_0916": 20,
                 "220628_0834": 21,
                 "220628_0917": 22,
                 "220628_0948": 23,
                 "220629_1009": 24,
                 "220630_1314": 25,
                 "220630_1410": 26,
                 "220701_1058": 27,
                 "220701_0930": 28,
                 "220701_1546": 29,
                 "220703_0928": 30,
                 "220703_0924": 31,
                 "220704_0931": 32,
                 "220705_0904": 33,
                 "220705_0941": 34,
                 "220706_1516": 35,
                 "220706_0934": 36}

if wav_scp:
    wav_dict = read_scp(wav_scp)
    df_wav = pd.DataFrame.from_dict(wav_dict, orient='index', columns=["wav"])

if segments:
    segments_dict = read_segments(segments)
    df_segments = pd.DataFrame(segments_dict).T
    df_segments = df_segments.rename(columns={'uttid': 'id'})

# Create df_info from wav.scp and segments
if wav_scp and segments:
    df_wav['recid'] = df_wav.index
    df_info = pd.merge(df_segments, df_wav, on='recid', how='left')
    df_info.index = df_info["id"].values
elif wav_scp:
    df_info = df_wav
elif segments:
    df_info = df_segments
else:
    df_info = pd.DataFrame()

if addinfo:
    # Read yonden additional information into df_info
    addinfo_dict = read_yonden_addinfo(addinfo)
    df_addinfo = pd.DataFrame(addinfo_dict).T
    df_addinfo = df_addinfo.rename(columns={'uttid': 'id'})
    if isinstance(df_info, pd.DataFrame):
        if ("id" in df_info.columns and "id" in df_addinfo.columns): del df_addinfo['id'] # remove duplicated id columns
        df_info = pd.merge(df_info, df_addinfo, how="outer", left_index=True, right_index=True) # merge by index

if data:
    # Kaldi typical files besides wav.scp and segments to keep order of keys in json
    kaldi_text = os.path.join(data, "text")
    kaldi_utt2spk = os.path.join(data, "utt2spk")
    kaldi_spk2gender = os.path.join(data, "spk2gender")

    if not addinfo and os.path.exists(kaldi_text):
        df_text = pd.DataFrame.from_dict(read_scp(kaldi_text), orient="index", columns=["text"])
        df_info = pd.merge(df_info, df_text, how="left", left_index=True, right_index=True).replace(np.NaN, "NA")

    if not addinfo and os.path.exists(kaldi_utt2spk):
        df_utt2spk = pd.DataFrame.from_dict(read_scp(kaldi_utt2spk), orient="index", columns=["speaker"])
        df_info = pd.merge(df_info, df_utt2spk, how="left", left_index=True, right_index=True).replace(np.NaN, "NA")

    if os.path.exists(kaldi_spk2gender):
        df_spk2gender = pd.DataFrame.from_dict(read_scp(kaldi_spk2gender), orient="index", columns=["gender"])
        df_utt2spk = pd.DataFrame.from_dict(read_scp(kaldi_utt2spk), orient="index", columns=["speaker"])
        df_utt2gender =  pd.DataFrame.merge(df_utt2spk, df_spk2gender, how='left', left_on="speaker", right_index=True)[['gender']]
        df_info = pd.merge(df_info, df_utt2gender, how="left", left_index=True, right_index=True).replace(np.NaN, "NA")

if addinfo or args.time2data:
    # Read yonden utterance information into df_info
    df_uttinfo = read_yonden_uttid_to_df_uttinfo(df_info.index)
    df_info = pd.merge(df_info, df_uttinfo, how="outer", left_index=True, right_index=True)

if data:
    # additional information about audio
    kaldi_dur_sec = os.path.join(data, "utt2dur")
    if os.path.exists(kaldi_dur_sec):
        df_dur_sec = pd.DataFrame.from_dict(read_scp(kaldi_dur_sec, value_type="float"), orient="index", columns=["dur_sec"])
        df_info = pd.merge(df_info, df_dur_sec, how="left", left_index=True, right_index=True).replace(np.NaN, "NA")

    kaldi_num_frames = os.path.join(data, "utt2num_frames")
    if os.path.exists(kaldi_num_frames):
        df_num_frames = pd.DataFrame.from_dict(read_scp(kaldi_num_frames, value_type="int"), orient="index", columns=["num_frames"])
        df_info = pd.merge(df_info, df_num_frames, how="left", left_index=True, right_index=True).replace(np.NaN, "NA")

    # yonden files
    for yonden_file in ["text.addinfo", "text.am" ,"text.am.chasen", "text.am.pos", "text.eval"]:
        yonden_path = os.path.join(data, yonden_file)

        if os.path.exists(yonden_path):
            df_yonden = pd.DataFrame.from_dict(read_scp(yonden_path), orient="index", columns=[yonden_file])
            df_info = pd.merge(df_info, df_yonden, how="left", left_index=True, right_index=True).replace(np.NaN, "NA")

if scps:
    df_scps_list = []
    for token in scps:
        scp_name, scp_file = token.split(":")
        scp_dict = read_scp(scp_file)
        df_scp = pd.DataFrame.from_dict(scp_dict, orient='index', columns=[scp_name])
        df_info = pd.merge(df_info, df_scp, how="left", left_index=True, right_index=True).replace(np.NaN, "NA")

if not addinfo: df_info['id'] = df_info.index

if normalize_yonden_data and "group" in df_info.columns and "speaker_label" in df_info.columns: # The yonden data has fields of 'group' and 'speaker_label'
    # Normalize speaker information
    # Counter({'指揮者ヒラオカ': 417, 'NA': 302, '作業指揮者': 288, '指揮者タナカ': 204, 'ナカヤマ': 155, 'イワモト': 95, 'ヨシダ': 95, 'ナガレ': 84, 'フクダ': 79, '平岡': 76, 'コバヤシ': 72, 'イノウエ': 71, '中山': 60, 'ヤノ': 57, 'スナゴ': 55, '亀田': 51, 'オオモリ': 48, '岩本': 48, 'カタヤマ': 27, 'シノハラ': 27, 'オカダ': 23, 'トミナガ': 20, '作業者指揮者': 19, 'クサナギ': 18, 'カメダ': 18, 'マツナガ': 13, 'アベ': 13, 'バンドウ': 12, 'オカザキ': 12, 'オグラ': 8, 'フクモト': 6, 'ホリケ': 6, 'モリ': 6, 'ゴウダ': 4, '安部': 3, '': 2, 'フジタ': 2, 'イケダ': 2, 'ウエダ': 1, 'ホソカワ': 1, '-': 1, 'ヒラオ': 1})
    group2speakerkana = {'平岡班': 'ヒラオカ', '田中班': 'タナカ', '松本班':'マツモト', '増田班': 'マスダ', '津田班': 'ツダ', '松本班1': 'マツモト'}
    # Speaker converting from '作業指揮者' to '指揮者マスダ' if in the group of '増田班'
    df_info.loc[df_info.speaker == "作業指揮者", 'speaker'] = "指揮者" + df_info.loc[df_info.speaker == "作業指揮者", 'group'].apply(lambda x: group2speakerkana[x]) 
    df_info.loc[df_info.speaker == "作業者指揮者", 'speaker'] = "指揮者" + df_info.loc[df_info.speaker == "作業者指揮者", 'group'].apply(lambda x: group2speakerkana[x])
    speaker_fixed = {'亀田':'カメダ', '平岡':'ヒラオカ', '岩本':'イワモト', '中山':'ナカヤマ', '安部':'アベ', '':'NA', '-':'NA'}
    df_info.loc[:, 'speaker'] = df_info.loc[:, 'speaker'].apply(lambda x: speaker_fixed[x] if x in speaker_fixed else x)
    df_info.loc[df_info.speaker_label == "指", 'speaker'] = df_info.loc[df_info.speaker_label == "指" , 'speaker'].apply(lambda x: "指揮者" + x  if ((not x.startswith("作業者")) and (not x.startswith("指揮者")) and x != "NA") else x)

    # Normalize speaker label
    # Counter({'指': 1037, 'B': 461, 'A': 388, 'C': 330, '': 225, 'D': 33, '？': 14, 'E': 13, '-': 1})
    df_info.loc[:, 'speaker_label'] = df_info.loc[:, 'speaker_label'].apply(lambda x: "NA" if x in ['？', '-', ''] else x)

if csv_file: df_info.to_csv(csv_file)
if json_file:
    df_info.T.to_json(json_file, indent=4, force_ascii=False)
else:
    if len(df_info): df_info.T.to_json(sys.stdout, indent=4, force_ascii=False)

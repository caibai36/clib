import os
import sys
sys.path.append(os.getcwd())

from pprint import pprint
import json

import numpy as np

import torchaudio
import torch

from omegaconf import OmegaConf
from clib.audacity import processing

import argparse

def get_chunk_info_json(chunk_size_ms, chunk_shift_ms, wavids, info_json, feat_yaml, token2id_yaml, feat_dir, dataset="data", overwrite_feat=True):
    """Get the dict of information of chunks from config files.

    Parameters
    ----------
    chunk_size_ms : The size of a chunk (in millisecond) that contains many frames
    chunk_shift_ms : The shift of a chunk (in millisecond)
    wavids : a list of wave ids. The list of pair1 and pair2 in the format of ([pair1_first, pair1_second, pair2_first, pair2_second, ...])
    info_json : the info.json that contains a map from wavids to the locations of wavefiles and segments (audacity)
    feat_yaml : the path of the yaml file or the config dict of the feature
    token2id_yaml : the yaml file that contains the token2id mapping
    feat_dir : the directory that would store the feature matrices
    dataset : the name of the dataset
    overwrite_feat : Overwrite the existing stored feature matrix when True

    Examples
    --------
    # Lists of wave ids ([pair1_first, pair1_second, pair2_first, pair2_second, ...]) for training, development, and test sets
    data_division_yaml = "conf/data/division.yaml"
    data_division = OmegaConf.load(data_division_yaml)

    chunk_size_ms = 500
    chunk_shift_ms = 400
    wavids = data_division['test'] + data_division['dev'] # pair1 and pair2 of lists like ([pair1_first, pair1_second, pair2_first, pair2_second, ...])
    info_json = "data/mit_data/info.json" # info_json = "data/mit_data/info_audio_only.json" # info_json = "data/mit_data/info_audio_only_single_audio.json"
    feat_yaml = "conf/feat/feat.yaml"
    token2id_yaml = "conf/dict/token2id_marmoset.yaml"
    feat_dir = "feat/test"

    info = get_chunk_info_json(chunk_size_ms=chunk_size_ms, chunk_shift_ms=chunk_shift_ms, wavids=wavids, info_json=info_json, feat_yaml=feat_yaml, token2id_yaml=token2id_yaml, feat_dir=feat_dir)
    pprint(info)

    Info
    ----
    The chunk info would contain the a map from uttid to a dict that contains keys of
    ['feat', 'feat_dim', 'num_frames', 'text', 'token', 'tokenid', 'num_tokens', 'vocab_size', 'utt2spk', 'uttid', 'info']
    Where the uttid is f"{audio_id1}_{audio_id2}_{chunk_id}" and the values of 'info' contains
    'wavid1', 'wavid2', 'chunkid', 'num_chunks', 'begin_frame_index', 'end_frame_index', 'begin_sec', 'end_sec', 'chunk_size_num_frames', 'chunk_shift_num_frames', 'frame_size_sec', 'frame_shift_sec', 'wav1', 'wav2', 'seg1', 'seg2', 'state'] and value of state contains "all_noises" or "has_calls"
    """
    instances = OmegaConf.load(info_json)
    feat = OmegaConf.load(feat_yaml) if type(feat_yaml) == str else feat_yaml
    token2id = OmegaConf.load(token2id_yaml)
    vocab_size = len(token2id)
    if not os.path.exists(feat_dir): os.makedirs(feat_dir)

    wavids_first = wavids[::2] # e.g., ['p1a1_toget', 'p2a1_toget']; the first animals of pair1, pair2, pair3, etc.
    wavids_second = wavids[1::2] # e.g, ['p1a2_toget', 'p2a2_toget']; the second animals of pair1, pair2, pair3, etc.
    assert len(wavids_first) == len(wavids_second), f"wavids: '{wavids}' should be even numbered such as ([pair1_first, pair1_second, pair2_first, pair2_second, ...])"

    feat_name = f"{dataset}_{feat.feat_type}{feat.feat_dim}x2_csize{chunk_size_ms}cshift{chunk_shift_ms}_fsize{feat.frame_size_ms}fshift{feat.frame_shift_ms}minfreq{feat.min_freq}meannorm{feat.feat_mean_norm}"
    # print(f"{feat_name=}")
    d = os.path.join(feat_dir, feat_name)
    if (not os.path.exists(d)): os.makedirs(d)

    info = dict()
    for i in range(0, len(wavids_first)):
        wav1 = instances[wavids_first[i]].wav
        seg1 = instances[wavids_first[i]].aud
        wav2 = instances[wavids_second[i]].wav
        seg2 = instances[wavids_second[i]].aud

        # print(f"{wav1=},\n{wav2=},\n{seg1=},\n{seg2=},\n{chunk_size_ms=},\n{chunk_shift_ms=}")
        _, _, feat_chunks12, _, _, label_chunks12, chunk_info = processing(wav1=wav1,
                                                               wav2=wav2,
                                                               seg1=seg1,
                                                               seg2=seg2,
                                                               sampling_rate=feat.sampling_rate,
                                                               frame_size_sec=feat.frame_size_ms/1000,
                                                               frame_shift_sec=feat.frame_shift_ms/1000,
                                                               feat_dim=feat.feat_dim,
                                                               min_freq=feat.min_freq,
                                                               feat_type=feat.feat_type,
                                                               chunk_size_sec=chunk_size_ms/1000,
                                                               chunk_shift_sec=chunk_shift_ms/1000,
                                                               feat_mean_norm=feat.feat_mean_norm,
                                                               verbose=True)
        # print(f"{feat_chunks12[0].shape=}\n{len(label_chunks12[0])=}")
        # print(f"{feat_chunks12[1].shape=}\n{len(label_chunks12[1])=}")
        # print(f"{feat_chunks12[-1].shape=}\n{len(label_chunks12[-1])=}")
        # print(f'{len(chunk_info["chunk_begin_end_indices"])=}\n{len(chunk_info["chunk_begin_end_seconds"])=}')

        audio_id1 = instances[wavids_first[i]].id
        audio_id2 = instances[wavids_second[i]].id
        wav1 = instances[wavids_first[i]].wav
        wav2 = instances[wavids_second[i]].wav
        aud1 = instances[wavids_first[i]].aud
        aud2 = instances[wavids_second[i]].aud

        num_chunks = len(feat_chunks12)
        for chunk_id in range(num_chunks):
            uttid = f"{audio_id1}_{audio_id2}_{chunk_id}"
            info[uttid] = {}
            info[uttid]['feat'] = os.path.abspath(os.path.join(feat_dir, feat_name, f"feats_{uttid}.npy"))
            np.save(info[uttid]['feat'], feat_chunks12[chunk_id].detach().numpy())
            info[uttid]['feat_dim'] = feat_chunks12[chunk_id].shape[-1]
            info[uttid]['num_frames'] = feat_chunks12[chunk_id].shape[0]
            info[uttid]['text'] = " ".join(label_chunks12[chunk_id])
            tokens = ["<sos>"] + label_chunks12[chunk_id] + ["<eos>"]
            info[uttid]['token'] = " ".join(tokens)
            tokenids = [str(token2id[token]) for token in tokens]
            info[uttid]['tokenid'] = " ".join(tokenids)
            info[uttid]['num_tokens'] = len(tokens)
            info[uttid]['vocab_size'] = len(token2id)
            info[uttid]['utt2spk'] = f"{audio_id1}_{audio_id2}" # spoken by two animals for one audio pair
            info[uttid]['uttid'] = uttid

            # additional info
            info[uttid]['info'] = {}
            info[uttid]['info']['wavid1'] = wavids_first[i]
            info[uttid]['info']['wavid2'] = wavids_second[i]
            info[uttid]['info']['chunkid'] = chunk_id
            info[uttid]['info']['num_chunks'] = num_chunks

            info[uttid]['info']['begin_frame_index'] = chunk_info['chunk_begin_end_indices'][chunk_id][0]
            info[uttid]['info']['end_frame_index'] = chunk_info['chunk_begin_end_indices'][chunk_id][1]
            info[uttid]['info']['begin_sec'] = chunk_info['chunk_begin_end_seconds'][chunk_id][0]
            info[uttid]['info']['end_sec'] = chunk_info['chunk_begin_end_seconds'][chunk_id][1]
            info[uttid]['info']['chunk_size_num_frames'] = chunk_info['chunk_size_num_frames']
            info[uttid]['info']['chunk_shift_num_frames'] = chunk_info['chunk_shift_num_frames']
            info[uttid]['info']['frame_size_sec'] = chunk_info['frame_size_sec']
            info[uttid]['info']['frame_shift_sec'] = chunk_info['frame_shift_sec']
            info[uttid]['info']['wav1'] = wav1
            info[uttid]['info']['wav2'] = wav2
            info[uttid]['info']['seg1'] = seg1
            info[uttid]['info']['seg2'] = seg2

            chunk = label_chunks12[chunk_id]
            if set(chunk) == {'noise'}:
                info[uttid]['info']['state'] = "all_noises"
            else:
                info[uttid]['info']['state'] = "has_calls"

    return info


parser = argparse.ArgumentParser(description="Get chunk info json files for datasets.")
parser.add_argument("--data_div_yaml", type=str, default="conf/data/division.yaml", help="the data division yaml containing lists of wave ids ({'train':[pair1_first, pair1_second, pair2_first, pair2_second, ...],...}) for training, development, and test sets (default: 'conf/data/division.yaml')")
parser.add_argument("--info_json", type=str, default="data/mit_data/info.json", help='the info.json that contains a map from wavids to the locations of wavefiles and segments (audacity) (e.g., {"id1":{"wav":wav_path1, "aud":seg_path1, "id":id1}, "id2":{"wav":wav_path2, "aud":seg_path2, "id":id2}) (default: "data/mit_data/info.json")')
parser.add_argument("--feat_yaml", type=str, default="conf/feat/feat.yaml", help='the path of the yaml file or the config dict of the feature (default: "conf/feat/feat.yaml")')
parser.add_argument("--token2id_yaml", type=str, default="conf/dict/token2id_marmoset.yaml", help='the yaml file that contains the token2id mapping (default: "conf/dict/token2id_marmoset.yaml")')
parser.add_argument("--train_chunk_size_ms", type=float, default=500, help='(training set) The size of a chunk (in millisecond) that contains many frames (default: 500)')
parser.add_argument("--train_chunk_shift_ms", type=float, default=150, help='(training set) The shift of a chunk (in millisecond)(default: 150)')
parser.add_argument("--dev_chunk_size_ms", type=float, default=500, help='(development set) The size of a chunk (in millisecond) (default: 500)')
parser.add_argument("--dev_chunk_shift_ms", type=float, default=400, help='(development set) The shift of a chunk (in millisecond)(default: 400)')
parser.add_argument("--test_chunk_size_ms", type=float, default=500, help='(test set) The size of a chunk (in millisecond) (default: 500)')
parser.add_argument("--test_chunk_shift_ms", type=float, default=400, help='(test set) The shift of a chunk (in millisecond)(default: 400)')
parser.add_argument("--feat_dir", type=str, default="feat/default", help='the directory that would store the feature matrices (default: "feat/default")')
parser.add_argument("--chunk_json_dir", type=str, default="data/mit_data/chunk_info/default", help='the directory that stores the chunk info json files (default: "data/mit_data/chunk_info/default")')

args = parser.parse_args()
print(args)

# Get args
data_div_yaml = args.data_div_yaml
info_json = args.info_json
feat_yaml = args.feat_yaml
token2id_yaml = args.token2id_yaml

train_chunk_size_ms = args.train_chunk_size_ms
train_chunk_shift_ms = args.train_chunk_shift_ms
dev_chunk_size_ms = args.dev_chunk_size_ms
dev_chunk_shift_ms = args.dev_chunk_shift_ms
test_chunk_size_ms = args.test_chunk_size_ms
test_chunk_shift_ms = args.test_chunk_shift_ms

feat_dir = args.feat_dir
chunk_json_dir = args.chunk_json_dir

# Process args
data_division = OmegaConf.load(data_div_yaml)
train_wavids = data_division['train']
dev_wavids = data_division['dev']
test_wavids = data_division['test']

if (not os.path.exists(chunk_json_dir)):
    os.makedirs(chunk_json_dir)

# Process chunk info
train_info = get_chunk_info_json(chunk_size_ms=train_chunk_size_ms, chunk_shift_ms=train_chunk_shift_ms, wavids=train_wavids, info_json=info_json, feat_yaml=feat_yaml, token2id_yaml=token2id_yaml, feat_dir=feat_dir, dataset="train")
dev_info = get_chunk_info_json(chunk_size_ms=dev_chunk_size_ms, chunk_shift_ms=dev_chunk_shift_ms, wavids=dev_wavids, info_json=info_json, feat_yaml=feat_yaml, token2id_yaml=token2id_yaml, feat_dir=feat_dir, dataset="dev")
test_info = get_chunk_info_json(chunk_size_ms=test_chunk_size_ms, chunk_shift_ms=test_chunk_shift_ms, wavids=test_wavids, info_json=info_json, feat_yaml=feat_yaml, token2id_yaml=token2id_yaml, feat_dir=feat_dir, dataset="test")

with open(os.path.join(chunk_json_dir, "train_chunk.json"), 'w', encoding='utf-8') as fuo:
    json.dump(train_info, fp=fuo, indent=4, sort_keys=False, ensure_ascii=False)
with open(os.path.join(chunk_json_dir, "dev_chunk.json"), 'w', encoding='utf-8') as fuo:
    json.dump(dev_info, fp=fuo, indent=4, sort_keys=False, ensure_ascii=False)
with open(os.path.join(chunk_json_dir, "test_chunk.json"), 'w', encoding='utf-8') as fuo:
    json.dump(test_info, fp=fuo, indent=4, sort_keys=False, ensure_ascii=False)

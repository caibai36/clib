import kaldi_io
import os
import sys
import argparse
#################################################
# Adding kaldi tools to shell path,
import os
if not 'KALDI_ROOT' in os.environ:
    # Default! To change run python with 'export KALDI_ROOT=/some_dir python'
    os.environ['KALDI_ROOT'] = '/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi'
#################################################
def main():
    parser = argparse.ArgumentParser(description="bjava bnf feature pruned by segments with same number of frames of original feature file")
    parser.add_argument("--ori_dataset", type=str, default="data/test", help="the original dataset")
    parser.add_argument("--bnf_dataset", type=str, default="exp/bnf2asr/BNF_default_asr_general_BNF42_from_MFCC39/bnf/data_bnf/test_bnf", help="the bnf dataset")
    parser.add_argument("--out_feat_scp", type=str, default="exp/tmp_feats.scp")
    parser.add_argument("--out_feat_ark", type=str, default="exp/tmp_feats.ark")
    parser.add_argument("--window_size_ms", type=float, default=25, help="window size in ms")
    parser.add_argument("--frame_shift_ms", type=float, default=10, help="frame shift in ms")
    parser.add_argument("--round_precision", type=float, default=0.00001, help="round_precision for time to frame index conversion")

    args = parser.parse_args()
    
    window_size_ms = args.window_size_ms
    frame_shift_ms = args.frame_shift_ms
    round_precision = args.round_precision

    ori_dataset = args.ori_dataset
    bnf_dataset = args.bnf_dataset

    out_feat_scp = args.out_feat_scp
    out_feat_ark = args.out_feat_ark

    window_size= float(window_size_ms) / 1000
    frame_shift = float(frame_shift_ms) / 1000

    ori_feat_scp = os.path.join(ori_dataset, "feats.scp")
    ori_segments = os.path.join(ori_dataset, "segments")
    bnf_feat_scp = os.path.join(bnf_dataset, "feats.scp")

    start_times = {}
    end_times = {}
    with open(ori_segments, 'r') as f:
        for line in f:
            uttid, _, start, end = line.split()
            start_times[uttid] = start
            end_times[uttid] = end

    ori_uttid_list, ori_feats = read_feat_scp(ori_feat_scp)
    bnf_uttid_list, bnf_feats = read_feat_scp(bnf_feat_scp)

    feats = {}
    for uttid in bnf_uttid_list:
        start_index = time2index(start_times[uttid], window_size, frame_shift, round_precision)
        feats[uttid] = bnf_feats[uttid][start_index: start_index+len(ori_feats[uttid])]
    write_feat_scp(feats, out_feat_scp, out_feat_ark, uttid_list=ori_uttid_list)

def index2time(frame_index, window_size=0.025, frame_shift=0.01):
    """ Convert the frame index to the time.

    Parameters
    ----------
    frame_index: the index of the frame
    window_size: the size of length of each frame (default: 0.025)
    frame_shift: the window shift between successive frames (default: 0.01)

    Returns
    -------
    the time of the give frame
    
    Note
    ----
    The time of the frame_index 0 is half of the window_size.
    """
    time = window_size / 2 + frame_index * frame_shift
    return time

def time2index(time, window_size=0.025, frame_shift=0.01, precision=0.00001):
    """ Convert time to the nearest frame index.

    Parameters
    ----------
    time: the given time
    window_size: the size of length of each frame (default: 0.025)
    frame_shift: the window shift between successive frames (default: 0.01)
    precision: the round precision when cutting the float to interger (default: 0.00001) 

    Returns
    ------
    the index of the frame

    Note
    ----
    Find the nearest frame index to the given time
    """
    time=float(time)
    if (time < window_size / 2): return 0
    prev_index = int((time - window_size / 2) / frame_shift)

    prev_time = index2time(prev_index)
    next_time = index2time(prev_index + 1)
    average_time = (prev_time + next_time) / 2

    # 'floor' may have some precision issue, so make a precision bound.
    assert time - prev_time > -precision and next_time - time > -precision

    index = prev_index if time -average_time < 0 else prev_index + 1
    return index

def read_feat_scp(feat_scp):
    """ Read the feature file into a map from uttid string to feature numpy array from a given scp file.
    
    Example
    -------
    feat_scp_to_read="/project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/wsj/s0/data/test3utts/feats.scp"
    uttid_list, feats=read_feat_scp(feat_scp_to_read)

    Note
    ----
    Call . path.sh in kaldi example directory to include 'copy-feats' into executable search path.
    The function can be alternatively using:
    feats = {key: mat for key, mat in kaldi_io.read_mat_scp(feat_scp)}
    this implementation can deal with the bug of "The formats CM2, CM3 are not supported..."
    for reading the raw feature scp file.
    """
    fd=f"copy-feats scp:{feat_scp} ark,t:- |"
    uttid_list = []
    feats = {}
    for uttid, feature in kaldi_io.read_mat_ark(fd):
        uttid_list.append(uttid)
        feats[uttid] = feature

    #uttid_list= [uttid for uttid, feature in kaldi_io.read_mat_ark(fd)]
    #feats = {uttid: feature for uttid, feature in kaldi_io.read_mat_ark(fd)}
    return uttid_list, feats

def write_feat_scp(feats, feat_scp, feat_ark, uttid_list=None):
    """ Write the map from uttid string to numpy array (feats) to feat scp file with its feat ark file 
    with the order in given uttid_list.

    Example
    -------
    feat_scp_to_read="/project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/wsj/s0/data/test3utts/feats.scp"
    _, feats=read_feat_scp(feat_scp_to_read)
    feat_scp_to_write="exp/tmp_feat.scp"
    feat_ark_to_write="exp/tmp_feat.ark"
    # feat_ark_to_write=os.path.join(os.getcwd(), "exp/tmp_feat.ark")
    write_feat_scp(feats, feat_scp_to_write, feat_ark_to_write)
    
    feat_scp_to_read="/project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/wsj/s0/data/test3utts/feats.scp"
    uttid_list, feats=read_feat_scp(feat_scp_to_read)
    feat_scp_to_write="exp/tmp_feat.scp"
    feat_ark_to_write="exp/tmp_feat.ark"
    # feat_ark_to_write=os.path.join(os.getcwd(), "exp/tmp_feat.ark")
    write_feat_scp(feats, feat_scp_to_write, feat_ark_to_write, uttid_list)
    """
    scp_file = feat_scp
    ark_file = feat_ark
    ark_scp_output=f'ark:| copy-feats --compress=false ark:- ark,scp:{ark_file},{scp_file}'
    if uttid_list:
        with kaldi_io.open_or_fd(ark_scp_output,'wb') as f:
            for uttid in uttid_list:
                kaldi_io.write_mat(f, feats[uttid], key=uttid)
    else:
        with kaldi_io.open_or_fd(ark_scp_output,'wb') as f:
            for key, mat in feats.items():
                kaldi_io.write_mat(f, mat, key=key)



if __name__ == "__main__":
    sys.exit(main())

                
uttid="BABEL_OP3_402_14929_20141110_010718_D1_scripted"

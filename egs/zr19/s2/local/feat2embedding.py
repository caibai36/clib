import numpy as np
import os
import argparse
import sys
import kaldi_io # pip install kaldi_io

feat_scp_file = "data/test_en/feats.scp"
repr_dir = "eval/abx/embedding/exp/test/mfcc39"
# feat_scp_file = "data/test_en_vtln/feats.scp"
# repr_dir = "eval/abx/post/exp/dpgmm/mfcc39_vtln"

parser = argparse.ArgumentParser()
parser.add_argument("--feat", type=str, default=feat_scp_file, help="the scp file of feature")
parser.add_argument("--result", type=str, default=repr_dir, help="the result directory")
args = parser.parse_args()

_, first_feat = next(iter(kaldi_io.read_mat_scp(args.feat)))
feat_dim = first_feat.shape[1]
print(f"feat_dim: {feat_dim}")

if not os.path.exists(args.result):
    os.makedirs(args.result)

for uttid, feat in kaldi_io.read_mat_scp(args.feat):
    np.savetxt(os.path.join(args.result, uttid + ".txt"), feat, fmt='%.7f')

import os
import unittest
import clib.kaldi.kaldi_io as io

os.environ['KALDI_ROOT'] = "/project/nakamura-lab08/Work/bin-wu/share/tools/kaldi"

class TestKaldiIO(unittest.TestCase):

    def test_read_kaldi_scp_file(self):
        """
        Run the official code of kaldi with timit example
        and test on the first speaker of the training set.
        """
        feat_file = os.path.join(os.environ['KALDI_ROOT'], "egs/timit/s5/data/train/feats.scp")
        assert os.path.exists(feat_file), "Feature file {} of timit not found".format(feat_file)


        speaker, feature = next(io.read_mat_scp(feat_file))
        assert speaker == "FAEM0_SI1392"
        assert feature.shape == (474, 13)

        # for speaker, feature in io.read_mat_scp(feat_file):
        #     assert feature.shape[1] == 13  # 13-dim mfcc feature

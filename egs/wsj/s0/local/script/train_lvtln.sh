##############################################################
# File: tso_train_lvtln.sh
# Implemented at 22:30 on 08 January 2018 by bin-wu.
# Wrapper function to get the file - utt2warp - for doing vtln.
##############################################################

. ./cmd.sh

# the number of jobs for running the ubm
ubm_nj=10
# the number of jobs for warp factors estimation
train_vtln_nj=10
# the number of jobs for estimating the warp factors
get_vtln_nj=10

if [ $# != 3 ]; then
    echo "Usage: $0 <num-gauss-ubm> <data-dir> <exp-dir>"
    echo "e.g.: $0 400 data/train exp/train_vtln"
    echo "main options (for others, see top of script file)"
    echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
    echo "  --config <config-file>                           # config containing options"
    echo "  --stage <stage>                                  # stage to do partial re-run from."
    exit 1;
fi

# the number of Gaussians of training an ubm model
num_gauss_ubm=$1
# Directory contains data specific and language specific files from data preperation of corpora.
data=$2
# Directory contains the actual experiments and models, as well as logs.
dir=$3

# Compute the Voice activity detection (vad) decision.
lid/compute_vad_decision.sh $data $dir/make_vad_vtln $dir/vad_vtln

# Train the Universal Background Model (UBM).
sid/train_diag_ubm.sh --nj $ubm_nj --cmd "$train_cmd" $data $num_gauss_ubm $dir/diag_ubm_vtln

# Computes some things you will need in order to extract VTLN-warped features.
lid/train_lvtln_model.sh --mfcc-config conf/mfcc.conf --nj $train_vtln_nj --cmd "$train_cmd"  $data $dir/diag_ubm_vtln $dir/vtln

# It computes per-utterance warp-factors utt2warp.
lid/get_vtln_warps.sh --nj $get_vtln_nj --cmd "$train_cmd" $data $dir/vtln $dir/warps

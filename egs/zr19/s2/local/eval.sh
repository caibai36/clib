# Install abx by running ../zs19_docker/system/deploy/set_up_eval.sh
# Set the abx install directory as ../zs19_docker/system/eval
# cp -r ../zs19_docker/system/ .

if [ "$#" -eq 0 ]; then
    echo "usage: bash eval.sh --DIST <edit|kl|cos> --TOOL <ToolDir> --EMB <EmbeddingDir> --RES <ResultDir>"
    echo "./local/eval.sh --DIST 'edit' --TOOL '.' --EMB 'test' --RES 'result'"
    echo "example: bash local/eval.sh --DIST 'cos' --EMB 'eval/abx/post/exp/dpgmm/mfcc/test.vtln.mfcc/' --RES 'eval/abx/result/exp/dpgmm/mfcc/test.vtln.mfcc/'"
    echo "for zs19_docker/system/eval => ToolDir is zs19_docker"
    echo "for zr19/s0/system/eval => ToolDir is zs19/s0"
    echo "Default ToolDir is $PWD"
    exit
fi

DIST="edit"

# TOOL=$PWD
TOOL="/project/nakamura-lab08/Work/bin-wu/share/tools/abx_2019"
EMB=sandbox/english/test
RES=sandbox/result
. utils/parse_options.sh

echo "distance: $DIST"
echo "tool_dir: $TOOL"
echo "embedding: $EMB"
echo "result_dir: $RES"

mkdir -p $RES
# make a temp directory (automatically erased at exit)
TMP_DIR=$(mktemp -d)
EVAL_DIR="$TOOL/system/eval"
LANG=english
TASK_ACROSS="$TOOL/system/info_test/$LANG/by-context-across-speakers.abx"
BITRATE_FILELIST="$TOOL/system/info_test/$LANG/bitrate_filelist.txt"
ABX_SCORE_FILE=$RES/abx.txt
BITRATE_SCORE_FILE=$RES/bit.txt

# env created by $TOOL/system/deploy/set_up_eval.sh
source activate eval

# the directory of embedding
echo "Evaluating ABX discriminability"
mkdir -p $TMP_DIR/abx_npz_files

# create npz files out of all onehot embeddings for ABX evaluation
python $EVAL_DIR/scripts/make_abx_files.py \
       $EMB $TMP_DIR/abx_npz_files || exit 1

# Create .features file
python $EVAL_DIR/ABXpy/ABXpy/misc/any2h5features.py \
       $TMP_DIR/abx_npz_files $TMP_DIR/features.features

# Computing distances
if [ $DIST = "edit" ]; then
    abx-distance $TMP_DIR/features.features $TASK_ACROSS \
                 $TMP_DIR/distance_across -d "levenshtein"
elif [ $DIST = "kl" ]; then
    abx-distance $TMP_DIR/features.features $TASK_ACROSS \
                 $TMP_DIR/distance_across -d "dtw_kl"
elif [ $DIST = "cos" ]; then
    abx-distance $TMP_DIR/features.features $TASK_ACROSS \
                 $TMP_DIR/distance_across -n 1
else
    failure "$DIST not implemented: choose 'kl', 'cos' or 'edit'"
fi

# Calculating scores
abx-score $TASK_ACROSS $TMP_DIR/distance_across $TMP_DIR/score_across
# Collapsing results in readable format
abx-analyze $TMP_DIR/score_across $TASK_ACROSS $TMP_DIR/analyze_across
# Print average score
python $EVAL_DIR/scripts/meanABX.py $TMP_DIR/analyze_across across > $ABX_SCORE_FILE
echo ABX calculated using $DIST >> $ABX_SCORE_FILE

echo "Evaluating bitrate"
python $EVAL_DIR/scripts/bitrate.py $EMB/ \
       $BITRATE_FILELIST > $BITRATE_SCORE_FILE || exit 1

echo ""
cat $ABX_SCORE_FILE
cat $BITRATE_SCORE_FILE

echo ""
echo "ABX score is stored at $ABX_SCORE_FILE"
echo "Bitrate score is stored at $BITRATE_SCORE_FILE"

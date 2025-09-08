#!/bin/bash

# Statistical evaluation script for marmoset vocalization classification
# Usage: ./local/sandbox/ci_stat_test/run_stat_eval.sh

set -e

# Configuration
tolerance=0.1
ref1=/data/share/bin-wu/data/marmoset/vocalization/marmoset_mit/processed/audacity/audacity_labels/p1a1_toget.txt
ref2=/data/share/bin-wu/data/marmoset/vocalization/marmoset_mit/processed/audacity/audacity_labels/p1a2_toget.txt
n_bootstrap=2000
confidence=0.95

# Create results directory in exp/sandbox/stat
mkdir -p exp/sandbox/stat
mkdir -p exp/sandbox/stat/results
mkdir -p exp/sandbox/stat/comparisons

echo "Starting Statistical Evaluation Pipeline"
echo "======================================="
echo "Results will be saved to: exp/sandbox/stat/"
echo ""

# MIT CNN
echo "Evaluating MIT CNN..."
hypo1_mit=/work01/home/bin-wu/workspace/projects/clib/egs/riken/mit_cnn_s0/exp/sys/mit_data/mit_data0/mit_cnn_72-run0/bs25lr0.0003evalinterval2000avgpredwin5/eval/test_pred1_p1a1_toget_cutoff0.6.txt
hypo2_mit=/work01/home/bin-wu/workspace/projects/clib/egs/riken/mit_cnn_s0/exp/sys/mit_data/mit_data0/mit_cnn_72-run0/bs25lr0.0003evalinterval2000avgpredwin5/eval/test_pred2_p1a2_toget_cutoff0.6.txt

python local/sandbox/ci_stat_test/mit_cnn_eval_acc_spk_with_stats.py \
    --hypo_files $hypo1_mit $hypo2_mit \
    --ref_files $ref1 $ref2 \
    --model_name "MIT_CNN" \
    --save_results exp/sandbox/stat/results/mit_cnn_results.pkl \
    --n_bootstrap $n_bootstrap \
    --confidence $confidence

echo "MIT CNN evaluation completed."
echo ""

# Our CNN
echo "Evaluating Our CNN..."
hypo1_our=exp/run/mit_data/mit_data0/mit_cnn_72-run0/bs25lr0.0003evalinterval1avgpredwin5/eval/test_pred1_p1a1_toget_cutoff0.6.txt
hypo2_our=exp/run/mit_data/mit_data0/mit_cnn_72-run0/bs25lr0.0003evalinterval1avgpredwin5/eval/test_pred2_p1a2_toget_cutoff0.6.txt

python local/sandbox/ci_stat_test/mit_cnn_eval_acc_spk_with_stats.py \
    --hypo_files $hypo1_our $hypo2_our \
    --ref_files $ref1 $ref2 \
    --model_name "Our_CNN" \
    --save_results exp/sandbox/stat/results/our_cnn_results.pkl \
    --n_bootstrap $n_bootstrap \
    --confidence $confidence

echo "Our CNN evaluation completed."
echo ""

# ViT
echo "Evaluating ViT..."
hypo1_vit=exp/run/mit_data/mit_data0/mit_vit_72-run0/bs25lr0.0003evalinterval1avgpredwin5_dim384depth6heads6mlpdim1536patch16dropout0.1sharedim1024/eval/test_pred1_p1a1_toget_cutoff0.65.txt
hypo2_vit=exp/run/mit_data/mit_data0/mit_vit_72-run0/bs25lr0.0003evalinterval1avgpredwin5_dim384depth6heads6mlpdim1536patch16dropout0.1sharedim1024/eval/test_pred2_p1a2_toget_cutoff0.65.txt

python local/sandbox/ci_stat_test/mit_cnn_eval_acc_spk_with_stats.py \
    --hypo_files $hypo1_vit $hypo2_vit \
    --ref_files $ref1 $ref2 \
    --model_name "ViT" \
    --save_results exp/sandbox/stat/results/vit_results.pkl \
    --n_bootstrap $n_bootstrap \
    --confidence $confidence

echo "ViT evaluation completed."
echo ""

# MAE ViT
echo "Evaluating MAE ViT..."
hypo1_mae=exp/run_mae_vit/mit_data/mit_data0/mit_vit_72-run0/bs25lr0.0003evalinterval1avgpredwin5_dim768depth12heads12mlpdim3072patch16dropout0.1sharedim1024maemaepretrainedmodel_epoch_400_base_48days.pttrainlayers6/eval/test_pred1_p1a1_toget_cutoff0.6.txt
hypo2_mae=exp/run_mae_vit/mit_data/mit_data0/mit_vit_72-run0/bs25lr0.0003evalinterval1avgpredwin5_dim768depth12heads12mlpdim3072patch16dropout0.1sharedim1024maemaepretrainedmodel_epoch_400_base_48days.pttrainlayers6/eval/test_pred2_p1a2_toget_cutoff0.6.txt

python local/sandbox/ci_stat_test/mit_cnn_eval_acc_spk_with_stats.py \
    --hypo_files $hypo1_mae $hypo2_mae \
    --ref_files $ref1 $ref2 \
    --model_name "MAE_ViT" \
    --save_results exp/sandbox/stat/results/mae_vit_results.pkl \
    --n_bootstrap $n_bootstrap \
    --confidence $confidence

echo "MAE ViT evaluation completed."
echo ""

echo "======================================="
echo "Individual Model Evaluations Completed"
echo "======================================="
echo ""

echo "======================================="
echo "Statistical Comparisons"
echo "======================================="

# Compare MAE ViT vs Our CNN (as requested by reviewer)
echo "Comparing MAE ViT vs Our CNN..."
python local/sandbox/ci_stat_test/compare_models.py \
    --model1_results exp/sandbox/stat/results/mae_vit_results.pkl \
    --model2_results exp/sandbox/stat/results/our_cnn_results.pkl \
    --output_dir exp/sandbox/stat/comparisons/mae_vit_vs_our_cnn \
    --n_bootstrap 10000

echo "MAE ViT vs Our CNN comparison completed."
echo ""

# Compare MAE ViT vs ViT
echo "Comparing MAE ViT vs ViT..."
python local/sandbox/ci_stat_test/compare_models.py \
    --model1_results exp/sandbox/stat/results/mae_vit_results.pkl \
    --model2_results exp/sandbox/stat/results/vit_results.pkl \
    --output_dir exp/sandbox/stat/comparisons/mae_vit_vs_vit \
    --n_bootstrap 10000

echo "MAE ViT vs ViT comparison completed."
echo ""

# Compare ViT vs Our CNN
echo "Comparing ViT vs Our CNN..."
python local/sandbox/ci_stat_test/compare_models.py \
    --model1_results exp/sandbox/stat/results/vit_results.pkl \
    --model2_results exp/sandbox/stat/results/our_cnn_results.pkl \
    --output_dir exp/sandbox/stat/comparisons/vit_vs_our_cnn \
    --n_bootstrap 10000

echo "ViT vs Our CNN comparison completed."
echo ""

# Compare Our CNN vs MIT CNN
echo "Comparing Our CNN vs MIT CNN..."
python local/sandbox/ci_stat_test/compare_models.py \
    --model1_results exp/sandbox/stat/results/our_cnn_results.pkl \
    --model2_results exp/sandbox/stat/results/mit_cnn_results.pkl \
    --output_dir exp/sandbox/stat/comparisons/our_cnn_vs_mit_cnn \
    --n_bootstrap 10000

echo "Our CNN vs MIT CNN comparison completed."
echo ""

echo "======================================="
echo "All Statistical Evaluations Completed!"
echo "======================================="

# Summary of results
echo ""
echo "Results Directory Structure:"
echo "exp/sandbox/stat/"
echo "├── results/"
echo "│   ├── mit_cnn_results.pkl"
echo "│   ├── our_cnn_results.pkl"
echo "│   ├── vit_results.pkl"
echo "│   └── mae_vit_results.pkl"
echo "└── comparisons/"
echo "    ├── mae_vit_vs_our_cnn/"
echo "    ├── mae_vit_vs_vit/"
echo "    ├── vit_vs_our_cnn/"
echo "    └── our_cnn_vs_mit_cnn/"

echo ""
echo "Key Files Generated:"
echo "- Individual model results with confidence intervals (.pkl files)"
echo "- Statistical comparison reports (console output saved in each comparison dir)"
echo "- Visualization plots:"
echo "  * confidence_intervals_comparison.png"
echo "  * bootstrap_distributions.png"

echo ""
echo "To view detailed comparison results, check:"
echo "exp/sandbox/stat/comparisons/mae_vit_vs_our_cnn/"
echo ""
echo "Statistical tests performed:"
echo "- Bootstrap confidence intervals (95%)"
echo "- McNemar's test for paired comparisons"
echo "- Permutation tests for metric differences"
echo "- Visualization of confidence intervals and distributions"

echo ""
echo "MAIN RESULT (for reviewer): MAE ViT vs Our CNN comparison"
echo "Location: exp/sandbox/stat/comparisons/mae_vit_vs_our_cnn/"

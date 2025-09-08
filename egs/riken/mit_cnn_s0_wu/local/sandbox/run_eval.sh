ref1=/data/share/bin-wu/data/marmoset/vocalization/marmoset_mit/processed/audacity/audacity_labels/p1a1_toget.txt
ref2=/data/share/bin-wu/data/marmoset/vocalization/marmoset_mit/processed/audacity/audacity_labels/p1a2_toget.txt
tolerance=0.1

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. local/scripts/parse_options.sh || exit 1

# MIT tensorflow CNN
# Fraction correctly classified: Noise:0.9963, Call:0.7212, Total:0.9899
# Recall:0.7212, Precision:0.8227, F1-score:0.7686
hypo1=/work01/home/bin-wu/workspace/projects/clib/egs/riken/mit_cnn_s0/exp/sys/mit_data/mit_data0/mit_cnn_72-run0/bs25lr0.0003evalinterval2000avgpredwin5/eval/test_pred1_p1a1_toget_cutoff0.6.txt
hypo2=/work01/home/bin-wu/workspace/projects/clib/egs/riken/mit_cnn_s0/exp/sys/mit_data/mit_data0/mit_cnn_72-run0/bs25lr0.0003evalinterval2000avgpredwin5/eval/test_pred2_p1a2_toget_cutoff0.6.txt
echo -e "\nMIT tensorflow CNN"
# python local/mit_cnn_eval_acc.py --hypo_files $hypo1 $hypo2 --ref_files $ref1 $ref2
python local/mit_cnn_eval_acc_spk.py --hypo_files $hypo1 $hypo2 --ref_files $ref1 $ref2
# python ../riken_cnn_s0/local/sandbox/riken_cnn_eval_acc_boundaries.py --hypo_file $hypo1 --ref_file $ref1
# python ../riken_cnn_s0/local/sandbox/riken_cnn_eval_acc_boundaries.py --hypo_file $hypo2 --ref_file $ref2
python ../riken_cnn_s0/local/sandbox/riken_cnn_eval_acc_boundaries.py --hypo_files $hypo1 $hypo2 --ref_files $ref1 $ref2 --tolerance $tolerance

# Our CNN
# Fraction correctly classified: Noise:0.9962, Call:0.7575, Total:0.9907
# Recall:0.7575, Precision:0.8257, F1-score:0.7901
hypo1=/work01/home/bin-wu/workspace/projects/clib/egs/riken/mit_cnn_s0_wu/exp/run/mit_data/mit_data0/mit_cnn_72-run0/bs25lr0.0003evalinterval1avgpredwin5/eval/test_pred1_p1a1_toget_cutoff0.6.txt
hypo2=/work01/home/bin-wu/workspace/projects/clib/egs/riken/mit_cnn_s0_wu/exp/run/mit_data/mit_data0/mit_cnn_72-run0/bs25lr0.0003evalinterval1avgpredwin5/eval/test_pred2_p1a2_toget_cutoff0.6.txt
echo -e "\nOur CNN"
python local/mit_cnn_eval_acc_spk.py --hypo_files $hypo1 $hypo2 --ref_files $ref1 $ref2
# python ../riken_cnn_s0/local/sandbox/riken_cnn_eval_acc_boundaries.py --hypo_file $hypo1 --ref_file $ref1
# python ../riken_cnn_s0/local/sandbox/riken_cnn_eval_acc_boundaries.py --hypo_file $hypo2 --ref_file $ref2
python ../riken_cnn_s0/local/sandbox/riken_cnn_eval_acc_boundaries.py --hypo_files $hypo1 $hypo2 --ref_files $ref1 $ref2 --tolerance $tolerance

# ViT
# Fraction correctly classified: Noise:0.9964, Call:0.7576, Total:0.9909
# Recall:0.7576, Precision:0.8341, F1-score:0.7940
hypo1=/work01/home/bin-wu/workspace/projects/clib/egs/riken/mit_cnn_s0_wu/exp/run/mit_data/mit_data0/mit_vit_72-run0/bs25lr0.0003evalinterval1avgpredwin5_dim384depth6heads6mlpdim1536patch16dropout0.1sharedim1024/eval/test_pred1_p1a1_toget_cutoff0.65.txt
hypo2=/work01/home/bin-wu/workspace/projects/clib/egs/riken/mit_cnn_s0_wu/exp/run/mit_data/mit_data0/mit_vit_72-run0/bs25lr0.0003evalinterval1avgpredwin5_dim384depth6heads6mlpdim1536patch16dropout0.1sharedim1024/eval/test_pred2_p1a2_toget_cutoff0.65.txt
echo -e "\nOur ViT"
python local/mit_cnn_eval_acc_spk.py --hypo_files $hypo1 $hypo2 --ref_files $ref1 $ref2
# python ../riken_cnn_s0/local/riken_cnn_eval_acc.py --hypo_files $hypo1 $hypo2 --ref_files $ref1 $ref2 --label2id_yaml ../riken_cnn_s0/conf/dict/label2id_marmoset.yaml
# python ../riken_cnn_s0/local/sandbox/riken_cnn_eval_acc_boundaries.py --hypo_file $hypo1 --ref_file $ref1
# python ../riken_cnn_s0/local/sandbox/riken_cnn_eval_acc_boundaries.py --hypo_file $hypo2 --ref_file $ref2
python ../riken_cnn_s0/local/sandbox/riken_cnn_eval_acc_boundaries.py --hypo_files $hypo1 $hypo2 --ref_files $ref1 $ref2 --tolerance $tolerance

# MAE ViT
# Fraction correctly classified: Noise:0.9964, Call:0.7576, Total:0.9909
# Recall:0.7576, Precision:0.8341, F1-score:0.7940
hypo1=exp/run_mae_vit/mit_data/mit_data0/mit_vit_72-run0/bs25lr0.0003evalinterval1avgpredwin5_dim768depth12heads12mlpdim3072patch16dropout0.1sharedim1024maemaepretrainedmodel_epoch_400_base_48days.pttrainlayers6/eval/test_pred1_p1a1_toget_cutoff0.6.txt
hypo2=exp/run_mae_vit/mit_data/mit_data0/mit_vit_72-run0/bs25lr0.0003evalinterval1avgpredwin5_dim768depth12heads12mlpdim3072patch16dropout0.1sharedim1024maemaepretrainedmodel_epoch_400_base_48days.pttrainlayers6/eval/test_pred2_p1a2_toget_cutoff0.6.txt
echo -e "\nOur MAE ViT"
python local/mit_cnn_eval_acc_spk.py --hypo_files $hypo1 $hypo2 --ref_files $ref1 $ref2
# python ../riken_cnn_s0/local/riken_cnn_eval_acc.py --hypo_files $hypo1 $hypo2 --ref_files $ref1 $ref2 --label2id_yaml ../riken_cnn_s0/conf/dict/label2id_marmoset.yaml
# python ../riken_cnn_s0/local/sandbox/riken_cnn_eval_acc_boundaries.py --hypo_file $hypo1 --ref_file $ref1
# python ../riken_cnn_s0/local/sandbox/riken_cnn_eval_acc_boundaries.py --hypo_file $hypo2 --ref_file $ref2
python ../riken_cnn_s0/local/sandbox/riken_cnn_eval_acc_boundaries.py --hypo_files $hypo1 $hypo2 --ref_files $ref1 $ref2 --tolerance $tolerance

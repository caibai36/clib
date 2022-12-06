# settings in general
tag=yonden_kana_baseline
feat_name=MEL80

train_data=$PWD/data/train_data3_5_7to17_baseline # /project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/wsj/s0/data/train_si84
dev_data=$PWD/data/dev_data4_baseline # /project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/wsj/s0/data/test_dev93
test_data=$PWD/data/test_data6_baseline # /project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/wsj/s0/data/test_eval92
train_set=all_data3-28_baseline # name of the training data set for get the token vocabulary

feat_train=$PWD/data/train_data3_5_7to17_baseline_mel80/feats.scp
feat_dev=$PWD/data/dev_data4_baseline_mel80/feats.scp
feat_test=$PWD/data/test_data6_baseline_mel80/feats.scp

./local/sandbox/asr_general/run_all_full.sh --stage 6 --run_kana_asr true --tag ${tag} --feat_name ${feat_name} \
					    --train_data ${train_data} --dev_data ${dev_data} --test_data ${test_data} --train_set ${train_set} \
					    --feat_train ${feat_train} --feat_dev ${feat_dev} --feat_test ${feat_test}

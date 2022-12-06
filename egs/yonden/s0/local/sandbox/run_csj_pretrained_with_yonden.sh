for feat in mfcc39 mel80; do
    tag=default_csj_pretrained_with_yonden

    # datasets="test_data6 test_data6_ampnorm test_data6_spkinfo test_data6_spkinfo_daily test_data36 test_data36_ampnorm test_data36_spkinfo test_data36_spkinfo_daily"
    datasets="test_data6 test_data36"
    pretrained_dict=conf/csj/dict/train_units.txt
    pretrain_model=conf/csj/model/$feat/best_model.mdl

    train="train_data3-36_remove_4_6_28_36"
    dev="dev_data4_28"
    for dataset in $datasets; do
	./run.sh --stage 5 --feat ${feat} --tag ${tag} \
		 --train ${train}_${feat} --dev ${dev}_${feat} --test ${dataset}_${feat} \
		 --pretrained_dict ${pretrained_dict} --pretrained_model ${pretrain_model}
    done
done

for feat in mfcc39 mel80; do
    tag=default_csj_pretrained_with_yonden

    # datasets="test_data6 test_data6_ampnorm test_data6_spkinfo test_data6_spkinfo_daily test_data36 test_data36_ampnorm test_data36_spkinfo test_data36_spkinfo_daily"
    datasets="test_data6_ampnorm test_data36_ampnorm"
    pretrained_dict=conf/csj/dict/train_units.txt
    pretrain_model=conf/csj/model/$feat/best_model.mdl

    train="train_data3-36_remove_4_6_28_36_ampnorm"
    dev="dev_data4_28_ampnorm"
    for dataset in $datasets; do
	./run.sh --stage 5 --feat ${feat} --tag ${tag} \
		 --train ${train}_${feat} --dev ${dev}_${feat} --test ${dataset}_${feat} \
		 --pretrained_dict ${pretrained_dict} --pretrained_model ${pretrain_model}
    done
done

# for feat in mfcc39 mel80; do
#     tag=default_csj_pretrained_with_yonden

#     # datasets="test_data6 test_data6_ampnorm test_data6_spkinfo test_data6_spkinfo_daily test_data36 test_data36_ampnorm test_data36_spkinfo test_data36_spkinfo_daily"
#     datasets="test_data6_spkinfo test_data36_spkinfo"
#     pretrained_dict=conf/csj/dict/train_units.txt
#     pretrain_model=conf/csj/model/$feat/best_model.mdl

#     train="train_data3-36_remove_4_6_28_36_spkinfo"
#     dev="dev_data4_28_spkinfo"
#     for dataset in $datasets; do
# 	./run.sh --stage 5 --feat ${feat} --tag ${tag} \
# 		 --train ${train}_${feat} --dev ${dev}_${feat} --test ${dataset}_${feat} \
# 		 --pretrained_dict ${pretrained_dict} --pretrained_model ${pretrain_model}
#     done
# done

# for feat in mfcc39 mel80; do
#     tag=default_csj_pretrained_with_yonden

#     # datasets="test_data6 test_data6_ampnorm test_data6_spkinfo test_data6_spkinfo_daily test_data36 test_data36_ampnorm test_data36_spkinfo test_data36_spkinfo_daily"
#     datasets="test_data6_spkinfo_daily test_data36_spkinfo_daily"
#     pretrained_dict=conf/csj/dict/train_units.txt
#     pretrain_model=conf/csj/model/$feat/best_model.mdl

#     train="train_data3-36_remove_4_6_28_36_spkinfo_daily"
#     dev="dev_data4_28_spkinfo_daily"
#     for dataset in $datasets; do
# 	./run.sh --stage 5 --feat ${feat} --tag ${tag} \
# 		 --train ${train}_${feat} --dev ${dev}_${feat} --test ${dataset}_${feat} \
# 		 --pretrained_dict ${pretrained_dict} --pretrained_model ${pretrain_model}
#     done
# done

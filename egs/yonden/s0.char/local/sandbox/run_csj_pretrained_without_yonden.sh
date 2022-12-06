for feat in mfcc39 mel80; do
    # datasets="test_data6 test_data6_ampnorm test_data6_spkinfo test_data6_spkinfo_daily test_data36 test_data36_ampnorm test_data36_spkinfo test_data36_spkinfo_daily"
    datasets="test_data6 test_data6_ampnorm test_data36 test_data36_ampnorm"
    pretrained_dict=conf/csj/dict/train_units.txt
    pretrain_model=conf/csj/model/$feat/best_model.mdl

    for dataset in $datasets; do
	./run.sh --stage 5 --train "" --dev "" --test ${dataset}_${feat} --feat ${feat} \
		 --pretrained_dict ${pretrained_dict} --pretrained_model ${pretrain_model}
    done
done

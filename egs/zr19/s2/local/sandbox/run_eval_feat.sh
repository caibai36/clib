stage=5
if [ $stage -le 5 ]; then
    echo "---------------------------------------------------"
    echo "Evaluate the embedding by ABX test"
    echo "---------------------------------------------------"

    for path in mfcc39 mfcc39_vtln mfcc40 mfcc80; do
	date
	# path=exp/feat/mfcc39
	task=$(basename $path)
	root=exp/feat/${task}

	abx_embedding=eval/abx/embedding/${root} # CHECK ME
	abx_result_cos=eval/abx/result/${root}/cos
	abx_result_kl=eval/abx/result/${root}/kl
	abx_result_edit=eval/abx/result/${root}/edit

	# /project/nakamura-lab08/Work/bin-wu/share/tools/abx_2019/system/deploy/set_up_eval.sh
	source activate eval
	./local/eval.sh --DIST 'cos' --EMB $abx_embedding --RES $abx_result_cos
	./local/eval.sh --DIST 'kl' --EMB $abx_embedding --RES $abx_result_kl
	./local/eval.sh --DIST 'edit' --EMB $abx_embedding --RES $abx_result_edit
	date
    done
fi

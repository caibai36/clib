tag=mfcc
ref=../s1/exp/tmp/timit/default/EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp-run0/mfcc39_batchsize32_cutoff1600_labelsmoothing0.05_lr0.001_gradclip5_factor0.5_patience3/eval/beamsize10/ref_word.txt
hyp=../s1/exp/tmp/timit/default/EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp-run0/mfcc39_batchsize32_cutoff1600_labelsmoothing0.05_lr0.001_gradclip5_factor0.5_patience3/eval/beamsize10/hypo_word.txt
./local/sclite_score.sh  --tag $tag --ref $ref --hyp $hyp

tag=mfcc_dpgmm
ref=exp/tmp/timit/default/EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp-run0/concat_mfcc39_dpgmm_embedding_seed123_K98_batchsize32_cutoff1600_labelsmoothing0.05_lr0.001_gradclip5_factor0.5_patience3/eval/beamsize10/ref_word.txt
hyp=exp/tmp/timit/default/EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp-run0/concat_mfcc39_dpgmm_embedding_seed123_K98_batchsize32_cutoff1600_labelsmoothing0.05_lr0.001_gradclip5_factor0.5_patience3/eval/beamsize10/hypo_word.txt
./local/sclite_score.sh  --tag $tag --ref $ref --hyp $hyp

tag=mfcc_hybrid
ref=exp/tmp_hybrid_asr_general/timit/default/EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp-run0/concat_MFCC39_HybridMSEseed123nseed123K98l8r8_bs256_hd512_nl5_ne20_batchsize32_cutoff1600_labelsmoothing0.05_lr0.001_gradclip5_factor0.5_patience3/eval/beamsize10/ref_word.txt
hyp=exp/tmp_hybrid_asr_general/timit/default/EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp-run0/concat_MFCC39_HybridMSEseed123nseed123K98l8r8_bs256_hd512_nl5_ne20_batchsize32_cutoff1600_labelsmoothing0.05_lr0.001_gradclip5_factor0.5_patience3/eval/beamsize10/hypo_word.txt
./local/sclite_score.sh  --tag $tag --ref $ref --hyp $hyp

tag=mfcc_bnf
ref=exp/tmp/timit/default/EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp-run0/concat_MFCC39_BNF42_batchsize32_cutoff1600_labelsmoothing0.05_lr0.001_gradclip5_factor0.5_patience3/eval/beamsize10/ref_word.txt
hyp=exp/tmp/timit/default/EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp-run0/concat_MFCC39_BNF42_batchsize32_cutoff1600_labelsmoothing0.05_lr0.001_gradclip5_factor0.5_patience3/eval/beamsize10/hypo_word.txt
./local/sclite_score.sh  --tag $tag --ref $ref --hyp $hyp

tag=bnf
ref=exp/tmp/timit/default/EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp-run0/single_BNF42_batchsize32_cutoff1600_labelsmoothing0.05_lr0.001_gradclip5_factor0.5_patience3/eval/beamsize10/ref_word.txt
hyp=exp/tmp/timit/default/EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp-run0/single_BNF42_batchsize32_cutoff1600_labelsmoothing0.05_lr0.001_gradclip5_factor0.5_patience3/eval/beamsize10/hypo_word.txt
./local/sclite_score.sh  --tag $tag --ref $ref --hyp $hyp

tag=mfcc_hybrid_bnf
ref=exp/tmp_asr_general_rnnbnf_aseed/aseed2021.gpuauto.bs32.cf1600.ls0.05.lr0.001.ne70.gp5.factor0.5.pat3.si1.searchbeam.mt250.beamsize.10/timit/default/EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp-run0/default_asr_general_concat_mfccrnn_bnf_batchsize32_cutoff1600_labelsmoothing0.05_lr0.001_gradclip5_factor0.5_patience3/eval/beamsize10/ref_word.txt
hyp=exp/tmp_asr_general_rnnbnf_aseed/aseed2021.gpuauto.bs32.cf1600.ls0.05.lr0.001.ne70.gp5.factor0.5.pat3.si1.searchbeam.mt250.beamsize.10/timit/default/EncRNNDecRNNAtt-enc3_bi256_ds3_drop-dec1_h512_do0.25-att_mlp-run0/default_asr_general_concat_mfccrnn_bnf_batchsize32_cutoff1600_labelsmoothing0.05_lr0.001_gradclip5_factor0.5_patience3/eval/beamsize10/hypo_word.txt
./local/sclite_score.sh  --tag $tag --ref $ref --hyp $hyp

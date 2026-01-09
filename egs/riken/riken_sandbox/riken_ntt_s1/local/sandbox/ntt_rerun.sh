echo "Launching job for kk..."
./local/sandbox/run_ma.sh \
    --stage 0 \
    --gpu 1 \
    --dataset ntt_kk \
    --info-csv /data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/processed/metas_speaker_force_alignment_phone/meta_kk_speaker.csv \
    --exp-dir /data02/share/bin-wu/exp/sandbox/ntt/kk/filter_begin0_end-2 \
    --fig-dir /data02/share/bin-wu/exp/sandbox/ntt/kk/filter_begin0_end-2/figures \
    --n-stages 2 \
    --whole-begin 0 \
    --whole-end -2 |& tee logs/run_ntt_kk.log

echo "Launching job for ma..."
./local/sandbox/run_ma.sh \
    --stage 0 \
    --gpu 2 \
    --dataset ntt_ma \
    --info-csv /data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/processed/metas_speaker_force_alignment_phone/meta_ma_speaker.csv \
    --exp-dir /data02/share/bin-wu/exp/sandbox/ntt/ma/filter_begin0_end-2 \
    --fig-dir /data02/share/bin-wu/exp/sandbox/ntt/ma/filter_begin0_end-2/figures \
    --n-stages 3 \
    --whole-begin 0 \
    --whole-end -2 |& tee logs/run_ntt_ma.log

echo "Launching job for sa..."
./local/sandbox/run_ma.sh \
    --stage 0 \
    --gpu 4 \
    --dataset ntt_sa \
    --info-csv /data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/processed/metas_speaker_force_alignment_phone/meta_sa_speaker.csv \
    --exp-dir /data02/share/bin-wu/exp/sandbox/ntt/sa/filter_begin0_end-2 \
    --fig-dir /data02/share/bin-wu/exp/sandbox/ntt/sa/filter_begin0_end-2/figures \
    --n-stages 3 \
    --whole-begin 0 \
    --whole-end -2 |& tee logs/run_ntt_sa.log

echo "Launching job for mk..."
./local/sandbox/run_ma.sh \
    --stage 0 \
    --gpu 3 \
    --dataset ntt_mk \
    --info-csv /data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/processed/metas_speaker_force_alignment_phone/meta_mk_speaker.csv \
    --exp-dir /data02/share/bin-wu/exp/sandbox/ntt/mk/filter_begin0_end-2 \
    --fig-dir /data02/share/bin-wu/exp/sandbox/ntt/mk/filter_begin0_end-2/figures \
    --n-stages 2 \
    --whole-begin 0 \
    --whole-end -2 |& tee logs/run_ntt_mk.log

echo "Launching job for sk..."
./local/sandbox/run_ma.sh \
    --stage 0 \
    --gpu 1 \
    --dataset ntt_sk \
    --info-csv /data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/processed/metas_speaker_force_alignment_phone/meta_sk_speaker.csv \
    --exp-dir /data02/share/bin-wu/exp/sandbox/ntt/sk/filter_begin0_end-2 \
    --fig-dir /data02/share/bin-wu/exp/sandbox/ntt/sk/filter_begin0_end-2/figures \
    --n-stages 3 \
    --whole-begin 0 \
    --whole-end -2 |& tee logs/run_ntt_sk.log

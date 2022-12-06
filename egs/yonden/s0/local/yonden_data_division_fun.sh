yonden_baseline() {
    # $1 is directory
    # $2 is path to dataset
    # $3 is dataset filter
    # Usage:
    # yonden_baseline "$dir" "$dataset" "$filter"
    mkdir -p $1/$2
    field="seg_wav"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/wav.scp
    speaker="recid"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df['$speaker'].to_csv(sep=' ',header=None), end='')" |  sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/utt2spk
    speaker="recid"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df.groupby('$speaker').apply(lambda x: ' '.join(x['id'].unique())).to_csv(sep=' ',header=None), end='')" |  sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/spk2utt
    field="kanji"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/text
    field="text.am.pos"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/text.am.pos
    field="kana"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/kana
    field="phone"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/phone
}

yonden_baseline_ampnorm() {
    # $1 is directory
    # $2 is path to dataset
    # $3 is dataset filter
    # Usage:
    # yonden_baseline_ampnorm "$dir" "$dataset" "$filter"
    mkdir -p $1/$2
    field="seg_wav_amp_norm"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/wav.scp
    speaker="recid"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df['$speaker'].to_csv(sep=' ',header=None), end='')" |  sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/utt2spk
    speaker="recid"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df.groupby('$speaker').apply(lambda x: ' '.join(x['id'].unique())).to_csv(sep=' ',header=None), end='')" |  sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/spk2utt
    field="kanji"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/text
    field="text.am.pos"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/text.am.pos
    field="kana"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/kana
    field="phone"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/phone
}

yonden_baseline_spkinfo() {
    # $1 is directory
    # $2 is path to dataset
    # $3 is dataset filter
    # Usage:
    # yonden_baseline_spkinfo_daily "$dir" "$dataset" "$filter"
    # e.g., uttid and speaker as 指揮者ヒラオカ_210920_1358_平岡班_無線機_00010_0001009_0001151 指揮者ヒラオカ # may be it need more data to achieve speaker normalization effect
    mkdir -p $1/$2
    field="seg_wav"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df.set_index(df['speaker'] + '_' + df['id'])['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/wav.scp
    speaker="speaker"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; df['speaker_new']=df['speaker']+'_'+df['work_date']+'_'+df['work_start_time']+'_'+df['group']; print(df.set_index(df['speaker'] + '_' + df['id'])['speaker'].to_csv(sep=' ',header=None), end='')" |  sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/utt2spk
    speaker="speaker"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; df['speaker_new']=df['speaker']+'_'+df['work_date']+'_'+df['work_start_time']+'_'+df['group']; print(df.groupby('speaker').apply(lambda x: ' '.join((x['speaker'] + '_' + x['id']).unique())).to_csv(sep=' ',header=None), end='')" |  sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/spk2utt
    field="kanji"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df.set_index(df['speaker'] + '_' + df['id'])['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/text
    field="text.am.pos"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df.set_index(df['speaker'] + '_' + df['id'])['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/text.am.pos
    field="kana"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df.set_index(df['speaker'] + '_' + df['id'])['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/kana
    field="phone"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df.set_index(df['speaker'] + '_' + df['id'])['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/phone
}

yonden_baseline_ampnorm_spkinfo() {
    # $1 is directory
    # $2 is path to dataset
    # $3 is dataset filter
    # Usage:
    # yonden_baseline_amp_norm_spkinfo "$dir" "$dataset" "$filter"
    # e.g., uttid and speaker as 指揮者ヒラオカ_210920_1358_平岡班_無線機_00010_0001009_0001151 指揮者ヒラオカ # may be it need more data to achieve speaker normalization effect
    mkdir -p $1/$2
    field="seg_wav_amp_norm"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df.set_index(df['speaker'] + '_' + df['id'])['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/wav.scp
    speaker="speaker"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; df['speaker_new']=df['speaker']+'_'+df['work_date']+'_'+df['work_start_time']+'_'+df['group']; print(df.set_index(df['speaker'] + '_' + df['id'])['speaker'].to_csv(sep=' ',header=None), end='')" |  sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/utt2spk
    speaker="speaker"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; df['speaker_new']=df['speaker']+'_'+df['work_date']+'_'+df['work_start_time']+'_'+df['group']; print(df.groupby('speaker').apply(lambda x: ' '.join((x['speaker'] + '_' + x['id']).unique())).to_csv(sep=' ',header=None), end='')" |  sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/spk2utt
    field="kanji"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df.set_index(df['speaker'] + '_' + df['id'])['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/text
    field="text.am.pos"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df.set_index(df['speaker'] + '_' + df['id'])['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/text.am.pos
    field="kana"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df.set_index(df['speaker'] + '_' + df['id'])['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/kana
    field="phone"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df.set_index(df['speaker'] + '_' + df['id'])['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/phone
}

yonden_baseline_spkinfo_daily() {
    # $1 is directory
    # $2 is path to dataset
    # $3 is dataset filter
    # Usage:
    # yonden_baseline_spkinfo_daily "$dir" "$dataset" "$filter"
    # e.g., uttid and speaker as 指揮者ヒラオカ_210920_1358_平岡班_無線機_00010_0001009_0001151 指揮者ヒラオカ_210920_1358_平岡班 # may be every utteraces vary by distance of mircophone
    mkdir -p $1/$2
    field="seg_wav"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df.set_index(df['speaker'] + '_' + df['id'])['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/wav.scp
    speaker="speaker"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; df['speaker_new']=df['speaker']+'_'+df['work_date']+'_'+df['work_start_time']+'_'+df['group']; print(df.set_index(df['speaker'] + '_' + df['id'])['speaker_new'].to_csv(sep=' ',header=None), end='')" |  sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/utt2spk
    speaker="speaker"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; df['speaker_new']=df['speaker']+'_'+df['work_date']+'_'+df['work_start_time']+'_'+df['group']; print(df.groupby('speaker_new').apply(lambda x: ' '.join((x['speaker'] + '_' + x['id']).unique())).to_csv(sep=' ',header=None), end='')" |  sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/spk2utt
    field="kanji"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df.set_index(df['speaker'] + '_' + df['id'])['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/text
    field="text.am.pos"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df.set_index(df['speaker'] + '_' + df['id'])['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/text.am.pos
    field="kana"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df.set_index(df['speaker'] + '_' + df['id'])['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/kana
    field="phone"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df.set_index(df['speaker'] + '_' + df['id'])['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/phone
}

yonden_baseline_ampnorm_spkinfo_daily() {
    # $1 is directory
    # $2 is path to dataset
    # $3 is dataset filter
    # Usage:
    # yonden_baseline_amp_norm_spkinfo "$dir" "$dataset" "$filter"
    # e.g., uttid and speaker as 指揮者ヒラオカ_210920_1358_平岡班_無線機_00010_0001009_0001151 指揮者ヒラオカ_210920_1358_平岡班 # may be every utteraces vary by distance of mircophone
    mkdir -p $1/$2
    field="seg_wav_amp_norm"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df.set_index(df['speaker'] + '_' + df['id'])['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/wav.scp
    speaker="speaker"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; df['speaker_new']=df['speaker']+'_'+df['work_date']+'_'+df['work_start_time']+'_'+df['group']; print(df.set_index(df['speaker'] + '_' + df['id'])['speaker_new'].to_csv(sep=' ',header=None), end='')" |  sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/utt2spk
    speaker="speaker"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; df['speaker_new']=df['speaker']+'_'+df['work_date']+'_'+df['work_start_time']+'_'+df['group']; print(df.groupby('speaker_new').apply(lambda x: ' '.join((x['speaker'] + '_' + x['id']).unique())).to_csv(sep=' ',header=None), end='')" |  sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/spk2utt
    field="kanji"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df.set_index(df['speaker'] + '_' + df['id'])['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/text
    field="text.am.pos"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df.set_index(df['speaker'] + '_' + df['id'])['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/text.am.pos
    field="kana"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df.set_index(df['speaker'] + '_' + df['id'])['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/kana
    field="phone"; python -c "import pandas as pd; df = pd.read_json('$info').T; df=df[$3]; print(df.set_index(df['speaker'] + '_' + df['id'])['$field'].to_csv(sep=' ',header=None), end='')" | sed -r 's/(.*) "(.*)"/\1 \2/g' | sort > $1/$2/phone
}

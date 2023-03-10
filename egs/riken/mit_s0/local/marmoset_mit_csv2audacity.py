import pandas as pd
import os

csv_file = "/project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/riken/mit_s0/data/local/csv/data_all_conditions_plus_calls_from_other_cages.csv"
out_dir = "/project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/riken/mit_s0/data/local/audacity"

df = pd.read_csv(csv_file, header=0, index_col=0)
if not os.path.exists(out_dir): os.makedirs(out_dir)

out_others_dir = os.path.join(out_dir, "calls_from_other_cages")
if not os.path.exists(out_others_dir): os.makedirs(out_others_dir)

for pair_index in range(1, 11):
    for animal_index in range(1, 3):
        for condition in ['together', 'animal1out', 'animal2out']:

            if condition == 'together':
                cond_short = 'toget'
            elif condition == 'animal1out':
                cond_short = 'a1out'
            else:
                cond_short = 'a2out'

            # p1a1_toget, p1a1_a1out, p1a1_a2out, p1a2_toget, p1a2_a1out, p1a2_a2out
            out_file = "p{}a{}_{}".format(pair_index, animal_index, cond_short)
            df_cur = df[(df.pair == "pair{}".format(pair_index)) & (df.animal == "animal{}".format(animal_index)) & (df.condition == condition)][["begin_time", "end_time", "call_type"]]
            df_cur.to_csv(os.path.join(out_dir, out_file + ".txt"), sep='\t', float_format="%.3f", header=False, index=False)

            # output the calls from other cages
            if condition == "together":
                df_cur = df[(df.pair == "pair{}".format(pair_index)) & (df.animal == "others")][["begin_time", "end_time"]]
                df_cur["not_available"] = "unk"
                df_cur.to_csv(os.path.join(out_others_dir, "p{}others_toget".format(pair_index) + ".txt"), sep='\t', float_format="%.3f", header=False, index=False)

#!/usr/bin/env python3

# Implemented by bin-wu at 22:06 on Feb. 25th, 2023

import os
import json
import datetime

import numpy as np
import pandas as pd
import argparse

description = '''
Notes:
The source of the json file
---
The official dataset of 'landman2020close - Close-range vocal interaction in the common marmoset (Callithrix jacchus)' from
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0227392

# Convert the original matlab annotation (annotation.mat) to a json file.
# ===
# data = load('E:\Workspace\Projects\Riken\marmoset.dataset\marmoset.short.range.vocalizaiton.2020.landman.MIT\annotations.mat') # annotations.mat from the original data
# fid = fopen('data.json','w');
# fprintf(fid,'%s',jsonencode(data, "PrettyPrint", true));
# fclose(fid);

The converted csv files
---
file_name = os.path.join(csv_dir, "data_all_conditions_plus_calls_from_other_cages.csv") # including calls from other cages
file_name = os.path.join(csv_dir, "data_all_conditions.csv") # excluding calls from other cages
file_name = os.path.join(csv_dir, "data_single_condition_together.csv") # excluding calls from other cages and only preserve together condition when two marmosets are in the same cage
'''

parser = argparse.ArgumentParser(description="Convert the json file to csv files for easy processing.\n" +
                                 "\nExample:\npython marmoset_mit_json2csv.py --json_annotation './data.json' --csv_dir './csv'\n" + description,
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--json_annotation", type=str, default="/project/nakamura-lab08/Work/bin-wu/share/data/marmoset_mit/processed/data.json",
                    help="the json file converted from annotation.mat from landman2020marmoset.short.range.vocalization.")
parser.add_argument("--csv_dir", type=str, default="/project/nakamura-lab08/Work/bin-wu/workspace/projects/clib/egs/riken/mit_s0/data/local/csv",
                    help="directories to output the csv files")

args = parser.parse_args()
annotation_json_file = args.json_annotation
csv_dir = args.csv_dir

if not os.path.exists(csv_dir): os.makedirs(csv_dir)

# The 'dataset' has two dimensions of 'pair' and 'condition':
# 10 pairs ('pair1',..., 'pair10') and 3 conditions ('together', 'animal1out', and 'animal2out')
with open(annotation_json_file, encoding='utf8') as f:
    annotation = json.load(f)
    dataset = annotation["dataset"]
conditions = ['together', 'animal1out', 'animal2out']

# Map pairs of animals to their names and sexes.                                                                          
animal2name = {}
animal2sex = {}
for pair_index in range(10):
    # animal_name and animal_sex only annotated for the condition of together
    recorded_data = dataset[pair_index][0]

    pair = "pair{}".format(pair_index+1)                                                                                             
    animal2name[pair + "_" + "animal1"] = recorded_data['animal1_name']
    animal2name[pair + "_" + "animal2"] = recorded_data['animal2_name']
    animal2sex[pair + "_" + "animal1"] = recorded_data['animal1_sex']
    animal2sex[pair + "_" + "animal2"] = recorded_data['animal2_sex']

data_df_all_list = []
for pair_index in range(10):
    for cond_index in range(3):
        # data from a recording session, recording animal1 and animal2 simultaneously.
        recorded_data = dataset[pair_index][cond_index]

        pair = "pair{}".format(pair_index+1)
        condition = conditions[cond_index]

        for animal in ['animal1', 'animal2', 'others']:
            # animal name, animal sex, and, call type are NOT AVALIABLE for the condition of 'others' (from other cages)
            animal_name = animal2name[pair + '_' + animal] if animal != 'others' else "NA"
            animal_sex = animal2sex[pair + '_' + animal] if animal != 'others' else "NA"
            call_type = recorded_data[animal+'_type'] if animal != 'others' else "NA"

            # others (calls from other cages) merely annotated for the condition of together
            if (condition != 'together' and animal == 'others'): continue

            # e.g., one annotation from 'animal1' of 'pair1' under the condition 'together'
            segments = {'begin_time' : recorded_data[animal+'_ts'],
                        'end_time'   : recorded_data[animal+'_tstp'],
                        'call_type'  : call_type,
                        'pair'       : pair,
                        'condition'  : condition,
                        'animal'     : animal,
                        'animal_name': animal_name,
                        'animal_sex' : animal_sex}
            segments_df = pd.DataFrame(segments)
            data_df_all_list.append(segments_df)

data_df_all = pd.concat(data_df_all_list) # including calls from other cages
abbr2full = {'tr': 'trill',
            'ph': 'phee',
            'trph': 'trillphee',
            'tw': 'twitter',
            'chi': 'chirp',
            'cha': 'chatter',
            'ek': 'ek',
            'ts': 'tsik',
            'pe': 'peep',
            'ic': 'infant_cry',
            'ot': 'other',
            'NA': 'NA'}
data_df_all['call'] = data_df_all.call_type.apply(lambda x: abbr2full[x])
data_df_all['begin_time_hms'] = data_df_all.begin_time.apply(lambda x:  str(datetime.timedelta(seconds=x))[:-3] + "s")
data_df_all['end_time_hms'] = data_df_all.end_time.apply(lambda x:  str(datetime.timedelta(seconds=x))[:-3] + "s")

data_df_all_conditions = data_df_all[data_df_all["animal"] != "others"] # excluding calls from other cages
data_df = data_df_all[(data_df_all.animal != "others") & (data_df_all.condition == "together")] # excluding calls from other cages and only preserve together condition

print("Totals number of calls: {}".format(len(data_df_all)))
print("Number of calls from other cages: {}".format(len(data_df_all[data_df_all["animal"] == "others"])))
print("Number of calls not from other cages: {}".format(len(data_df_all_conditions)))
print("Number of calls from condition animal1out: {}".format(len(data_df_all[(data_df_all.animal != "others") & (data_df_all.condition == "animal1out")])))
print("Number of calls from condition animal2out: {}".format(len(data_df_all[(data_df_all.animal != "others") & (data_df_all.condition == "animal2out")])))
print("Number of calls from condition together: {}".format(len(data_df)))

file_name = os.path.join(csv_dir, "data_all_conditions_plus_calls_from_other_cages.csv") # including calls from other cages
data_df_all.to_csv(file_name)

file_name = os.path.join(csv_dir, "data_all_conditions.csv") # excluding calls from other cages
data_df_all_conditions.to_csv(file_name)

file_name = os.path.join(csv_dir, "data_single_condition_together.csv") # excluding calls from other cages and only preserve together condition
data_df.to_csv(file_name)

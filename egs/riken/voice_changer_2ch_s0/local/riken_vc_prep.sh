#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -uo pipefail

# general configuration
stage=0  # start from 0 if you need to start from data preparation

dataset_name="riken_vc"

# "train_dev" is all uttids for training and development sets. Uttids are a sequence of animal pairs.
# e.g., "A B C D", where "A" and "B" are the first pair; "C" and "D" are the second pair.
# "dev_pair_ind" is the indices of pairs in train_dev array taken as the development set. Taking value 1 means take the second pair as the development set.
# e.g., ["A", "B", "C", "D", "E", "F"] would take ["C", "D"] as the development set and the remaining pairs of ["A", "B", "E", "F"] as the training set.
# train_dev="p2a1_toget p2a2_toget p3a1_toget p3a2_toget p4a1_toget p4a2_toget p5a1_toget p5a2_toget p6a1_toget p6a2_toget p7a1_toget p7a2_toget p8a1_toget p8a2_toget p9a1_toget p9a2_toget p10a1_toget p10a2_toget"
train_dev="p1a1_240716-1015 p1a2_240716-1015 p1a1_240711-0987 p1a2_240711-0987 p1a1_240711-0989 p1a2_240711-0989 p1a1_240712-0991 p1a2_240712-0991 p1a1_240712-0993 p1a2_240712-0993 p1a1_240712-0995 p1a2_240712-0995 p1a1_240713-0997 p1a2_240713-0997 p1a1_240713-0999 p1a2_240713-0999 p1a1_240713-1002 p1a2_240713-1002 p1a1_240716-1019 p1a2_240716-1019 p1a1_240717-1026 p1a2_240717-1026 p1a1_240717-1028 p1a2_240717-1028 p1a1_240718-1029 p1a2_240718-1029 p1a1_240718-1031 p1a2_240718-1031 p1a1_240718-1033 p1a2_240718-1033 p1a1_240718-1035 p1a2_240718-1035 p1a1_240719-1036 p1a2_240719-1036 p1a1_240719-1038 p1a2_240719-1038 p1a1_240719-1040 p1a2_240719-1040 p1a1_240719-1042 p1a2_240719-1042 p1a1_240720-1044 p1a2_240720-1044 p1a1_240720-1046 p1a2_240720-1046 p1a1_240720-1048 p1a2_240720-1048 p1a1_240720-1050 p1a2_240720-1050 p1a1_240721-1051 p1a2_240721-1051 p1a1_240721-1053 p1a2_240721-1053 p1a1_240721-1055 p1a2_240721-1055 p1a1_240722-1059 p1a2_240722-1059 p1a1_240722-1064 p1a2_240722-1064 p1a1_240723-1067 p1a2_240723-1067 p1a1_240723-1071 p1a2_240723-1071 p1a1_240724-1074 p1a2_240724-1074 p1a1_240725-1081 p1a2_240725-1081 p1a1_240725-1085 p1a2_240725-1085 p1a1_240726-1090 p1a2_240726-1090 p1a1_240727-1095 p1a2_240727-1095 p1a1_240727-1097 p1a2_240727-1097 p1a1_240728-1102 p1a2_240728-1102 p1a1_240728-1104 p1a2_240728-1104 p1a1_240728-1106 p1a2_240728-1106 p1a1_240729-1113 p1a2_240729-1113 p1a1_240730-1116 p1a2_240730-1116 p1a1_240730-1120 p1a2_240730-1120 p1a1_240801-1127 p1a2_240801-1127 p1a1_240801-1129 p1a2_240801-1129 p1a1_240801-1131 p1a2_240801-1131 p1a1_240801-1133 p1a2_240801-1133 p1a1_240802-1136 p1a2_240802-1136 p1a1_240802-1138 p1a2_240802-1138 p1a1_240802-1140 p1a2_240802-1140 p1a1_240803-1141 p1a2_240803-1141 p1a1_240803-1143 p1a2_240803-1143 p1a1_240803-1145 p1a2_240803-1145 p1a1_240803-1147 p1a2_240803-1147 p1a1_240804-1148 p1a2_240804-1148 p1a1_240804-1150 p1a2_240804-1150 p1a1_240804-1152 p1a2_240804-1152 p1a1_240806-1165 p1a2_240806-1165 p1a1_240806-1169 p1a2_240806-1169 p1a1_240806-1173 p1a2_240806-1173 p1a1_240807-1180 p1a2_240807-1180 p1a1_240807-1184 p1a2_240807-1184 p1a1_240807-1188 p1a2_240807-1188 p1a1_240807-1192 p1a2_240807-1192 p1a1_240808-1195 p1a2_240808-1195 p1a1_240808-1199 p1a2_240808-1199 p1a1_240808-1203 p1a2_240808-1203 p1a1_240809-1210 p1a2_240809-1210 p1a1_240809-1216 p1a2_240809-1216 p1a1_240809-1219 p1a2_240809-1219 p1a1_240810-1222 p1a2_240810-1222 p1a1_240810-1226 p1a2_240810-1226 p1a1_240810-1230 p1a2_240810-1230 p1a1_240810-1237 p1a2_240810-1237 p1a1_240811-1241 p1a2_240811-1241 p1a1_240811-1245 p1a2_240811-1245 p1a1_240811-1251 p1a2_240811-1251 p1a1_240812-1254 p1a2_240812-1254 p1a1_240812-1267 p1a2_240812-1267 p1a1_240813-1270 p1a2_240813-1270 p1a1_240813-1274 p1a2_240813-1274 p1a1_240813-1278 p1a2_240813-1278 p1a1_240813-1281 p1a2_240813-1281 p1a1_240817-1341 p1a2_240817-1341 p1a1_240817-1345 p1a2_240817-1345 p1a1_240817-1349 p1a2_240817-1349 p1a1_240818-1352 p1a2_240818-1352 p1a1_240818-1356 p1a2_240818-1356 p1a1_240818-1360 p1a2_240818-1360 p1a1_240818-1364 p1a2_240818-1364 p1a1_240819-1367 p1a2_240819-1367 p1a1_240819-1371 p1a2_240819-1371 p1a1_240819-1375 p1a2_240819-1375 p2a1_240805-1163 p2a2_240805-1163 p2a1_240806-1166 p2a2_240806-1166 p2a1_240806-1170 p2a2_240806-1170 p2a1_240806-1174 p2a2_240806-1174 p2a1_240806-1178 p2a2_240806-1178 p2a1_240807-1189 p2a2_240807-1189 p2a1_240808-1204 p2a2_240808-1204 p2a1_240809-1220 p2a2_240809-1220 p2a1_240810-1223 p2a2_240810-1223 p2a1_240810-1227 p2a2_240810-1227 p2a1_240810-1230 p2a2_240810-1230 p2a1_240810-1231 p2a2_240810-1231 p2a1_240810-1234 p2a2_240810-1234 p2a1_240810-1238 p2a2_240810-1238 p2a1_240811-1242 p2a2_240811-1242 p2a1_240811-1246 p2a2_240811-1246 p2a1_240812-1264 p2a2_240812-1264 p2a1_240812-1268 p2a2_240812-1268 p2a1_240813-1271 p2a2_240813-1271 p2a1_240813-1275 p2a2_240813-1275 p2a1_240813-1279 p2a2_240813-1279 p2a1_240813-1282 p2a2_240813-1282 p2a1_240817-1342 p2a2_240817-1342 p2a1_240817-1346 p2a2_240817-1346 p2a1_240817-1350 p2a2_240817-1350 p2a1_240818-1357 p2a2_240818-1357 p2a1_240819-1368 p2a2_240819-1368 p2a1_240819-1372 p2a2_240819-1372 p2a1_240819-1376 p2a2_240819-1376 p3a1_240806-1171 p3a2_240806-1171 p3a1_240806-1175 p3a2_240806-1175 p3a1_240806-1179 p3a2_240806-1179 p3a1_240808-1197 p3a2_240808-1197 p3a1_240808-1201 p3a2_240808-1201 p3a1_240808-1205 p3a2_240808-1205 p3a1_240809-1217 p3a2_240809-1217 p3a1_240809-1221 p3a2_240809-1221 p3a1_240810-1224 p3a2_240810-1224 p3a1_240810-1235 p3a2_240810-1235 p3a1_240811-1243 p3a2_240811-1243 p3a1_240811-1247 p3a2_240811-1247 p3a1_240811-1253 p3a2_240811-1253 p3a1_240812-1256 p3a2_240812-1256 p3a1_240812-1262 p3a2_240812-1262 p3a1_240813-1283 p3a2_240813-1283 p3a1_240817-1343 p3a2_240817-1343 p3a1_240818-1358 p3a2_240818-1358 p3a1_240818-1362 p3a2_240818-1362 p3a1_240819-1373 p3a2_240819-1373 p3a1_240819-1377 p3a2_240819-1377"
dev_pair_ind=0
test_wav_uttids="p1a1_240711-0985 p1a2_240711-0985" # a sequence of pairs

# Data
data=/data01/share/bin-wu/data/marmoset/vocalization/voice_changer_2ch

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. local/scripts/parse_options.sh || exit 1

if [ ${stage} -le 1 ]; then
    echo "Preparing wav.scp of wav files and aud.scp of audacity labels..."
    mkdir -p data/$dataset_name/
    rm -rf data/$dataset_name/wav.scp data/$dataset_name/aud.scp

    for file in $data/pair*/*wav; do
	echo $(basename $file .wav | sed -e 's/animal/a/g' -e 's/together/toget/g' | sed -r 's/^pair([0-9]+)_/p\1/g') $file >> data/$dataset_name/wav.scp
    done

    for file in $data/processed/clean_labels/pair*/annotations/*.txt; do
	echo $(basename $file .txt) $file >> data/$dataset_name/aud.scp
    done

    python local/scripts/data2info.py --data data/$dataset_name --scps aud:data/$dataset_name/aud.scp > data/$dataset_name/info.json
fi

if [ ${stage} -le 2 ]; then
    date
    echo "Preparing spectra and labels for training and development sets..."
    python local/mit_cnn_data_prep.py --info_json data/$dataset_name/info.json \
	   --train_dev  $train_dev \
	   --dev_pair_ind $dev_pair_ind \
	   --out_dir exp/data/$dataset_name
    date
fi

if [ ${stage} -le 3 ]; then
    date
    echo "Converting an onehot target into two onehot targets of an animal pair for training and development sets..."
    python local/mit_cnn_target_norm_single2.py --train_target_multi exp/data/$dataset_name/train_target_multi \
	   --dev_target_multi exp/data/$dataset_name/dev_target_multi \
	   --train_target_single1 exp/data/$dataset_name/train_target_single1 \
	   --train_target_single2 exp/data/$dataset_name/train_target_single2 \
	   --dev_target_single1 exp/data/$dataset_name/dev_target_single1 \
	   --dev_target_single2 exp/data/$dataset_name/dev_target_single2
    date
fi

if [ ${stage} -le 4 ]; then
    date
    echo "Create 2500ms spectral segments to prepare the inputs of test sets..."
    python local/mit_cnn_wav_into_test_2500_raw.py --info_json data/$dataset_name/info.json \
	   --test_wav_uttids  $test_wav_uttids \
	   --out_dir exp/data/$dataset_name
    date
fi

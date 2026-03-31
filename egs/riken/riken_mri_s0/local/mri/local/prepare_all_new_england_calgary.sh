python local/mri/local/prepare_new_england_mri.py |& tee logs/prep_new_england_mri2.log

python local/mri/local/prepare_calgary_mri.py |& tee logs/prep_calgary_mri.log

python local/mri/local/prepare_new_england_calgary_info_json.py |& tee logs/prep_info_json.log
 
python3 -c "import json; data=json.load(open('./data_mri/new_england/t1w/info.json')); print('\n'.join([f'{k}: {v[\"nii\"]}' for k,v in data.items() if v.get('age_rounded',999)<=50]))" > conf/mri/id2nii/new_england_id2nii_age_le_50.yaml

python3 -c "import json,yaml; data=json.load(open('./data_mri/calgary/t1w/info.json')); filtered={k:v['nii'] for k,v in data.items() if v.get('age_rounded',999)<=50}; print(yaml.dump(filtered, default_flow_style=False, sort_keys=False))" > conf/mri/id2nii/calgary_id2nii_age_le_50.yaml

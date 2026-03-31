import yaml

# Load the original id2nii.yaml files
with open('/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_mri_s0/data_mri/calgary/t1w/id2nii.yaml', 'r') as f:
    calgary_id2nii = yaml.safe_load(f)

with open('/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_mri_s0/data_mri/new_england/t1w/id2nii.yaml', 'r') as f:
    new_england_id2nii = yaml.safe_load(f)

# Load the subject lists for age > 50 months
with open('/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_mri_s0/conf/mri/subjects/subjects_calgary_t1_age_gt_50th_month.txt', 'r') as f:
    calgary_gt50_ids = [line.strip() for line in f if line.strip()]

with open('/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_mri_s0/conf/mri/subjects/subjects_new_england_t1_age_gt_50th_month.txt', 'r') as f:
    new_england_gt50_ids = [line.strip() for line in f if line.strip()]

# Filter id2nii dictionaries for age > 50 months
calgary_id2nii_gt50 = {id: path for id, path in calgary_id2nii.items() if id in calgary_gt50_ids}
new_england_id2nii_gt50 = {id: path for id, path in new_england_id2nii.items() if id in new_england_gt50_ids}

# Save to new YAML files
with open('/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_mri_s0/conf/mri/id2nii/calgary_id2nii_age_gt_50.yaml', 'w') as f:
    yaml.dump(calgary_id2nii_gt50, f, default_flow_style=False, sort_keys=True)

with open('/work01/home/bin-wu/workspace/projects/clib/egs/riken/riken_mri_s0/conf/mri/id2nii/new_england_id2nii_age_gt_50.yaml', 'w') as f:
    yaml.dump(new_england_id2nii_gt50, f, default_flow_style=False, sort_keys=True)

# Print statistics
print("\n" + "=" * 70)
print("YAML FILES CREATED FOR AGE > 50 MONTHS")
print("=" * 70)
print(f"\nCalgary:")
print(f"  IDs in subject list: {len(calgary_gt50_ids)}")
print(f"  IDs in YAML file: {len(calgary_id2nii_gt50)}")
print(f"  Output: conf/mri/id2nii/calgary_id2nii_age_gt_50.yaml")

print(f"\nNew England:")
print(f"  IDs in subject list: {len(new_england_gt50_ids)}")
print(f"  IDs in YAML file: {len(new_england_id2nii_gt50)}")
print(f"  Output: conf/mri/id2nii/new_england_id2nii_age_gt_50.yaml")

print(f"\nTotal:")
print(f"  Combined IDs: {len(calgary_id2nii_gt50) + len(new_england_id2nii_gt50)}")

# Show sample entries
print(f"\nSample entries (first 2):")
print(f"\nCalgary:")
for i, (id, path) in enumerate(sorted(calgary_id2nii_gt50.items())[:2]):
    print(f"  {id}: {path}")

print(f"\nNew England:")
for i, (id, path) in enumerate(sorted(new_england_id2nii_gt50.items())[:2]):
    print(f"  {id}: {path}")

print("=" * 70 + "\n")

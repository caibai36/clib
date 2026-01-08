import pandas as pd

# Read CSV
df = pd.read_csv('/data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/processed/all_kana_comment_token_speaker.csv')

# Filter rows where kana_comment_token contains <XXX> tags
df_tags = df[df['kana_comment_token'].str.contains('<[^>]+>', regex=True, na=False)]

print(f"Total rows with <XXX> in kana_comment_token: {len(df_tags)}")

# Verify the total
total_from_pivot = df_tags.groupby(['kana_comment_token', 'speaker']).size().sum()
print(f"Total from pivot table: {total_from_pivot}")

print("\n" + "="*80)
print("Speaker Distribution for Each <XXX> Tag")
print("="*80)

# Group by tag and speaker
result = df_tags.groupby(['kana_comment_token', 'speaker']).size().reset_index(name='count')

# Calculate total per tag and ratio
result['total_per_tag'] = result.groupby('kana_comment_token')['count'].transform('sum')
result['ratio'] = (result['count'] / result['total_per_tag'] * 100).round(2)

# Display each tag group
for tag in sorted(result['kana_comment_token'].unique()):
    tag_data = result[result['kana_comment_token'] == tag].sort_values('count', ascending=False)
    total = tag_data['count'].sum()
    
    print(f"\n{tag}")
    print("-" * 70)
    for _, row in tag_data.iterrows():
        print(f"  {row['speaker']:15s}: {row['count']:6d}  ({row['ratio']:6.2f}%)")
    print(f"  {'TOTAL':15s}: {total:6d}  (100.00%)")

print("\n" + "="*80)
print("Pivot Table View:")
print("="*80)
pivot = df_tags.pivot_table(index='kana_comment_token', 
                             columns='speaker', 
                             values='session_id', 
                             aggfunc='count', 
                             fill_value=0)
print(pivot)
print(f"\nGrand Total: {pivot.sum().sum()}")

print("\n" + "="*80)
print("Summary Statistics:")
print("="*80)
print(f"Total <XXX> tags: {len(df_tags)}")
print(f"Unique tag types: {df_tags['kana_comment_token'].nunique()}")
print(f"\nTags by speaker:")
speaker_totals = df_tags['speaker'].value_counts()
for speaker, count in speaker_totals.items():
    ratio = count / len(df_tags) * 100
    print(f"  {speaker:15s}: {count:6d}  ({ratio:6.2f}%)")

# Fixed line - removed quotes around variable name
print(f"\nChild speaker tags: {speaker_totals.get('child', 0):6d}  ({speaker_totals.get('child', 0)/len(df_tags)*100:6.2f}%)")

# # simplified version
# import pandas as pd

# # Read CSV
# df = pd.read_csv('/data01/share/bin-wu/data/human/speech/ntt_infant/ntt_infant/processed/all_kana_comment_token_speaker.csv')

# # Filter rows with <XXX> tags
# df_tags = df[df['kana_comment_token'].str.contains('<[^>]+>', regex=True, na=False)]

# print(f"Total rows with <XXX> tags: {len(df_tags)}\n")

# # Group by tag and speaker
# result = df_tags.groupby(['kana_comment_token', 'speaker']).size().reset_index(name='count')

# # Calculate ratio for each tag
# result['total_per_tag'] = result.groupby('kana_comment_token')['count'].transform('sum')
# result['ratio'] = (result['count'] / result['total_per_tag'] * 100).round(2)

# # Sort and display
# result = result.sort_values(['kana_comment_token', 'count'], ascending=[True, False])
# print(result.to_string(index=False))

# # Pivot table view
# print("\n" + "="*60)
# print("Pivot Table View:")
# print("="*60)
# pivot = df_tags.pivot_table(index='kana_comment_token',
#                              columns='speaker',
#                              values='session_id',
#                              aggfunc='count',
#                              fill_value=0)
# print(pivot)

#!/usr/bin/env python3
"""
Add Distinctive_feature_simplified column to distinctive_feature.csv
"""

import csv

def add_simplified_column(input_path, output_path):
    """
    Add Distinctive_feature_simplified column based on Distinctive_feature
    """
    
    # Mapping rules: extract the base category before voicing/palatalization markers
    def simplify_feature(distinctive_feature):
        """Extract simplified feature from full distinctive feature"""
        # Remove voicing markers (_u, _v) and palatalization (_pal)
        simplified = distinctive_feature
        
        # Remove suffixes
        simplified = simplified.replace('_u_pal', '')
        simplified = simplified.replace('_v_pal', '')
        simplified = simplified.replace('_u', '')
        simplified = simplified.replace('_v', '')
        simplified = simplified.replace('_pal', '')
        
        return simplified
    
    # Read input CSV
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    
    # Insert new column after Distinctive_feature
    new_fieldnames = []
    for field in fieldnames:
        new_fieldnames.append(field)
        if field == 'Distinctive_feature':
            new_fieldnames.append('Distinctive_feature_simplified')
    
    # Add simplified values
    for row in rows:
        distinctive_feature = row['Distinctive_feature']
        row['Distinctive_feature_simplified'] = simplify_feature(distinctive_feature)
    
    # Write output CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✓ Added 'Distinctive_feature_simplified' column")
    print(f"✓ Input:  {input_path}")
    print(f"✓ Output: {output_path}")
    print(f"✓ Total rows: {len(rows)}")
    
    # Show mapping summary
    print("\nMapping Summary:")
    print("-" * 60)
    mapping = {}
    for row in rows:
        orig = row['Distinctive_feature']
        simp = row['Distinctive_feature_simplified']
        if simp not in mapping:
            mapping[simp] = set()
        mapping[simp].add(orig)
    
    for simplified in sorted(mapping.keys()):
        originals = sorted(mapping[simplified])
        print(f"{simplified:15} <- {', '.join(originals)}")

if __name__ == "__main__":
    input_file = 'conf/dict/distinctive_feature.csv.bak'
    output_file = 'conf/dict/distinctive_feature.csv'
    
    add_simplified_column(input_file, output_file)
    
    print("\n" + "="*60)
    print("✓ DONE - File saved with UTF-8 encoding")
    print("="*60)


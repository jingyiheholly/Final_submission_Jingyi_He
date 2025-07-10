import pandas as pd

# Load both CSVs
csv1 = pd.read_csv('/home/hejin/Aimed/AI4Med/cleaned_discrption_file.csv')  # has 'img_name' and 'labels'
csv2 = pd.read_csv('/home/hejin/Aimed/AI4Med/llama_eval_results_from_json.csv')  # has 'img_id' and 'labels'

# Rename columns to unify the ID field
csv1.rename(columns={'img_name': 'image_id', 'label': 'labels_1'}, inplace=True)
csv2.rename(columns={'label': 'labels_2'}, inplace=True)

# Merge on img_id
merged = pd.merge(csv1[['image_id', 'labels_1']], csv2[['image_id', 'labels_2']], on='image_id', how='inner')

# Save result
merged.to_csv('/home/hejin/Aimed/AI4Med/merged_labels.csv', index=False)

print("Saved merged CSV as 'merged_labels.csv'")

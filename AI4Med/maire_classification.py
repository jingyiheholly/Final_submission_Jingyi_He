import pandas as pd

# Load your CSV
df = pd.read_csv('/home/hejin/Aimed/AI4Med/maira_all_predictions_2.csv')

# Normalize the column name if it's 'image_id' instead of 'img_id'
if 'image_id' in df.columns:
    df.rename(columns={'image_id': 'img_id'}, inplace=True)

# Function to check if a bbox value is empty or NaN
def is_bbox_empty(val):
    if pd.isna(val):
        return True
    return str(val).strip() == ''

# Add a boolean column for empty bbox
df['is_empty'] = df['bbox'].apply(is_bbox_empty)

# Group by img_id and determine if all entries are empty
status_df = df.groupby('img_id')['is_empty'].all().reset_index()
status_df['status'] = status_df['is_empty'].apply(lambda x: 'healthy' if x else 'unhealthy')

# Drop the helper column
status_df.drop(columns=['is_empty'], inplace=True)

# Save to CSV
output_path = '/home/hejin/Aimed/AI4Med/img_id_health_status.csv'
status_df.to_csv(output_path, index=False)

print(f"Saved health status CSV to: {output_path}")

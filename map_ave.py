import pandas as pd

# Load the CSV file
df = pd.read_csv("/Users/jingyihe/Desktop/map_brain_2.csv")

# Replace 'not found' with 0 and convert columns to numeric
for col in ['mAP@50:95', 'mAP@50', 'mAP@75']:
    df[col] = pd.to_numeric(df[col].replace("not found", 0))

# Calculate averages
avg_50_95 = df['mAP@50:95'].mean()
avg_50 = df['mAP@50'].mean()
avg_75 = df['mAP@75'].mean()

# Print results
print(f"Average mAP@50:95: {avg_50_95:.5f}")
print(f"Average mAP@50:   {avg_50:.5f}")
print(f"Average mAP@75:   {avg_75:.5f}")

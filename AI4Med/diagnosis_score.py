import pandas as pd

# Replace with your CSV file path
csv_path = '/home/hejin/Aimed/AI4Med/llama_prompts_and_results.csv'

# Load the CSV
df = pd.read_csv(csv_path)

# Count the occurrences of 'Valid' and 'Not Valid'
valid_count = (df['label'] == 'Valid').sum()
not_valid_count = (df['label'] == 'Not Valid').sum()

# Print results
print(f"Valid: {valid_count}")
print(f"Not Valid: {not_valid_count}")

import pandas as pd

# Load your CSV
df = pd.read_csv('/home/hejin/Aimed/AI4Med/llama_prompts_and_results_2.csv')

# Remove '.png' from img_id column
df['img_name'] = df['img_name'].str.replace('.png', '', regex=False)

# (Optional) Save back to CSV
df.to_csv('/home/hejin/Aimed/AI4Med/cleaned_discrption_file.csv', index=False)

print("Removed '.png' from img_id and saved to 'cleaned_file.csv'")

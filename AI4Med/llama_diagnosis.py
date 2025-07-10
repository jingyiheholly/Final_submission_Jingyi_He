import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    cache_dir="/home/hejin/hf_models"
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    cache_dir="/home/hejin/hf_models",
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# Load CSV with predictions
csv_path = "/home/hejin/Aimed/AI4Med/diagnosis_output.csv"
df = pd.read_csv(csv_path)

# Load annotation JSON
json_path = "/home/hejin/VLM-Seminar25-Dataset/nova_brain/annotations.json"
with open(json_path, "r") as f:
    annotations = json.load(f)

# Helper to extract label
def extract_label(text):
    text = text.lower()
    if "not valid" in text:
        return "Not Valid"
    elif "valid" in text:
        return "Valid"
    return "Uncertain"

# Generate response with retries
def generate_response_with_retries(prompt, max_retries=3):
    for attempt in range(max_retries):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        result = decoded[len(prompt):].strip()
        label = extract_label(result)
        if label != "Uncertain":
            return result, label
    return result, "Uncertain"

# Process each row
prompts = []

for _, row in df.iterrows():
    case_id = row["case_id"]
    predicted_dia = row["diagnosis"]

    try:
        final_diagnosis = annotations[case_id].get("final_diagnosis", "N/A")

        # Prompt
        prompt = f"""You are a helpful medical assistant trained to validate predicted diagnoses based on clinical context.

Given a predicted diagnosis, compare it to the ground truth final diagnosis and assess whether the prediction is medically valid.

### Ground Truth Final Diagnosis:
{final_diagnosis}

### Predicted Diagnosis:
{predicted_dia}

### Task:
Determine if the predicted diagnosis is clinically valid based on the final confirmed diagnosis.

Respond strictly using the following format:

- Start with either **"Valid"** or **"Not Valid"** (without quotes).
- Then, provide a brief explanation in one or two sentences.

### Your Response:
"""


        result, label = generate_response_with_retries(prompt, max_retries=3)

        if label == "Uncertain":
            print(f"[❌ Error] Could not determine validity for: {case_id}")

        prompts.append({
            "case_id": case_id,
            "final_diagnosis": final_diagnosis,
            "predicted_diagnosis": predicted_dia,
            "prompt": prompt,
            "result": result,
            "label": label
        })

    except KeyError:
        print(f"[⚠️ Warning] Missing data for {case_id}, skipping.")

# Save results to CSV
output_csv_path = "/home/hejin/Aimed/AI4Med/llama_prompts_and_results.csv"
df_prompts = pd.DataFrame(prompts)
df_prompts.to_csv(output_csv_path, index=False)

print(f"\n✅ Saved {len(df_prompts)} prompts and results to {output_csv_path}")

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

# Load prediction JSON (input)
prediction_json_path = "/home/hejin/Aimed/AI4Med/caption_with_similarity.json"
with open(prediction_json_path, "r") as f:
    predictions = json.load(f)

# Load annotation JSON (ground truth info)
annotation_json_path = "/home/hejin/VLM-Seminar25-Dataset/nova_brain/annotations.json"
with open(annotation_json_path, "r") as f:
    annotations = json.load(f)

# Helper to extract Valid/Not Valid
def extract_label(text):
    text = text.lower()
    if "not valid" in text:
        return "Not Valid"
    elif "valid" in text:
        return "Valid"
    return "Uncertain"

# Prompting with 3 tries max
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
    return None, None  # Failed after all retries

# Process each prediction
prompts = []

for item in predictions:
    image_id = item["imageID"]
    img_name = f"{image_id}.png"  # assuming image file ends with .png
    predicted_caption = item["predicted_caption"]
    gt_caption = item["gt_caption"]

    case_id = image_id.split('_')[0]

    try:
        clinical_history = annotations[case_id].get("clinical_history", "N/A")
        final_diagnosis = annotations[case_id].get("final_diagnosis", "N/A")

        prompt = f"""You are a helpful medical assistant trained to validate radiology image descriptions.

Given the following image ID, predicted AI-generated description, and the ground truth description from an expert, assess whether the AI prediction is clinically valid.

### Image ID:
{img_name}

### Clinical History:
{clinical_history}

### Final Diagnosis:
{final_diagnosis}

### Ground Truth Description:
"{gt_caption}"

### Predicted Description:
"{predicted_caption}"

### Task:
Determine if the predicted description is clinically and radiologically valid based on the ground truth, clinical history, and diagnosis. If valid, respond with "Valid" and explain briefly. If not, respond with "Not Valid" and explain what is incorrect or misleading.

### Your Response:
"""

        result, label = generate_response_with_retries(prompt, max_retries=3)

        if result is None:
            print(f"[❌ Error] Could not get a valid response after 3 tries for: {img_name}")
            result = "Uncertain after 3 retries"
            label = "Uncertain"

        prompts.append({
            "image_id": image_id,
            "prompt": prompt,
            "result": result,
            "label": label,
            "similarity_score": item.get("similarity_score", None)
        })

    except KeyError:
        print(f"[⚠️ Warning] Missing annotation data for {img_name}, skipping.")

# Save results to CSV
output_csv_path = "/home/hejin/Aimed/AI4Med/llama_eval_results_from_json.csv"
df_prompts = pd.DataFrame(prompts)
df_prompts.to_csv(output_csv_path, index=False)

print(f"\n✅ Saved {len(df_prompts)} prompts and results to {output_csv_path}")

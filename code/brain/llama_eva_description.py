import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from omegaconf import OmegaConf
cfg = OmegaConf.load("../config/config.yaml")
IMG_DIR=cfg.brain.img_dir
EVA_MODEL_NAME = cfg.brain.eva_model
CSV_PATH = "raw_brain_description.csv"
ANNOTATION_JSON = cfg.brain.annotations
OUTPUT_CSV = "Llama_description.csv"


def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer


def extract_label(text):
    text = text.lower()
    if "not valid" in text:
        return "Not Valid"
    elif "valid" in text:
        return "Valid"
    return "Uncertain"


def generate_response_with_retries(prompt, tokenizer, model, max_retries=3):
    attempt=0
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
        attempt+=1
    return None, "Uncertain"

def build_prompt(predicted_caption, gt_caption,
                 clinical_history, final_diagnosis):
    return f"""You are a helpful medical assistant trained to validate radiology image descriptions.

Given the following image ID, predicted AI-generated description, and the ground truth description from an expert, assess whether the AI prediction is clinically valid.

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

def process_all(model, tokenizer, df, annotations):
    prompts = []
    for _, row in df.iterrows():
        img_full_path = row["img_path"]
        predicted_caption = row["output"]
        img_name = os.path.basename(img_full_path)
        case_id = img_name.split('_')[0]

        try:
            case_data = annotations[case_id]
            gt_caption = case_data["image_findings"][img_name]["caption"]
            clinical_history = case_data.get("clinical_history", "N/A")
            final_diagnosis = case_data.get("final_diagnosis", "N/A")

            prompt = build_prompt(img_name, predicted_caption, gt_caption, clinical_history, final_diagnosis)
            result, label = generate_response_with_retries(prompt, tokenizer, model)

            if result is None:
                print(f"No valid response after 3 tries for {img_name}")
                result = "Uncertain after 3 retries"

            prompts.append({
                "img_name": img_name,
                "prompt": prompt,
                "result": result,
                "label": label
            })

        except KeyError:
            print(f"Missing data for {img_name}, skipping.")

    return pd.DataFrame(prompts)

def main():
    model, tokenizer = load_model_and_tokenizer(EVA_MODEL_NAME)
    df = pd.read_csv(CSV_PATH)
    with open(ANNOTATION_JSON, "r") as f:
        annotations = json.load(f)
    df_results = process_all(model, tokenizer, df, annotations)
    df_results.to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    main()

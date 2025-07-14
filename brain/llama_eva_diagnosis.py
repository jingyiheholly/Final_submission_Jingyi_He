import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from omegaconf import OmegaConf

cfg = OmegaConf.load("../config/config.yaml")
IMG_DIR=cfg.brain.img_dir
EVA_MODEL_NAME = cfg.brain.eva_model
CSV_PATH = "raw_brain_description.csv"
ANNOTATION_PATH = cfg.brain.annotations
OUTPUT_CSV = "Llama_description.csv"


# === CONFIGURATION ===
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_CACHE = "/home/hejin/hf_models"
CSV_PATH = "/home/hejin/Aimed/AI4Med/diagnosis_output.csv"
ANNOTATION_PATH = "/home/hejin/VLM-Seminar25-Dataset/nova_brain/annotations.json"
OUTPUT_CSV_PATH = "/home/hejin/Aimed/AI4Med/llama_prompts_and_results.csv"


# === LOADERS ===
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer


def load_data(csv_path, json_path):
    df = pd.read_csv(csv_path)
    with open(json_path, "r") as f:
        annotations = json.load(f)
    return df, annotations

def extract_label(text):
    text = text.lower()
    if "not valid" in text:
        return "Not Valid"
    elif "valid" in text:
        return "Valid"
    return "Uncertain"


def generate_response_with_retries(prompt, tokenizer, model, max_retries: int = 3):
    for _ in range(max_retries):
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


def build_prompt(predicted_dia, final_diagnosis):
    return f"""You are a helpful medical assistant trained to validate predicted diagnoses based on clinical context.

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


def process_diagnoses(df: pd.DataFrame, annotations: dict, tokenizer, model) -> pd.DataFrame:
    results = []

    for _, row in df.iterrows():
        case_id = row["case_id"]
        predicted_dia = row["diagnosis"]

        try:
            final_diagnosis = annotations[case_id].get("final_diagnosis", "N/A")
            prompt = build_prompt(predicted_dia, final_diagnosis)
            result, label = generate_response_with_retries(prompt, tokenizer, model)

            if label == "Uncertain":
                print(f"Could not determine validity for: {case_id}")

            results.append({
                "case_id": case_id,
                "final_diagnosis": final_diagnosis,
                "predicted_diagnosis": predicted_dia,
                "prompt": prompt,
                "result": result,
                "label": label
            })

        except KeyError:
            print(f"Missing data for {case_id}, skipping.")

    return pd.DataFrame(results)



def main():
    model, tokenizer = load_model_and_tokenizer(EVA_MODEL_NAME)
    df, annotations = load_data(CSV_PATH, ANNOTATION_PATH)
    result_df = process_diagnoses(df, annotations, tokenizer, model)
    result_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"Saved {len(result_df)} prompts and results.")

if __name__ == "__main__":
    main()

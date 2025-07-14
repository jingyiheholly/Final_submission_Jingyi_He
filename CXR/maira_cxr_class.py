from transformers import AutoModelForCausalLM, AutoProcessor
from pathlib import Path
import torch
from PIL import Image
import os
from omegaconf import OmegaConf
import os
import csv
from typing import List, Tuple, Optional
import pandas as pd

cfg = OmegaConf.load("../config/config.yaml")
IMG_DIR = cfg.chest.img_dir
ANNOTATIONS_PATH = cfg.chest.annotations
MODEL=cfg.chest.model
device = torch.device("cuda")
CSV_OUTPUT_PATH = "maira_raw_predictions.csv"

def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True,
    trust_remote_code=True,
)
    processor = AutoProcessor.from_pretrained(model_path,trust_remote_code=True)
    device = torch.device("cuda")
    model = model.to(device)
    return model, processor

def get_eva_data(IMAGES_PATH,img_id):
    """
    Get evaluation dataset
    """

    img_file = os.path.join(IMAGES_PATH, img_id + '.png')
    img = Image.open(img_file).convert("RGB")
    img = img.resize((518, 518))
    frontal_image = img

    sample_data = {
        "frontal": frontal_image,
        "lateral": None,
        "indication": "None.",
        "comparison": "None.",
        "technique": "Frontal view of the chest.",
        "phrase": "None."  # For the phrase grounding example. This patient has pleural effusion.
    }
    return sample_data

def prompt_the_model(sample_data,processor,device):
    processed_inputs = processor.format_and_preprocess_reporting_input(
    current_frontal=sample_data["frontal"],
    current_lateral=sample_data["lateral"],
    prior_frontal=None,  # Our example has no prior
    indication=sample_data["indication"],
    technique=sample_data["technique"],
    comparison=sample_data["comparison"],
    prior_report=None,  # Our example has no prior
    return_tensors="pt",
    get_grounding=True,  # For this example we generate a non-grounded report
)
    processed_inputs = processed_inputs.to(device)
    with torch.no_grad():
        output_decoding = model.generate(
        **processed_inputs,
        max_new_tokens=450,  # Set to 450 for grounded reporting
        use_cache=True,
    )
    prompt_length = processed_inputs["input_ids"].shape[-1]
    decoded_text = processor.decode(output_decoding[0][prompt_length:], skip_special_tokens=True)
    decoded_text = decoded_text.lstrip()  # Findings generation completions have a single leading space
    prediction = processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)
    print("Parsed prediction:", prediction)
    return prediction


def post_process_maira_output(
    prediction_tuples: List[Tuple[str, Optional[List[Tuple[float, float, float, float]]]]]
):
    """
    Convert model output into a list of dicts for CSV writing.
    Each dict contains: sentence (str), bbox (str or empty)
    """
    processed = []
    for sentence, bbox in prediction_tuples:
        if not sentence or not isinstance(sentence, str):
            continue
        sentence = sentence.strip()
        if not sentence.endswith('.'):
            sentence += '.'
        sentence = sentence[0].upper() + sentence[1:]
        bbox_str = str(bbox[0]) if bbox else ""  
        processed.append({"sentence": sentence, "bbox": bbox_str})
    return processed


def save_all_predictions_to_csv(all_predictions, csv_path):
    """
    Save a list of all predictions to one CSV file.
    Each entry must have: image_id, sentence, bbox
    """
    with open(csv_path, mode="w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "sentence", "bbox"])
        writer.writeheader()
        writer.writerows(all_predictions)

def is_bbox_empty(val):
    if pd.isna(val):
        return True
    return str(val).strip() == ''

def post_process_classification(file_path):
    df = pd.read_csv(file_path)
    if 'image_id' in df.columns:
        df.rename(columns={'image_id': 'img_id'}, inplace=True)
    df['is_empty'] = df['bbox'].apply(is_bbox_empty)
    status_df = df.groupby('img_id')['is_empty'].all().reset_index()
    status_df['status'] = status_df['is_empty'].apply(lambda x: 'healthy' if x else 'unhealthy')
    status_df.drop(columns=['is_empty'], inplace=True)
    output_path = 'classification_result.csv'
    status_df.to_csv(output_path, index=False)
    print(f"Saved health status CSV to: {output_path}")



if __name__ == "__main__":
    model,processor=load_model(MODEL)
    model = model.eval()
    all_results = []
    for filename in os.listdir(IMG_DIR):
        if filename.endswith(".png"):
            img_id = os.path.splitext(filename)[0]
            try:
                eva_data = get_eva_data(IMG_DIR, img_id)
                prediction = prompt_the_model(eva_data, processor, device)
                structured = post_process_maira_output(prediction)
                for entry in structured:
                    entry["image_id"] = img_id  
                all_results.extend(structured)
                print(f"Processed: {img_id}")
            except Exception as e:
                print(f"Error with {img_id}: {e}")
    df = pd.DataFrame(all_results)
    df.to_csv(CSV_OUTPUT_PATH, index=False)
    print(f"\nSaved predictions to: {CSV_OUTPUT_PATH}")   
    post_process_classification(CSV_OUTPUT_PATH)
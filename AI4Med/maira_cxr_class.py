from transformers import AutoModelForCausalLM, AutoProcessor
from pathlib import Path
import torch
from PIL import Image
import os
FOLDER = 'VLM-Seminar25-Dataset/chest_xrays'
ANNOTATIONS_PATH = os.path.join(FOLDER, 'annotations_len_50.json')
IMAGES_PATH = os.path.join(FOLDER, 'images')
MODEL_PATH="microsoft/maira-2"
CACHE_DIR="hf_models"
device = torch.device("cuda")
def load_model(model_path, trust_remote_code=True):
    model = AutoModelForCausalLM.from_pretrained(
    model_path,
    load_in_8bit=True,
    trust_remote_code=True,
    cache_dir=CACHE_DIR
)
    processor = AutoProcessor.from_pretrained(model_path,trust_remote_code=True)
    device = torch.device("cuda")
    #model = model.to(device)
    return model, processor

model,processor=load_model(MODEL_PATH)
model = model.eval()

def get_eva_data(IMAGES_PATH,img_id) -> dict[str, Image.Image | str]:
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
import os
import csv
from typing import List, Tuple, Optional

CSV_OUTPUT_PATH = "maira_all_predictions_2.csv"


def post_process_maira_output(
    prediction_tuples: List[Tuple[str, Optional[List[Tuple[float, float, float, float]]]]]
) -> List[dict]:
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
        bbox_str = str(bbox[0]) if bbox else ""  # 只保留第一个 box（如果存在）
        processed.append({"sentence": sentence, "bbox": bbox_str})
    return processed


def save_all_predictions_to_csv(all_predictions: List[dict], csv_path: str):
    """
    Save a list of all predictions to one CSV file.
    Each entry must have: image_id, sentence, bbox
    """
    with open(csv_path, mode="w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "sentence", "bbox"])
        writer.writeheader()
        writer.writerows(all_predictions)


# 🔁 批量处理并保存
all_results = []

for filename in os.listdir(IMAGES_PATH):
    if filename.endswith(".png"):
        img_id = os.path.splitext(filename)[0]
        try:
            eva_data = get_eva_data(IMAGES_PATH, img_id)
            prediction = prompt_the_model(eva_data, processor, device)
            structured = post_process_maira_output(prediction)
            for entry in structured:
                entry["image_id"] = img_id  # ← 给每个句子加上 image id
            all_results.extend(structured)
            print(f"✅ Processed: {img_id}")
        except Exception as e:
            print(f"❌ Error with {img_id}: {e}")

# 最后统一写入一个 CSV
save_all_predictions_to_csv(all_results, CSV_OUTPUT_PATH)
print(f"📁 All predictions saved to {CSV_OUTPUT_PATH}")

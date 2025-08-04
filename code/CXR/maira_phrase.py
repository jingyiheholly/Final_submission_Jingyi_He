from transformers import AutoModelForCausalLM, AutoProcessor
from pathlib import Path
import torch
from PIL import Image
import os
import json
import numpy as np
from scripts.calculate_map import compute_map_supervision
from omegaconf import OmegaConf
import csv
from typing import List, Tuple, Optional

CSV_OUTPUT_PATH = "map_result_cxr.csv"

cfg = OmegaConf.load("../config/config.yaml")
IMG_DIR = cfg.chest.img_dir
ANNOTATIONS_PATH = cfg.chest.annotations
MODEL=cfg.chest.model
device = torch.device("cuda")

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



def get_eva_data(IMAGES_PATH,img_id,ground_truth):
    """
    Get evaluation dataset
    """

    img_file = os.path.join(IMAGES_PATH, img_id + '.png')
    img = Image.open(img_file).convert("RGB")
    img = img.resize((518, 518))
    frontal_image = img
    gt_boxes = ground_truth[img_id].get("bbox_2d")

    a = {}

    for box in gt_boxes:
      coords = box[:4]
      label = box[4]
    
      if label not in a:
          a[label] = []
    
      a[label].append(coords)
    sample_data_list=[]
    label_list=[]
    for disease, __ in a.items():
        sample_data = {
        "frontal": frontal_image,
        "lateral": None,
        "indication": "None.",
        "comparison": "None.",
        "technique": "Frontal view of the chest.",
        "phrase": disease  # For the phrase grounding example. This patient has pleural effusion.
    }
        sample_data_list.append(sample_data)
        label_list.append(disease)

    return sample_data_list,label_list




def prompt_the_model(sample_data,processor,device):
    processed_inputs = processor.format_and_preprocess_phrase_grounding_input(
    frontal_image=sample_data["frontal"],
    phrase=sample_data["phrase"],
    return_tensors="pt",
)
    processed_inputs = processed_inputs.to(device)
    with torch.no_grad():
        output_decoding = model.generate(
        **processed_inputs,
        max_new_tokens=150,  # Set to 450 for grounded reporting
        use_cache=True,
    )
    prompt_length = processed_inputs["input_ids"].shape[-1]
    decoded_text = processor.decode(output_decoding[0][prompt_length:], skip_special_tokens=True)
    prediction = processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)

    print("Parsed prediction:", prediction)
    return prediction


def extract_gt_boxes(ground_truth: dict,img_id:str) :
    """
    Create two aligned lists of ground truth and predicted labels.

    Args:
        ground_truth: dict with image_id → {status: "healthy"/"unhealthy"}
        prediction_status: dict with image_id → "healthy"/"unhealthy"

    Returns:
        (gt, pred): Tuple of lists with labels
    """
    gt_boxes = ground_truth[img_id].get("bbox_2d")
    true_boxes = []
    true_classes = []
    class_name_to_index = {}
    index_to_class_name = {}
    class_index_counter = 0
    for box in gt_boxes:
        coords = box[:4]
        label = box[4]
        if label not in class_name_to_index:
            class_name_to_index[label] = class_index_counter
            index_to_class_name[class_index_counter] = label
            class_index_counter += 1
        true_boxes.append(coords)
        true_classes.append(class_name_to_index[label])
        
    return true_boxes,true_classes

if __name__ == "__main__":
    model,processor=load_model(MODEL)
    model = model.eval()
    with open(ANNOTATIONS_PATH, "r") as f:
        ground_truth = json.load(f)
        all_results=[]
        with open(CSV_OUTPUT_PATH, mode="w", newline='', encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image_id", "disease_name", "mAP@50:95", "mAP@50", "mAP@75","gt_box","pre_box"])
            writer.writeheader()
            for filename in os.listdir(IMG_DIR):
                if filename.endswith(".png"):
                    img_id = os.path.splitext(filename)[0]
                    try:
                        eva_data_list,gt_label_list = get_eva_data(IMG_DIR, img_id,ground_truth)
                        for index, eva_data in enumerate(eva_data_list):
                            prediction = prompt_the_model(eva_data, processor, device)
                            print(f" Processed: {img_id}/{index}")
                            disease_to_boxes={}
                            post_boxes=[]
                            true_boxes,true_classes=extract_gt_boxes(ground_truth,img_id)
                            for sentence, boxes in prediction:                      
                                disease = sentence.strip().rstrip(".")
                                if boxes is None:
                                    writer.writerow({
                                 "image_id": img_id,
                                 "disease_name": gt_label_list[index],
                                 "mAP@50:95": "not found",
                                 "mAP@50": "not found",
                                 "mAP@75": "not found",
                                 "gt_box":true_boxes,
                                 "pre_box":"not found"
                                    })
                                    continue
                                for box in boxes:
                                    box_np = np.array(box) * 1024
                                    post_boxes.append(box_np.tolist())
                                pre_boxes = []
                                pre_classes = []
                                class_name_to_index = {}
                                index_to_class_name = {}
                                class_index_counter = 0 
                                if boxes:
                                    disease_to_boxes[disease] = post_boxes

                                    for disease, boxes in disease_to_boxes.items():
                                        if disease not in class_name_to_index:
                                            class_name_to_index[disease] = class_index_counter
                                            index_to_class_name[class_index_counter] = disease
                                            class_index_counter += 1

                                        for box in boxes:
                                            pre_boxes.append(box)
                                            pre_classes.append(class_name_to_index[disease])
                                    result = compute_map_supervision(pre_boxes, pre_classes, true_boxes, true_classes)
                                    writer.writerow({
                                        "image_id": img_id,
                                        "disease_name": disease,
                                        "mAP@50:95": result.map50_95,
                                        "mAP@50": result.map50,
                                        "mAP@75": result.map75,
                                        "gt_box": true_boxes,
                                        "pre_box":pre_boxes
                                    })
                    except Exception as e:
                        print(f" Error with {img_id}: {e}")
        



    

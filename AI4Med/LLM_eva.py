# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.utils.data import Dataset
import json
import torch
import os
from transformers import pipeline
from torch.utils.data import DataLoader
import re

import csv
import ast

# Path to your CSV file
csv_file_path = '/root/autodl-tmp/HuatuoGPT-Vision/Aimed/AI4Med/image_descriptions_finer.csv'

# Dictionary to store image name and description
img_description_dict_pre = {}

# Read the CSV
with open(csv_file_path, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip the header row

    for row in reader:
        img_path = row[0].strip()
        description_str = row[1].strip()
        
        # Extract image filename only
        img_name = img_path.split('/')[-1]
        
        # Store in dictionary
        img_description_dict_pre[img_name] = description_str


json_path = '/root/autodl-tmp/VLM_Seminar25_Dataset/nova_brain/annotations.json'

# Load the JSON
with open(json_path, 'r') as f:
    data = json.load(f)

# Dictionary to store image filename -> caption
img_caption_dict = {}

# Loop through each case
for case_id, case_info in data.items():
    if 'image_findings' in case_info:
        for img_filename, img_info in case_info['image_findings'].items():
            caption = img_info.get('caption', '')
            img_caption_dict[img_filename] = caption
query = (
    "Image context: Brain MRI\n"
    "Predicted Description:\n"
    "{pre_dis}\n"
    "Ground Truth Description:\n"
    "{gt_dis}\n"
    "Question:\n"
    "Does the predicted description sufficiently align with the clinical findings in the ground truth description?\n"
    "Please answer:\n"
    "1. Alignment (choose one): Aligned / Not Aligned\n"
    "2. Brief explanation in one sentence highlighting key similarities or differences.\n"
    "3. A score from 0 to 1 indicating how closely the prediction reflects the ground truth.\n"
)
def prompt(pre_dis,gt_dis):
    instruction = (
        "You are an expert radiologist assistant AI trained to evaluate the correctness of radiology report descriptions."
        "Given a predicted description and a ground truth description, your job is to determine whether the predicted description accurately matches the findings and clinical implications described in the ground truth. "
        "Be precise, objective, and concise in your judgment."
    )

    filled_query = query.format(pre_dis=pre_dis, gt_dis=gt_dis)


    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content":  filled_query}
    ]

import re

def process_llm_outputs_without_ids(raw_responses, img_ids):
    results = []

    for img_id, entry in zip(img_ids, raw_responses):
        alignment_match = re.search(r"(?:1\.\s*)?Alignment:\s*(Aligned|Not Aligned)|1\.\s*(Aligned|Not Aligned)", entry, re.IGNORECASE)
        score_match = re.search(r"Score:\s*([0-1](?:\.\d+)?)", entry)
        #explanation_match = re.search(r"(?:2\.|Explanation:)\s*(.*?)(?=\n3\.|Score:|$)", entry, re.DOTALL)

        if not (alignment_match and score_match):
            print(f"[ERROR] Could not parse complete response for image: {img_id}")

        alignment = (alignment_match.group(1) or alignment_match.group(2)).strip().capitalize() if alignment_match else "Unknown"
        score = float(score_match.group(1)) if score_match else "N/A"
        #explanation = explanation_match.group(1).strip() if explanation_match else ""

        results.append({
            "image_id": img_id,
            "alignment": alignment,
            "score": score
        })

    return results



if __name__ == "__main__":
    



    cache_directory = "/root/autodl-tmp/hf_model"  # Replace with your actual cache path

    tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    cache_dir=cache_directory
)
    # After loading the tokenizer
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    cache_dir=cache_directory
)

    model = model.to("cuda")

    model.eval()
    for img_id in img_description_dict_pre.keys():
        pre_dis=img_description_dict_pre[img_id]
        gt_dis=img_caption_dict[img_id]
        messages=prompt(pre_dis,gt_dis)
        #prompt_text= tokenizer.apply_chat_template(messages, tokenize=False)
        #inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True)

       # input_ids = inputs["input_ids"].to(model.device)
       # attention_mask = inputs["attention_mask"].to(model.device)

       # output = model.generate(
  # input_ids=input_ids,
  #  attention_mask=attention_mask,
  #  max_new_tokens=1000,
  ##  pad_token_id=tokenizer.eos_token_id
#)


        #generated_tokens = output[0][input_ids.shape[-1]:]
        #response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        #print("Clean response:", response)
        with open("llm_prompt.txt", "a") as f:  # "a" appends to the file
            f.write("Clean response:\n")
            f.write(img_id+"\n")
            for msg in messages:
              role = msg["role"]
              content = msg["content"]
              f.write(f"{role.upper()}:\n{content}\n\n")


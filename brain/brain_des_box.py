from HuatuoGPT_Vision.cli import HuatuoChatbot
import os
import csv
from omegaconf import OmegaConf
import ast
from PIL import Image

cfg = OmegaConf.load("../config/config.yaml")
IMG_DIR=cfg.brain.img_dir

query_discription = (
    "You are a radiologist. Please provide a concise radiology-style description (caption) of the brain MRI image, including: "
    "- MRI sequence type (e.g., T1, T2, FLAIR); "
    "- Anatomic location of abnormality; "
    "- Signal characteristics (e.g., hyperintense, hypointense); "
    "- Possible interpretations (e.g., calcification, hemorrhage)."
)
query_detection = (
    "You are a radiologist reviewing a brain MRI. "
    "Mark all potentially abnormal regions—lesions, hemorrhages, infarcts, calcifications, or any subtle or ambiguous findings.\n\n"
    "Output each as a 2D bounding box in **absolute pixel coordinates**: [x_min, y_min, x_max, y_max].\n\n"
    "Guidelines:\n"
    "- Return a list of boxes if multiple regions exist.\n"
    "- If uncertain, err on the side of inclusion.\n"
    "- If no abnormality is visible, return an empty list: [].\n"
    "- Output **only** the list of bounding boxes.\n\n"
)

def denormalize_box(bbox_str, image_size):
    # Convert the outer string to a list
    parsed = ast.literal_eval(bbox_str)
    
    if not parsed or parsed == ['[]'] or parsed[0] == '[]':
        return None  # no box present
    
    # Get the inner list and convert to floats
    bbox = ast.literal_eval(parsed[0])  # parse the inner '[x, y, x2, y2]'
    bbox = [float(x) for x in bbox]
    
    W, H = image_size
    x_min = int(bbox[0] * W)
    y_min = int(bbox[1] * H)
    x_max = int(bbox[2] * W)
    y_max = int(bbox[3] * H)
    return [x_min, y_min, x_max, y_max]



if __name__ == "__main__":
    # change here to th eactual path to the model
    bot = HuatuoChatbot("path/huatuogpt")
    img_path = []
    csv_file_dis = 'raw_brain_description.csv'
    csv_file_box = 'raw_brain_box.csv'

    for filename in os.listdir(IMG_DIR):
    # Optionally filter only image files (e.g., jpg, png, etc.)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')):
            full_path = os.path.join(IMG_DIR, filename)
            img_path.append(full_path)
    with open(csv_file_dis, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['img_path', 'output'])

        for img_path in img_path:
            try:
                output = bot.inference(query_discription, img_path)
                writer.writerow([img_path, output])
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    with open(csv_file_box, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['img_path', 'output'])

        for img_path in img_path:
            try:
                output = bot.inference(query_detection, img_path)
                writer.writerow([img_path, output])
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
import csv
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import ast
import json
import re
from scripts.calculate_map import compute_map_supervision, draw_boxes_2
from omegaconf import OmegaConf

CSV_PATH = "maira_raw_predictions.csv"
cfg = OmegaConf.load("../config/config.yaml")
IMG_DIR = cfg.chest.img_dir
ANNOTATIONS_PATH = cfg.chest.annotations
MODEL=cfg.chest.model
device = torch.device("cuda")

def norm_box_to_token_box(bbox, grid_size=100):
    x1, y1, x2, y2 = bbox
    return [
        f"<x{int(round(x1 * (grid_size - 1)))}>",
        f"<y{int(round(y1 * (grid_size - 1)))}>",
        f"<x{int(round(x2 * (grid_size - 1)))}>",
        f"<y{int(round(y2 * (grid_size - 1)))}>"
    ]

def parse_grouped_token_boxes(token_boxes, image_size=(1024, 1024), grid_size=100):
    def extract_index(token):
        match = re.search(r"<[xy](\d+)>", token)
        return int(match.group(1)) if match else None

    boxes_in_px = []
    w, h = image_size

    for tokens in token_boxes:
        if len(tokens) != 4:
            continue
        x1 = extract_index(tokens[0])
        y1 = extract_index(tokens[1])
        x2 = extract_index(tokens[2])
        y2 = extract_index(tokens[3])

        if None not in (x1, y1, x2, y2):
            x1_px = (x1 / (grid_size - 1)) * w
            y1_px = (y1 / (grid_size - 1)) * h
            x2_px = (x2 / (grid_size - 1)) * w
            y2_px = (y2 / (grid_size - 1)) * h
            boxes_in_px.append((x1_px, y1_px, x2_px, y2_px))

    return boxes_in_px


def postprocessing_csv(csv_path: str) -> dict:
    """
    Read predictions from CSV and classify each image_id as 'healthy' or 'unhealthy'.
    
    Returns:
        A dictionary: {image_id: 'healthy' | 'unhealthy'}
    """
    health_status = defaultdict(lambda: "healthy")
    token_boxes=[]
    bbox_dict = defaultdict(list)
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames + ["Processed_box_2"]
        for row in rows:
            image_id = row["image_id"]
            bbox_str = row["bbox"].strip()
            bbox = ast.literal_eval(bbox_str) if bbox_str else None
            if bbox:  # If any bbox is present, mark as unhealthy
                health_status[image_id] = "unhealthy"
                token_box = norm_box_to_token_box(bbox)
                pixel_box = parse_grouped_token_boxes([token_box])  # returns a list with one item
                if pixel_box:
                    bbox_dict[image_id].append(pixel_box[0])
                    row["Processed_box_2"]=pixel_box[0]
                    
            '''with open(csv_path, mode="w", newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)'''
    
    return dict(health_status),bbox_dict

def extract_gt_and_pred_lists(ground_truth: dict, prediction_status: dict) -> tuple[list[str], list[str]]:
    """
    Create two aligned lists of ground truth and predicted labels.

    Args:
        ground_truth: dict with image_id → {status: "healthy"/"unhealthy"}
        prediction_status: dict with image_id → "healthy"/"unhealthy"

    Returns:
        (gt, pred): Tuple of lists with labels
    """
    gt = []
    pred = []

    for image_id in ground_truth:
        gt_status = ground_truth[image_id].get("status")
        pred_status = prediction_status.get(image_id, "healthy")  # default to healthy if missing

        if gt_status is not None:
            gt.append(gt_status)
            pred.append(pred_status)

    return gt, pred


if __name__ == "__main__":
    status_dict,tokon_dict = postprocessing_csv(CSV_PATH)
    with open(ANNOTATIONS_PATH, "r") as f:
        ground_truth = json.load(f)   
    for image_id in ground_truth:
        full_path=IMG_DIR/image_id
        print("image_id is"+str(image_id))
        gt_boxes = ground_truth[image_id].get("bbox_2d")
        for i, box in enumerate(gt_boxes):
            gt_boxes[i] = box[:4]
        pre_boxes=tokon_dict[image_id]
        list_boxes = [list(box) for box in pre_boxes]
        result=compute_map_supervision(list_boxes, None, gt_boxes, None)
        if gt_boxes!=[]:
            save_path=f'cxr_phrase_results/{image_id}.jpg'
            draw_boxes_2(pre_boxes, gt_boxes, image_size=(1024, 1024),img_path=full_path, save_path=save_path)





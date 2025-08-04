import csv
import ast
import os
import json
from PIL import Image
from pathlib import Path
from VLM_Seminar25_Dataset.scripts.calculate_map import compute_map_supervision,draw_boxes_2
from omegaconf import OmegaConf
cfg = OmegaConf.load("../config/config.yaml")
IMG_DIR=cfg.brain.img_dir
ANNOTATION_PATH=cfg.brain.annotations


def parse_predicted_boxes(csv_path):
    """Reads predicted normalized boxes from CSV and converts to absolute pixel coordinates."""
    results = []
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            img_path = row['img_path']
            img_id = os.path.basename(img_path)
            try:
                parsed_boxes = ast.literal_eval(row['boxes'])
                img = Image.open(img_path)
                W, H = img.size

                boxes = []
                for box_str in parsed_boxes:
                    box_str = box_str.strip()
                    if box_str == '[]':
                        continue
                    norm_box = ast.literal_eval(box_str)
                    x1, y1 = norm_box[0] * W, norm_box[1] * H
                    x2, y2 = norm_box[2] * W, norm_box[3] * H
                    boxes.append([x1, y1, x2, y2])

            except Exception as e:
                print(f"[ERROR] Failed to process {img_path}: {e}")
                boxes = []

            results.append((img_id, boxes))
    return results


def load_ground_truth(json_path):
    """Loads ground truth boxes from a JSON annotation file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    gt = []
    for case_info in data.values():
        image_findings = case_info.get("image_findings", {})
        for img_name, img_info in image_findings.items():
            boxes = img_info.get("bbox_2d_gold", [])
            for box in boxes:
                gt.append((img_name, box))
    return gt


def evaluate_and_save(predictions, ground_truth, output_csv):
    """Evaluates mAP scores and writes results to a CSV file."""
    with open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["img_id", "mAP@50:95", "mAP@50", "mAP@75","predicted_boxes","ground_truth_boxes"])

        for img_id, pred_boxes in predictions:
            full_img_path = IMG_DIR / img_id
            gt_boxes = [box for name, box in ground_truth if name == img_id]
            if not gt_boxes:
                print(f"[WARN] No ground truth for {img_id}, skipping.")
                continue

            try:
                result = compute_map_supervision(
                    pred_boxes if pred_boxes else [], None,
                    gt_boxes, None
                )
                writer.writerow([img_id, result.map50_95, result.map50, result.map75,pred_boxes,gt_boxes])
                if gt_boxes!=[]:
                    save_path=f'brain_box_results/{img_id}.jpg'
                    draw_boxes_2(pred_boxes, gt_boxes, image_size=(1024, 1024),img_path=full_path, save_path=save_path)


            except Exception as e:
                print(f"Failed to compute mAP for {img_id}: {e}")


def main():
    # === Config paths ===
    pred_csv_path = Path("raw_brain_box.csv")
    gt_json_path = ANNOTATION_PATH
    output_csv_path = Path("map_brain.csv")

    predictions = parse_predicted_boxes(pred_csv_path)
    ground_truth = load_ground_truth(gt_json_path)
    evaluate_and_save(predictions, ground_truth, output_csv_path)
    print(f"Results saved to: {output_csv_path}")


if __name__ == "__main__":
    main()

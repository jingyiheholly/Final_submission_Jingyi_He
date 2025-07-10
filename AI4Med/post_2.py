import csv
import ast
import os
import json
from PIL import Image
from VLM_Seminar25_Dataset.scripts.calculate_map import compute_map_supervision, draw_boxes_2
input_file = '/root/autodl-tmp/HuatuoGPT-Vision/image_boxes_3.csv'
output_data = []

# Step 1: Read normalized boxes and convert to absolute pixel coords
with open(input_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        img_path = row['img_path']
        boxes_str = row['boxes']
        img_id = os.path.basename(img_path)

        try:
            parsed_boxes = ast.literal_eval(boxes_str)  # e.g., ['[0.47, 0.26, 0.81, 0.59]'] or ['[]']
    
            # Open image to get dimensions
            img = Image.open(img_path)
            W, H = img.size

            boxes = []
            for box_str in parsed_boxes:
                box_str = box_str.strip()
                if box_str == '[]':
                    continue  # skip empty boxes
                norm_box = ast.literal_eval(box_str)  # e.g., [0.47, 0.26, 0.81, 0.59]
                x1 = norm_box[0] * W
                y1 = norm_box[1] * H
                x2 = norm_box[2] * W
                y2 = norm_box[3] * H
                boxes.append([x1, y1, x2, y2])
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
            boxes = []


        output_data.append((img_id, boxes))

# Load JSON data (replace 'your_file.json' with your actual file path)
with open('/root/autodl-tmp/VLM-Seminar25-Dataset/nova_brain/annotations.json', 'r') as f:
    data = json.load(f)

output_gt = []

# Loop over each case
for case_id, case_info in data.items():
    image_findings = case_info.get("image_findings", {})
    for img_name, img_info in image_findings.items():
        boxes = img_info.get("bbox_2d_gold", [])
        for box in boxes:
            output_gt.append((img_name, box))

csv_output_path = "map_brain.csv"
# Print or use the output_data
with open(csv_output_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Write header
    writer.writerow(["img_id", "mAP@50:95", "mAP@50", "mAP@75"])

    for img_id, boxes in output_data:
        print("image_id is " + str(img_id))

        # Find all ground truth boxes for this image
        boxes_for_img = [box for name, box in output_gt if name == img_id]

        # Skip if no GT
        if not boxes_for_img:
            print(f"No ground truth for {img_id}, skipping.")
            continue

        # Compute mAP
        result = compute_map_supervision(boxes[0], None, boxes_for_img, None)

        # Print results
        print("mAP@50:95:", result.map50_95)
        print("mAP@50:   ", result.map50)
        print("mAP@75:   ", result.map75)

        # Write to CSV
        writer.writerow([img_id, result.map50_95, result.map50, result.map75])
    

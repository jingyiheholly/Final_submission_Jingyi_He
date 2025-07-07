import csv
import ast
import os
import json
input_file = '/root/autodl-tmp/HuatuoGPT-Vision/image_boxes_2.csv'  # replace with your file path
output_data = []
from VLM_Seminar25_Dataset.scripts.calculate_map import compute_map_supervision, draw_boxes_2
with open(input_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        img_path = row['img_path']
        boxes_str = row['boxes']

        # Get the image ID (filename only)
        img_id = os.path.basename(img_path)

        # Parse the box string into a list of lists of floats
        try:
            parsed_boxes = ast.literal_eval(boxes_str)  # e.g., ['[263.7, 184.5, 309.1, 220]']
            boxes = [ast.literal_eval(box) for box in parsed_boxes if box.strip()]
        except:
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


# Print or use the output_data
for img_id, boxes in output_data:
    print("image_id is"+str(img_id))
    boxes_for_img = [box for name, box in output_gt if name == img_id]
    result=compute_map_supervision(boxes[0], None, boxes_for_img, None)
    print(result)
    

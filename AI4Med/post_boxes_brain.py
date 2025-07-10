import csv
import ast
from PIL import Image

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

# Path to your CSV file
csv_file = "/root/autodl-tmp/HuatuoGPT-Vision/Aimed/AI4Med/image_boxes_3.csv"

# Read and process each row
with open(csv_file, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        img_path = row["img_path"]
        bbox_str = row["output"]

        # Load image
        try:
            img = Image.open(img_path)
            size = img.size
        except FileNotFoundError:
            print(f"Image not found: {img_path}")
            continue

        # Process bounding box
        abs_box = denormalize_box(bbox_str, size)
        if abs_box:
            print(f"Image: {img_path}")
            print(f"Absolute bbox: {abs_box}")
        else:
            print(f"Image: {img_path} - No bounding box")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import ast
import os

def draw_and_save_boxes_from_csv_row(row, image_root, save_dir):
    image_id = row['img_id']
    image_path = os.path.join(image_root, image_id)
    save_path = os.path.join(save_dir, image_id)

    # Parse boxes safely
    try:
        pred_boxes = ast.literal_eval(row['predicted_boxes']) if row['predicted_boxes'] != "N/A" else []
    except:
        pred_boxes = []

    try:
        true_boxes = ast.literal_eval(row['ground_truth_boxes']) if row['ground_truth_boxes'] != "N/A" else []
    except:
        true_boxes = []

    # Load image
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        return

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img)
    img_w, img_h = img.size

    # Draw predicted boxes (red)
    for box in pred_boxes:
        if len(box) == 4:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

    # Draw ground truth boxes (green)
    for box in true_boxes:
        if len(box) == 4:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='green', facecolor='none')
            ax.add_patch(rect)

    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
    ax.axis('off')
    plt.tight_layout()

    # Save figure
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

import pandas as pd

df = pd.read_csv("/Users/jingyihe/Aimed/AI4Med/map_brain_2.csv")
image_root = "/Users/jingyihe/VLM-Seminar25-Dataset/nova_brain/images"
save_dir = "/Users/jingyihe/Aimed/AI4Med/brain_results"

for i, row in df.iterrows():
    draw_and_save_boxes_from_csv_row(row, image_root, save_dir)

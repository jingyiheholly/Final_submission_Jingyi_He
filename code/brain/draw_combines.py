import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import ast
import os

def draw_and_save_boxes_from_csv_row(img_id, image_root, save_dir,pred_boxes_1,pred_boxes_2,true_boxes):
    image_id = img_id
    image_path = os.path.join(image_root, image_id)
    save_path = os.path.join(save_dir, image_id)

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
    for box in pred_boxes_1:
        if len(box) == 4:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
    for box in pred_boxes_2:
        if len(box) == 4:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='yellow', facecolor='none')
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

image_root = "/Users/jingyihe/Aimed/VLM_Seminar25_Dataset/nova_brain/images"
save_dir = "/Users/jingyihe/Aimed/brain/brain_results_combined"

# Select a specific image ID
target_img_id = "case0043_002"
target_img_path = "case0043_002.png"
a=[[]]
b=[[230,
                    230,
                    460,
                    820],
                    [540,
                    230,
                    770,
                    820]]
c=[[586.7178385416667, 316.2794596354167, 745.727978515625, 722.6367838541667], [232.772265625, 215.330126953125, 467.3535970052083, 820.3775878906249]]
draw_and_save_boxes_from_csv_row(target_img_path, image_root,save_dir,a,b,c)

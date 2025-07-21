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
        if len(box) == 5:
            x1, y1, x2, y2 = box[:4]
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
        if len(box) == 5:
            x1, y1, x2, y2 = box[:4]
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

image_root = "/Users/jingyihe/Aimed/VLM_Seminar25_Dataset/chest_xrays/images"
save_dir = "/Users/jingyihe/Aimed/CXR/CXR_results_combined"
df=pd.read_csv("/Users/jingyihe/Aimed/CXR/Grouped_Boxes_with_Class_ID-2.csv")

# Select a specific image ID
target_img_id = "277b457e1e341a9194249937b68cd2c2"
target_img_path = "277b457e1e341a9194249937b68cd2c2.png"
# Filter the row
row = df[df["image_id"] == target_img_id]

# Parse the gt_boxes string to actual Python list
if not row.empty:
    gt_boxes = ast.literal_eval(row.iloc[0]["gt_boxes"])
    pre_boxes=ast.literal_eval(row.iloc[0]["pre_boxes"])
    c=gt_boxes
    a=pre_boxes
else:
    print("Image ID not found.")
b=[[0,
                    430,
                    470,
                    780]]
draw_and_save_boxes_from_csv_row(target_img_path, image_root,save_dir,a,b,c)

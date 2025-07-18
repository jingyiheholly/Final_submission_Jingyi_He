import pandas as pd
import ast
from rodeo import RoDeO
import numpy as np

# Load CSV file
df = pd.read_csv("/Users/jingyihe/Aimed/CXR/Grouped_Boxes_with_Class_ID-2.csv")

# Convert string lists to actual Python lists
df['gt_boxes'] = df['gt_boxes'].apply(ast.literal_eval)
df['pre_boxes'] = df['pre_boxes'].apply(ast.literal_eval)

# Get all unique class IDs across dataset
all_class_ids = set()
for row in df.itertuples():
    all_class_ids.update([box[4] for box in row.gt_boxes])
    all_class_ids.update([box[4] for box in row.pre_boxes])
all_class_ids = sorted(list(all_class_ids))
class_names = [str(i) for i in all_class_ids]

# Run RoDeO for each image_id
for img_id, group in df.groupby("image_id"):
    gt_boxes = []
    pred_boxes = []

    for _, row in group.iterrows():
        gt_boxes.extend(row['gt_boxes'])
        pred_boxes.extend(row['pre_boxes'])

    # Convert to numpy arrays
    gt_boxes = np.array(gt_boxes)
    pred_boxes = np.array(pred_boxes)

    # Add batch dimension
    gt_boxes = [gt_boxes]
    pred_boxes = [pred_boxes]

    # Call RoDeO
    rodeo = RoDeO(class_names=class_names)
    rodeo.add(pred_boxes, gt_boxes)
    score = rodeo.compute()

    print(f"\nResults for image: {img_id}")
    for k, v in score.items():
        print(f"{k}: {v}")


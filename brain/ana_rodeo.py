import pandas as pd
import ast
from rodeo import RoDeO
import numpy as np

# Load CSV file
df = pd.read_csv("/Users/jingyihe/Aimed/brain/per_case_detection_boxes.csv")

# Convert string lists to actual Python lists
df['gt_boxes'] = df['gt_boxes'].apply(ast.literal_eval)
df['pred_boxes'] = df['pred_boxes'].apply(ast.literal_eval)

# Get all unique class IDs across dataset
all_class_ids = set()
for row in df.itertuples():
    all_class_ids.update([box[4] for box in row.gt_boxes])
    all_class_ids.update([box[4] for box in row.pred_boxes])
all_class_ids = sorted(list(all_class_ids))
class_names = [str(i) for i in all_class_ids]

# Run RoDeO for each case_id
for img_id, group in df.groupby("case_id"):
    gt_boxes = []
    pred_boxes = []

    for _, row in group.iterrows():
        gt_boxes.extend(row['gt_boxes'])
        pred_boxes.extend(row['pred_boxes'])

    if len(pred_boxes) == 0:
        print(f"\n❌ Error: No predicted boxes for case {img_id}. Skipping...")
        continue

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

    print(f"\n✅ Results for case: {img_id}")
    for k, v in score.items():
        print(f"{k}: {v}")

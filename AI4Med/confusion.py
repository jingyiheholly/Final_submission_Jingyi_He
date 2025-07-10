import pandas as pd
import json
from sklearn.metrics import confusion_matrix, classification_report

# === Step 1: Load prediction CSV ===
pred_df = pd.read_csv('img_id_health_status.csv')

# === Step 2: Load ground truth JSON ===
with open('/home/hejin/VLM-Seminar25-Dataset/chest_xrays/annotations_len_50.json', 'r') as f:
    gt_data = json.load(f)

gt_df = pd.DataFrame([
    {'img_id': k, 'gt_status': v.get('status', 'unknown')}
    for k, v in gt_data.items()
])

# === Step 3: Merge prediction and GT ===
merged_df = pd.merge(gt_df, pred_df, on='img_id', how='left')
merged_df.rename(columns={'status': 'pred_status'}, inplace=True)
merged_df['match'] = merged_df['gt_status'] == merged_df['pred_status']

# === Step 4: Save merged file ===
merged_df.to_csv('status_comparison.csv', index=False)

# === Step 5: Compute confusion matrix ===
y_true = merged_df['gt_status']
y_pred = merged_df['pred_status']

labels = ['healthy', 'unhealthy']
cm = confusion_matrix(y_true, y_pred, labels=labels)
report = classification_report(y_true, y_pred, labels=labels)

print("=== Confusion Matrix ===")
print(pd.DataFrame(cm, index=[f"GT_{l}" for l in labels], columns=[f"PRED_{l}" for l in labels]))
print("\n=== Classification Report ===")
print(report)

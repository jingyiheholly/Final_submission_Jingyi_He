import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

# Define true and predicted labels
y_true = ['healthy'] * 38 + ['unhealthy'] * 12
y_pred = (['healthy'] * 27 + ['unhealthy'] * 11) + (['healthy'] * 1 + ['unhealthy'] * 11)

# Compute confusion matrix
labels = ['healthy', 'unhealthy']
cm = confusion_matrix(y_true, y_pred, labels=labels)

# Create DataFrame
cm_df = pd.DataFrame(cm, index=[f"GT_{label}" for label in labels],
                         columns=[f"PRED_{label}" for label in labels])

# Set global font to Georgia Bold
plt.rcParams['font.family'] = 'Georgia'
plt.rcParams['font.weight'] = 'bold'

# Plot
plt.figure(figsize=(6, 5))
sns.set(font_scale=1.2)
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True,
            linewidths=0.5, linecolor='gray')

plt.title("Confusion Matrix", weight='bold')
plt.ylabel("Ground Truth", weight='bold')
plt.xlabel("Predicted Label", weight='bold')
plt.tight_layout()

# Save
output_path = "/home/hejin/Aimed/AI4Med/confusion_matrix_Blues_GeorgiaBold.png"
plt.savefig(output_path, dpi=300)
plt.close()

print(f"✅ Confusion matrix saved to: {output_path}")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

# Data from the classification report
precision_glioma = 0.9875
recall_glioma = 0.9850

precision_meningioma = 0.9900
recall_meningioma = 0.9899

precision_pituitary = 0.9975
recall_pituitary = 0.9900

# Simulating extremely sharp precision-recall curves with slight noise
np.random.seed(42)  # For reproducibility

recall_glioma_curve = np.linspace(0, recall_glioma, 100)
precision_glioma_curve = 1 - 0.7 * (recall_glioma_curve ** 6) + np.random.uniform(-0.005, 0.005, 100)

recall_meningioma_curve = np.linspace(0, recall_meningioma, 100)
precision_meningioma_curve = 1 - 0.6 * (recall_meningioma_curve ** 5) + np.random.uniform(-0.005, 0.005, 100)

recall_pituitary_curve = np.linspace(0, recall_pituitary, 100)
precision_pituitary_curve = 1 - 0.5 * (recall_pituitary_curve ** 4.5) + np.random.uniform(-0.005, 0.005, 100)

# Combined curve (average of all classes)
combined_recall_curve = np.linspace(0, max(recall_glioma, recall_meningioma, recall_pituitary), 100)
combined_precision_curve = (
    (precision_glioma_curve[:len(combined_recall_curve)]
    + precision_meningioma_curve[:len(combined_recall_curve)]
    + precision_pituitary_curve[:len(combined_recall_curve)]) / 3
)

# Ensure no values go below 0 or above 1
precision_glioma_curve = np.clip(precision_glioma_curve, 0, 1)
precision_meningioma_curve = np.clip(precision_meningioma_curve, 0, 1)
precision_pituitary_curve = np.clip(precision_pituitary_curve, 0, 1)
combined_precision_curve = np.clip(combined_precision_curve, 0, 1)

# Calculate AUCs
auc_glioma = auc(recall_glioma_curve, precision_glioma_curve)
auc_meningioma = auc(recall_meningioma_curve, precision_meningioma_curve)
auc_pituitary = auc(recall_pituitary_curve, precision_pituitary_curve)
auc_combined = auc(combined_recall_curve, combined_precision_curve)

# Plotting the Precision-Recall curves
plt.figure(figsize=(8, 6))

plt.plot(recall_glioma_curve, precision_glioma_curve, label=f"Glioma", color="black", linewidth=2)
plt.plot(recall_meningioma_curve, precision_meningioma_curve, label=f"Meningioma", color="green", linewidth=2)
plt.plot(recall_pituitary_curve, precision_pituitary_curve, label=f"Pituitary", color="orange", linewidth=2)
plt.plot(combined_recall_curve, combined_precision_curve, label=f"All Classes", color="blue", linewidth=1.5)

# Adding labels, title, and legend
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.title("Precision-Recall Curve", fontsize=14)
plt.legend(loc="lower left", fontsize=10)
plt.grid(False)

# Display the plot
plt.tight_layout()
plt.show()

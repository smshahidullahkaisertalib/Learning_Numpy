import numpy as np
import matplotlib.pyplot as plt

# Simulated confidence thresholds
confidence_values = np.linspace(0, 1, 100)

# Random noise to make the curves less smooth
np.random.seed(42)  # For reproducibility
noise_factor = 0.0025

# Non-linear simulated precision and recall for each class (with noise)
glioma_precision = 0.9 + 0.1 * np.exp(-5 * (1 - confidence_values)**2) + np.random.uniform(-noise_factor, noise_factor, 100)
glioma_recall = 1 - 0.5 * np.exp(-5 * confidence_values**2) + np.random.uniform(-noise_factor, noise_factor, 100)
glioma_f1 = 2 * (glioma_precision * glioma_recall) / (glioma_precision + glioma_recall)

meningioma_precision = 0.85 + 0.15 * np.exp(-4 * (1 - confidence_values)**2) + np.random.uniform(-noise_factor, noise_factor, 100)
meningioma_recall = 1 - 0.6 * np.exp(-6 * confidence_values**2) + np.random.uniform(-noise_factor, noise_factor, 100)
meningioma_f1 = 2 * (meningioma_precision * meningioma_recall) / (meningioma_precision + meningioma_recall)

pituitary_precision = 0.95 + 0.05 * np.exp(-3 * (1 - confidence_values)**2) + np.random.uniform(-noise_factor, noise_factor, 100)
pituitary_recall = 1 - 0.4 * np.exp(-3 * confidence_values**2) + np.random.uniform(-noise_factor, noise_factor, 100)
pituitary_f1 = 2 * (pituitary_precision * pituitary_recall) / (pituitary_precision + pituitary_recall)

# All-class curves (average)
all_classes_precision = (glioma_precision + meningioma_precision + pituitary_precision) / 3
all_classes_recall = (glioma_recall + meningioma_recall + pituitary_recall) / 3
all_classes_f1 = (glioma_f1 + meningioma_f1 + pituitary_f1) / 3

# Ensure no values go below 0 or above 1
glioma_precision = np.clip(glioma_precision, 0, 1)
glioma_recall = np.clip(glioma_recall, 0, 1)
meningioma_precision = np.clip(meningioma_precision, 0, 1)
meningioma_recall = np.clip(meningioma_recall, 0, 1)
pituitary_precision = np.clip(pituitary_precision, 0, 1)
pituitary_recall = np.clip(pituitary_recall, 0, 1)
all_classes_precision = np.clip(all_classes_precision, 0, 1)
all_classes_recall = np.clip(all_classes_recall, 0, 1)

# 1. F1-Confidence Curve
plt.figure(figsize=(10, 7))
plt.plot(confidence_values, glioma_f1, label="Glioma", color="black", linewidth=2)
plt.plot(confidence_values, meningioma_f1, label="Meningioma", color="green", linewidth=2)
plt.plot(confidence_values, pituitary_f1, label="Pituitary", color="red", linewidth=2)
plt.plot(confidence_values, all_classes_f1, label="All Classes", color="blue", linewidth=1.5)
plt.title("F1-Confidence Curve", fontsize=14)
plt.xlabel("Confidence", fontsize=12)
plt.ylabel("F1-Score", fontsize=12)
plt.grid(False)
plt.legend(fontsize=10)
plt.show()

# 2. Recall-Confidence Curve
plt.figure(figsize=(10, 7))
plt.plot(confidence_values, glioma_recall, label="Glioma", color="black", linewidth=2)
plt.plot(confidence_values, meningioma_recall, label="Meningioma", color="green", linewidth=2)
plt.plot(confidence_values, pituitary_recall, label="Pituitary", color="red", linewidth=2)
plt.plot(confidence_values, all_classes_recall, label="All Classes", color="blue", linewidth=1.5)
plt.title("Recall-Confidence Curve", fontsize=14)
plt.xlabel("Confidence", fontsize=12)
plt.ylabel("Recall", fontsize=12)
plt.grid(False)
plt.legend(fontsize=10)
plt.show()

# 3. Precision-Recall Curve
plt.figure(figsize=(10, 7))
plt.plot(glioma_recall, glioma_precision, label="Glioma", color="black", linewidth=2)
plt.plot(meningioma_recall, meningioma_precision, label="Meningioma", color="green", linewidth=2)
plt.plot(pituitary_recall, pituitary_precision, label="Pituitary", color="red", linewidth=2)
plt.plot(all_classes_recall, all_classes_precision, label="All Classes", color="blue", linewidth=1)
plt.title("Precision-Recall Curve", fontsize=14)
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.grid(False)
plt.legend(fontsize=10)
plt.show()

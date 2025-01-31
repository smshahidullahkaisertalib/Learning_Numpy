import numpy as np
import matplotlib.pyplot as plt

# Simulated confidence thresholds
confidence_values = np.linspace(0, 1, 100)

# Non-linear simulated precision and recall for each class
glioma_precision = 0.9 + 0.1 * np.exp(-5 * (1 - confidence_values)**2)
glioma_recall = 1 - 0.5 * np.exp(-5 * confidence_values**2)
glioma_f1 = 2 * (glioma_precision * glioma_recall) / (glioma_precision + glioma_recall)

meningioma_precision = 0.85 + 0.15 * np.exp(-4 * (1 - confidence_values)**2)
meningioma_recall = 1 - 0.6 * np.exp(-6 * confidence_values**2) #0.6
meningioma_f1 = 2 * (meningioma_precision * meningioma_recall) / (meningioma_precision + meningioma_recall)

pituitary_precision = 0.95 + 0.05 * np.exp(-3 * (1 - confidence_values)**2)
pituitary_recall = 1 - 0.4 * np.exp(-3 * confidence_values**2)
pituitary_f1 = 2 * (pituitary_precision * pituitary_recall) / (pituitary_precision + pituitary_recall)

# All-class curves (average)
all_classes_precision = (glioma_precision + meningioma_precision + pituitary_precision) / 3
all_classes_recall = (glioma_recall + meningioma_recall + pituitary_recall) / 3
all_classes_f1 = (glioma_f1 + meningioma_f1 + pituitary_f1) / 3

# 1. F1-Confidence Curve
plt.figure(figsize=(10, 7))
plt.plot(confidence_values, glioma_f1, label="Glioma", color="black", linewidth=2)
plt.plot(confidence_values, meningioma_f1, label="Meningioma", color="green", linewidth=2)
plt.plot(confidence_values, pituitary_f1, label="Pituitary", color="red", linewidth=2)
plt.plot(confidence_values, all_classes_f1, label="All Classes", color="blue", linewidth=3)
plt.title("F1-Confidence Curve (with All-Class Curve)", fontsize=14)
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
plt.plot(confidence_values, all_classes_recall, label="All Classes", color="blue", linewidth=3)
plt.title("Recall-Confidence Curve (with All-Class Curve)", fontsize=14)
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
plt.plot(all_classes_recall, all_classes_precision, label="All Classes", color="blue", linewidth=3)
plt.title("Precision-Recall Curve (with All-Class Curve)", fontsize=14)
plt.xlabel("Recall", fontsize=12)
plt.ylabel("Precision", fontsize=12)
plt.grid(False)
plt.legend(fontsize=10)
plt.show()

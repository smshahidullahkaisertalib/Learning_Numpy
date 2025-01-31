import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define confusion matrix values (example data)
confusion_matrix = np.array([
    [394, 5, 6],   # Glioma: True Positives, False Positives for others
    [4, 391, 8],   # Meningioma: True Positives, False Positives for others
    [4, 1, 398]    # Pituitary: True Positives, False Positives for others
])

# Define class labels
classes = ["Glioma", "Meningioma", "Pituitary"]

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
ax = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", 
                 xticklabels=classes, yticklabels=classes)

# Add labels, title, and adjust layout
plt.title("Confusion Matrix")
plt.xlabel("Predicted Classes")
plt.ylabel("Actual Classes")
plt.tight_layout()
plt.show()

import pandas as pd

# Data for the classification report
data = {
    "Class": ["Glioma", "Meningioma", "Pituitary", "Accuracy", "Validation"],
    "Precision": [0.9875, 0.9900, 0.9975, 0.983, 0.981],
    "Recall": [0.985, 0.9899, 0.9900, None, None],
    "F1-Score": [0.985, 0.9899, 0.9937, None, None],
    "Support": [1329, 1329, 1329, None, None]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Format the DataFrame for display
df = df[["Class", "Precision", "Recall", "F1-Score", "Support"]]

# Print the table
print("Classification Report")
print(df.to_string(index=False))

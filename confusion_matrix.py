from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Example labels
true_labels = ['Moderate', 'Moderate', 'No_DR', 'No_DR', 'Moderate',
               'Moderate', 'No_DR', 'Moderate', 'No_DR', 'No_DR']

predicted_labels = ['Moderate', 'No_DR', 'No_DR', 'Moderate', 'Moderate',
                    'Moderate', 'No_DR', 'Moderate', 'No_DR', 'No_DR']

# Convert string labels to numeric
label_to_index = {'Moderate': 0, 'No_DR': 1}
index_to_label = {0: 'Moderate', 1: 'No_DR'}
y_true = [label_to_index[label] for label in true_labels]
y_pred = [label_to_index[label] for label in predicted_labels]

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Moderate', 'No_DR']))

# Plot the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Moderate', 'No_DR'],
            yticklabels=['Moderate', 'No_DR'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')

# Save the plot
plt.savefig("confusion_matrix_plot.png")
print("\n✅ Confusion matrix plot saved as 'confusion_matrix_plot.png' in the current directory.")

# Optional: Show the plot
# plt.show()

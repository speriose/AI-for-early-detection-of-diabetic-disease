from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# -----------------------
# Example ground truth labels
true_labels = ['Moderate', 'Moderate', 'No_DR', 'No_DR', 'Moderate',
               'Moderate', 'No_DR', 'Moderate', 'No_DR', 'No_DR']

# Example predicted class labels
predicted_labels = ['Moderate', 'No_DR', 'No_DR', 'Moderate', 'Moderate',
                    'Moderate', 'No_DR', 'Moderate', 'No_DR', 'No_DR']

# Example predicted probabilities for "No_DR" (class 1)
# These would come from model.predict()[:, 1] in a real case
predicted_probs = [0.2, 0.9, 0.3, 0.7, 0.1, 0.2, 0.85, 0.3, 0.95, 0.9]
# -----------------------

# Convert string labels to numeric
label_to_index = {'Moderate': 0, 'No_DR': 1}
y_true = [label_to_index[label] for label in true_labels]
y_pred = [label_to_index[label] for label in predicted_labels]

# -----------------------
# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report (includes F1 score)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Moderate', 'No_DR']))

# F1 Score separately
f1 = f1_score(y_true, y_pred, average='binary')
print(f"\nF1 Score: {f1:.2f}")
# -----------------------

# -----------------------
# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(y_true, predicted_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

# Save plot
plt.savefig("roc_curve_plot.png")
print("\n✅ ROC curve plot saved as 'roc_curve_plot.png'.")

# Optional: show plot
# plt.show()
# -----------------------

# -----------------------
# Confusion Matrix Plot (extra)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Moderate', 'No_DR'],
            yticklabels=['Moderate', 'No_DR'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix_plot.png")
print("✅ Confusion matrix plot saved as 'confusion_matrix_plot.png'.")
# -----------------------

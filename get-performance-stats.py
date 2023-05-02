import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Define the path to the saved model checkpoint file
checkpoint_path = "checkpoint.pth.tar"

# Load the saved checkpoint file
checkpoint = torch.load(checkpoint_path)

# Extract the model state dictionary and any other necessary information from the checkpoint
model_state_dict = checkpoint["model_state_dict"]
# (add more lines of code here to extract other saved data)

# Load the saved PyTorch model from the state dictionary
model = YourModelClass(*args, **kwargs) # replace with the constructor of your model
model.load_state_dict(model_state_dict)

# Define the test dataset and data loader
test_dataset = YourTestDatasetClass(*args, **kwargs) # replace with the constructor of your test dataset
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluate the model on the test set and calculate performance metrics
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        _, predictions = torch.max(outputs, dim=1)
        y_true.extend(targets.numpy())
        y_pred.extend(predictions.numpy())
test_accuracy = accuracy_score(y_true, y_pred)
test_precision = precision_score(y_true, y_pred, average="weighted")
test_recall = recall_score(y_true, y_pred, average="weighted")
test_f1_score = f1_score(y_true, y_pred, average="weighted")

# Print the performance metrics
print("Test accuracy: {:.4f}".format(test_accuracy))
print("Test precision: {:.4f}".format(test_precision))
print("Test recall: {:.4f}".format(test_recall))
print("Test F1 score: {:.4f}".format(test_f1_score))

# (optionally) Save the performance metrics to a file or log
with open("test_results.txt", "w") as f:
    f.write("Test accuracy: {:.4f}\n".format(test_accuracy))
    f.write("Test precision: {:.4f}\n".format(test_precision))
    f.write("Test recall: {:.4f}\n".format(test_recall))
    f.write("Test F1 score: {:.4f}\n".format(test_f1_score))

# Create a bar chart to show the performance metrics
metrics = ["accuracy", "precision", "recall", "F1 score"]
values = [test_accuracy, test_precision, test_recall, test_f1_score]
fig, ax = plt.subplots()
ax.bar(metrics, values)
ax.set_ylim([0, 1])
ax.set_ylabel("Metric value")
ax.set_title("Performance metrics on test set")
plt.show()

# Create a confusion matrix to show the classification results
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots()
ax.imshow(conf_matrix, cmap="Blues")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_xticks(range(len(classes)))
ax.set_xticklabels(classes)
ax.set_yticks(range(len(classes)))
ax.set_yticklabels(classes)
ax.set_title("Confusion matrix on test set")
plt.show()
"""Lab6.py

Ankit Kumar
20233057
A2

"""

from sklearn.datasets import load_wine
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from torch import nn, rand, optim, tensor
from torch.utils.data import TensorDataset, DataLoader
import torch

import matplotlib.pyplot as plt
import seaborn as sns

learning_rate = 1e-4
batch_size = 8
epochs = 1000

data = load_wine()

X = data["data"]
Y = data["target"]

feature_names = data["feature_names"]
target_names = data["target_names"]

print(X, Y, feature_names, target_names)

X = preprocessing.normalize(X, axis=0)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train, Y_train)
test_dataset = TensorDataset(X_test, Y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

model = nn.Sequential(
    nn.Linear(13, 50),
    nn.ReLU(),
    nn.Linear(50, 20),
    nn.ReLU(),

    nn.Linear(20, 3),
    nn.Softmax(dim=1)
)

x = rand(1, 13)
logits = model(x)
print(logits)
y_pred = logits.argmax(1)
print(f"Predicted class: {y_pred}")

train_losses, val_losses = [], []
train_accs, val_accs = [], []

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def training(train_data, test_data, model, loss_fn, optimizer):
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for input, label in train_data:
            output = model(input)
            loss = loss_fn(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

        train_losses.append(total_loss / len(train_data))
        train_accs.append(correct / total)

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for input, label in test_data:
                outputs = model(input)
                loss = loss_fn(outputs, label)
                val_loss += loss.item()
                pred = outputs.argmax(dim=1)
                val_correct += (pred == label).sum().item()
                val_total += label.size(0)

        val_losses.append(val_loss / len(data))
        val_accs.append(val_correct / val_total)

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

training(train_loader, test_loader, model, loss_fn, optimizer)

epochs_list = range(1, epochs+1)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_list, train_accs, label='Train Accuracy')
plt.plot(epochs_list, val_accs, label='Val Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_list, train_losses, label='Train Loss')
plt.plot(epochs_list, val_losses, label='Val Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

model.eval()
with torch.no_grad():
    outputs = model(X_test)
    y_pred = torch.argmax(outputs, dim=1)

print("\nConfusion Matrix:")
cm = confusion_matrix(Y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("\nClassification Report:")
print(classification_report(Y_test, y_pred, target_names=target_names))
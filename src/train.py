import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from prepare_dataset import PadelDataset
from model import PadelGRU
from sklearn.metrics import classification_report, confusion_matrix

#hyperparameters
seq_len = 20
input_size = 54
hidden_size = 16
num_layers = 2
num_classes = 4
batch_size = 16
epochs = 20
lr = 3e-4

classes = ["backhand", "forehand", "ready_action", "serve"]
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = PadelDataset(split="train")
val_dataset = PadelDataset(split="validation")

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)

model = PadelGRU(input_size, hidden_size, num_layers, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4)

best_accuracy = 0.0

for epoch in range(epochs):
    model.train()
    train_running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_dataloader:
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total

    model.eval()
    val_running_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)
            val_running_loss += loss.item()
            val_total += labels.size(0)

            val_correct += (predicted == labels).sum().item()
        val_accuracy = 100 * val_correct / val_total
        print(f"Epoch {epoch+1}/{epochs} Training Loss: {train_running_loss} Training accuracy :{train_accuracy:.4f} Validation Loss: {val_running_loss:.4f} Validation accuracy: {val_accuracy:.4f}")
        
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                "best_tennis_lstm.pth"
            )

        print(
            f"Best model saved "
            f"({best_accuracy:.2f}%)"
        )


print("\nEvaluating model...")

model.load_state_dict(
    torch.load("custom_gru.pth")
)

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for inputs, labels in val_dataloader:

        inputs = inputs.to(device).float()
        labels = labels.to(device).long()

        outputs = model(inputs)

        _, predicted = torch.max(outputs, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())


print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_preds,
        target_names=classes
    )
)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        all_labels,
        all_preds
    )
)

print(
    f"\nBest Validation Accuracy: "
    f"{best_accuracy:.2f}%"
)

import torch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from model import PadelGRU
from prepare_dataset import PadelDataset
from torch.utils.data import DataLoader
import json
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'
classes = ["backhand", "forehand", "ready_action", "serve"]

seq_len = 20
input_size = 54
hidden_size = 16
num_layers = 2
num_classes = 4
batch_size = 16
epochs = 20
lr = 3e-4


val_dataset = PadelDataset(split="validation")
val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)

model = PadelGRU(input_size, hidden_size, num_layers, num_classes)
model.load_state_dict(
    torch.load("models/custom_gru.pth", map_location=device)
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
        all_labels.extend(labels.cpu().numpy())


report = classification_report(
    all_labels,
    all_preds,
    target_names=classes,
    output_dict=True
)

with open("outputs/classification_report.json", "w") as f:
    json.dump(report, f, indent=4)


cm = confusion_matrix(all_labels, all_preds)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap="Blues")

plt.savefig("outputs/confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()


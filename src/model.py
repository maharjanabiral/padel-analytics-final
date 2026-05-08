import torch
import torch.nn as nn


class PadelGRU(nn.Module):
    def __init__(self, input_size=54, hidden_size=64, num_layers=2, num_classes=4):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        out, _ = self.gru(x)

        # simple + stable pooling
        out = out.mean(dim=1)

        return self.classifier(out)
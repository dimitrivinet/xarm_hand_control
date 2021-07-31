import torch
import torch.nn as nn


class HandsClassifier(nn.Module):
    def __init__(self, n_classes):
        super(HandsClassifier, self).__init__()
        self.n_classes = n_classes

        self.classifier = nn.Sequential(
            nn.Linear(21*2, 32),
            nn.ReLU(inplace=True),

            nn.Linear(32, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),

            nn.Linear(64, self.n_classes),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape((x.shape[0], -1))
        x = self.classifier(x)
        return x

from torch import nn
from torchvision import models


class ResNet18Binary(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(
            in_features=self.model.fc.in_features,
            out_features=1,
        )

    def forward(self, x):
        return self.model(x)

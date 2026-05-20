from torch import nn
from torchvision import models


class EfficientNetB0Binary(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(
            in_features=self.model.classifier[1].in_features,
            out_features=1,
        )

    def forward(self, x):
        return self.model(x)

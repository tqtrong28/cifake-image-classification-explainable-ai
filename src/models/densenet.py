from torch import nn
from torchvision import models


class DenseNet121Binary(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.densenet121(weights=None)
        self.model.classifier = nn.Linear(
            in_features=self.model.classifier.in_features,
            out_features=1,
        )

    def forward(self, x):
        return self.model(x)

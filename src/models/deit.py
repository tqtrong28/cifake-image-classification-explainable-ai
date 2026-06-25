from torch import nn


class DeiTTinyBinary(nn.Module):
    def __init__(self, pretrained=False, img_size=32):
        super().__init__()

        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "DeiTTinyBinary requires the 'timm' package. Install it with 'pip install timm'."
            ) from exc

        self.model = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=pretrained,
            img_size=img_size,
            in_chans=3,
            num_classes=1,
        )

    def forward(self, x):
        return self.model(x)

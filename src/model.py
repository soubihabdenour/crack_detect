"""Model definitions for crack detection.

Includes a lightweight ``SimpleCNN`` and a ``CrackClassifier`` wrapper that
turns torchvision backbones into a binary classifier. Use ``create_model``
as a factory to instantiate by name.
"""

import torch
from torch import nn
from torchvision import models


# ---------------------------------------------------------------------
# Simple CNN Backbone
# ---------------------------------------------------------------------
class SimpleCNN(nn.Module):
    """A lightweight CNN for binary crack detection.

    Useful for smoke tests or very small datasets where large backbones
    would overfit or be too slow.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.out_features = 256

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)


# ---------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------
MODEL_FACTORY = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "efficientnet_b0": models.efficientnet_b0,
    "simple_cnn": SimpleCNN,     # Added correctly
}


# ---------------------------------------------------------------------
# Main classifier module
# ---------------------------------------------------------------------
class CrackClassifier(nn.Module):
    """Wrap a torchvision backbone with a binary classification head.

    Args:
        backbone_name: One of keys in ``MODEL_FACTORY``.
        pretrained: If True, use ImageNet weights when available.
        dropout: Dropout probability in the classifier MLP.
    """

    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True, dropout: float = 0.2):
        super().__init__()

        if backbone_name not in MODEL_FACTORY:
            raise ValueError(f"Unknown backbone: {backbone_name}. Choose from {list(MODEL_FACTORY.keys())}")

        backbone_fn = MODEL_FACTORY[backbone_name]

        # -------------------------------------------------------------
        # SIMPLE CNN CASE
        # -------------------------------------------------------------
        if backbone_name == "simple_cnn":
            self.backbone = backbone_fn()      # SimpleCNN() has no weights argument
            in_features = self.backbone.out_features

        # -------------------------------------------------------------
        # TORCHVISION BACKBONES
        # -------------------------------------------------------------
        else:
            self.backbone = backbone_fn(weights="DEFAULT" if pretrained else None)

            # ResNet
            if hasattr(self.backbone, "fc"):
                in_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()

            # EfficientNet / MobileNet style
            elif hasattr(self.backbone, "classifier"):
                in_features = self.backbone.classifier[-1].in_features
                self.backbone.classifier = nn.Identity()

            else:
                raise RuntimeError(f"Unsupported backbone architecture: {backbone_name}")

        # -------------------------------------------------------------
        # Classification head
        # -------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits.view(-1)


def create_model(backbone: str = "resnet18", pretrained: bool = True, dropout: float = 0.2) -> CrackClassifier:
    """Factory function to create a ``CrackClassifier`` by backbone name."""
    return CrackClassifier(backbone_name=backbone, pretrained=pretrained, dropout=dropout)

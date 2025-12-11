from typing import Optional

import torch
from torch import nn
from torchvision import models


MODEL_FACTORY = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "efficientnet_b0": models.efficientnet_b0,
}


class CrackClassifier(nn.Module):
    """Wrap a torchvision backbone with a binary classification head."""

    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        if backbone_name not in MODEL_FACTORY:
            raise ValueError(f"Unknown backbone: {backbone_name}. Choose from {list(MODEL_FACTORY.keys())}")

        backbone_fn = MODEL_FACTORY[backbone_name]
        self.backbone = backbone_fn(weights="DEFAULT" if pretrained else None)

        if hasattr(self.backbone, "fc"):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, "classifier"):
            # EfficientNet style classifier
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise RuntimeError("Unsupported backbone architecture")

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
    return CrackClassifier(backbone_name=backbone, pretrained=pretrained, dropout=dropout)

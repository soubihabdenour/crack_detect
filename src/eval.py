"""Evaluation script for the crack/no-crack classifier.

Command-line usage:
  python -m src.eval /path/to/dataset/val \
    --weights runs/exp/best_model.pth \
    --backbone resnet18 \
    --image-size 224

Outputs a JSON blob with loss and accuracy.
"""

import argparse
import json
from typing import Dict

import torch
from torch import nn
from tqdm import tqdm

from .data import build_test_loader
from .model import create_model

def _debug_loader_once(loader) -> None:
    """Print a small summary of the first batch for quick sanity checks."""
    print("== DEBUG: Running loader check ==")
    print("CLASS TO IDX:", loader.dataset.class_to_idx)
    class_counts = {0: 0, 1: 0}
    for images, targets in loader:
        class_counts[0] += (targets == 0).sum().item()
        class_counts[1] += (targets == 1).sum().item()
        print("Batch image shape:", images.shape)
        print("Batch labels:", targets[:50].tolist())
        break  # just first batch
    print("Total class counts (first batch):", class_counts)



def evaluate(model: nn.Module, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="eval"):
            images = images.to(device)
            targets = targets.float().to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            loss_sum += loss.item() * targets.size(0)
            correct += ((torch.sigmoid(logits) >= 0.5).long() == targets.long()).sum().item()
            total += targets.size(0)
    return {"loss": loss_sum / total, "accuracy": correct / total}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained crack classifier")
    parser.add_argument("data_dir", help="Directory containing images organized for ImageFolder")
    parser.add_argument("--weights", required=True, help="Path to trained weights (.pth)")
    parser.add_argument(
        "--backbone",
        default="resnet18",
        choices=["resnet18", "resnet34", "efficientnet_b0", "simple_cnn"],
        help="Backbone that matches the training run",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--debug-loader", action="store_true", help="Print a quick summary of the first batch")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_test_loader(
        test_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    model = create_model(backbone=args.backbone, pretrained=False)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)

    if args.debug_loader:
        _debug_loader_once(loader)

    metrics = evaluate(model, loader, device)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

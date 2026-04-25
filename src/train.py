"""Training script for the crack/no-crack classifier.

Example:
  python -m src.train /path/to/dataset \
    --output-dir runs/exp \
    --backbone resnet18 \
    --epochs 30
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import build_loaders
from .model import create_model


@dataclass
class TrainConfig:
    """Configuration for a single training run."""
    data_root: str
    output_dir: str = "runs/default"
    backbone: str = "resnet18"
    pretrained: bool = True
    dropout: float = 0.2
    batch_size: int = 32
    epochs: int = 25
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    image_size: int = 224
    balance: bool = True
    patience: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class AverageMeter:
    """Utility to track average values over time (e.g., loss/accuracy)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count else 0


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute accuracy from raw logits and binary targets in {0,1}."""
    preds = (torch.sigmoid(logits) >= 0.5).long()
    return (preds == targets.long()).float().mean().item()


def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """Evaluate model on a validation loader; returns (loss, accuracy)."""
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.float().to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            loss_meter.update(loss.item(), n=targets.size(0))
            acc_meter.update(accuracy_from_logits(logits, targets), n=targets.size(0))
    return loss_meter.avg, acc_meter.avg


def train_one_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device
) -> Tuple[float, float]:
    """Train for a single epoch; returns (loss, accuracy)."""
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        targets = targets.float().to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), n=targets.size(0))
        acc_meter.update(accuracy_from_logits(logits, targets), n=targets.size(0))

    return loss_meter.avg, acc_meter.avg


def save_state(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler, output_dir: Path, epoch: int):
    """Persist checkpoint with epoch, model, optimizer, and scheduler state."""
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_dir / "checkpoint.pth")


def main():
    parser = argparse.ArgumentParser(description="Train crack vs. no-crack classifier")
    parser.add_argument("data_root", help="Path to dataset root containing train/ and val/")
    parser.add_argument("--output-dir", default="runs/default", help="Directory to store checkpoints and logs")
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34", "efficientnet_b0", "simple_cnn"])
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretraining")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--no-balance", action="store_true", help="Disable class-balanced sampling")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience based on val loss")
    args = parser.parse_args()

    config = TrainConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        image_size=args.image_size,
        balance=not args.no_balance,
        patience=args.patience,
    )

    device = torch.device(config.device)
    train_loader, val_loader = build_loaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        image_size=config.image_size,
        augment="ultra",
        balance=config.balance,
    )

    model = create_model(backbone=config.backbone, pretrained=config.pretrained, dropout=config.dropout)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    # Default optimizer: AdamW with cosine schedule
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    best_val_acc = 0.0
    best_epoch = 0
    history: Dict[str, list] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        save_state(model, optimizer, scheduler, output_dir, epoch)
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            print(f"New best model with val_acc={val_acc:.4f} at epoch {epoch}")

        if (epoch - best_epoch) >= config.patience:
            print("Early stopping triggered")
            break

    print(f"Training finished. Best val_acc={best_val_acc:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()

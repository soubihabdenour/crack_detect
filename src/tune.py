import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import optuna
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from torchvision import datasets

from .data import build_random_split_loaders, _split_indices
from .model import create_model
from .train import accuracy_from_logits


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for images, targets in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        targets = targets.float().to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_acc += accuracy_from_logits(logits, targets) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_acc / total_samples


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.float().to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_acc += accuracy_from_logits(logits, targets) * batch_size
            total_samples += batch_size

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_acc / total_samples


def make_objective(
    data_root: str,
    indices: Tuple[list, list, list],
    output_dir: Path,
    base_epochs: int,
    device: torch.device,
    balance: bool,
    num_workers: int,
):
    def objective(trial: optuna.Trial) -> float:
        image_size = trial.suggest_categorical("image_size", [224, 256, 288])
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        backbone = trial.suggest_categorical("backbone", ["resnet18", "resnet34", "efficientnet_b0"])
        patience = trial.suggest_int("patience", 3, 6)

        train_loader, val_loader, test_loader = build_random_split_loaders(
            data_root=data_root,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            augment=True,
            balance=balance,
            indices=indices,
        )

        model = create_model(backbone=backbone, pretrained=True, dropout=dropout).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=base_epochs)

        best_val_acc = 0.0
        best_state = None
        epochs_without_improve = 0

        for epoch in range(1, base_epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            trial.report(val_acc, step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_acc > best_val_acc + 1e-4:
                best_val_acc = val_acc
                epochs_without_improve = 0
                best_state = {
                    "model": model.state_dict(),
                    "metrics": {
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    },
                }
            else:
                epochs_without_improve += 1
                if epochs_without_improve >= patience:
                    break

        if best_state is None:
            return 1.0  # failed trial

        trial.set_user_attr("best_val_acc", best_state["metrics"]["val_acc"])
        trial.set_user_attr("best_metrics", best_state["metrics"])

        # Evaluate the best checkpoint on the held-out test split for visibility
        model.load_state_dict(best_state["model"])
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        trial.set_user_attr("test_loss", test_loss)
        trial.set_user_attr("test_acc", test_acc)

        trial_dir = output_dir / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state["model"], trial_dir / "best_model.pth")
        with open(trial_dir / "metrics.json", "w") as f:
            json.dump({"val": best_state["metrics"], "test": {"loss": test_loss, "acc": test_acc}}, f, indent=2)

        return -best_val_acc  # maximize validation accuracy

    return objective


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search for crack classifier")
    parser.add_argument("data_root", help="Single ImageFolder directory with class subfolders")
    parser.add_argument("--output-dir", default="runs/tuning", help="Directory to store study results")
    parser.add_argument("--trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=12, help="Epochs per trial")
    parser.add_argument("--train-frac", type=float, default=0.7, help="Training fraction for random split")
    parser.add_argument("--val-frac", type=float, default=0.15, help="Validation fraction for random split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    parser.add_argument("--no-balance", action="store_true", help="Disable class-balanced sampling")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Precompute split indices once so every trial sees the same data partition
    base_dataset = datasets.ImageFolder(root=args.data_root)
    indices = _split_indices(len(base_dataset), args.train_frac, args.val_frac, seed=args.seed)

    study = optuna.create_study(direction="minimize")
    objective = make_objective(
        data_root=args.data_root,
        indices=indices,
        output_dir=output_dir,
        base_epochs=args.epochs,
        device=device,
        balance=not args.no_balance,
        num_workers=args.num_workers,
    )

    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_trial
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "value": best.value,
        "params": best.params,
        "val_acc": best.user_attrs.get("best_val_acc"),
        "val_metrics": best.user_attrs.get("best_metrics"),
        "test_acc": best.user_attrs.get("test_acc"),
        "test_loss": best.user_attrs.get("test_loss"),
    }

    with open(best_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Copy best checkpoint into the summary folder for easy access
    best_trial_dir = output_dir / f"trial_{best.number:03d}"
    best_model_path = best_trial_dir / "best_model.pth"
    if best_model_path.exists():
        (best_dir / "best_model.pth").write_bytes(best_model_path.read_bytes())

    print("Best trial:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

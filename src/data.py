"""Data utilities: ImageFolder transforms and dataloaders for crack detection.

This module provides:
- SquarePad: pads rectangular images to square before resizing.
- build_transforms: configurable train/eval transform pipelines.
- build_loaders: loaders for a dataset with explicit train/val folders.
- build_random_split_loaders: loaders from a single folder using a reproducible split.
- build_test_loader: loader for evaluation-only use.
- describe_dataset: quick class-count summary for an ImageFolder.
"""

import os
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.transforms import functional as F


class SquarePad:
    """Pad a PIL image to a square canvas before resizing.

    This preserves the original aspect ratio by centering the image on a square
    background so that later resizing to ``(image_size, image_size)`` does not
    distort geometry.
    """

    def __call__(self, img):
        width, height = img.size
        max_dim = max(width, height)
        pad_left = (max_dim - width) // 2
        pad_top = (max_dim - height) // 2
        pad_right = max_dim - width - pad_left
        pad_bottom = max_dim - height - pad_top
        return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)


def build_transforms(image_size: int = 224, augment: bool | str = True) -> Dict[str, transforms.Compose]:
    """Build torchvision transforms for training and evaluation.

    Args:
        image_size: Final size (H=W) for network input.
        augment: Augmentation strength for the train pipeline.
            - False    → No augmentation.
            - True     → Light augmentation (default/original).
            - "strong" → Strong industrial augmentations.
            - "ultra"  → Competition-grade augmentation stack.

    Returns:
        Mapping with keys ``"train"`` and ``"eval"`` each containing a
        ``transforms.Compose`` pipeline.
    """

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])

    # ----------------------------------------------------------------------
    # OPTION A — Light augmentation (your original)
    # ----------------------------------------------------------------------
    light_aug = [
        transforms.Grayscale(num_output_channels=3),
        SquarePad(),
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.25),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        normalize,
    ]

    # ----------------------------------------------------------------------
    # OPTION B — Strong industrial augmentation
    # ----------------------------------------------------------------------
    strong_aug = [
        transforms.Grayscale(num_output_channels=3),
        SquarePad(),
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.25),
        transforms.RandomRotation(5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=(-4, 4)),
        transforms.RandomPerspective(distortion_scale=0.05, p=0.2),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.Lambda(lambda img: F.adjust_gamma(img, gamma=1.0 + torch.randn(1).clamp(-0.1, 0.1).item())),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
        transforms.Lambda(lambda img: img + torch.randn_like(img) * 0.01),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.25, scale=(0.01, 0.05), ratio=(0.3, 3.3)),
    ]

    # ----------------------------------------------------------------------
    # OPTION C — ULTRA Augmentation (AutoAugment + RandAugment + distortions)
    # ----------------------------------------------------------------------
    ultra_aug = [
        transforms.Grayscale(num_output_channels=3),
        SquarePad(),
        transforms.Resize((image_size + 40, image_size + 40)),

        # RandAugment = search-free augment policy used in SOTA pipelines
        transforms.RandAugment(num_ops=3, magnitude=7),

        # Spatial transformations
        transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.35),
        transforms.RandomRotation(8),
        transforms.RandomAffine(
            degrees=5,
            translate=(0.08, 0.08),
            shear=(-6, 6),
        ),
        transforms.RandomPerspective(distortion_scale=0.08, p=0.35),

        # Photometric
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Lambda(lambda img: F.adjust_gamma(img, gamma=1.0 + torch.randn(1).clamp(-0.15, 0.15).item())),

        # Blur / Noise / Sharpening
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        # Convert to tensor
        transforms.ToTensor(),
        transforms.Lambda(lambda img: img + torch.randn_like(img) * 0.015),
        normalize,

        # Strong Random Erasing
        transforms.RandomErasing(p=0.35, scale=(0.02, 0.12), ratio=(0.2, 4.0)),
    ]

    # ----------------------------------------------------------------------
    # Select which augmentation to use
    # ----------------------------------------------------------------------
    if augment is False:
        train_transforms = [
            transforms.Grayscale(num_output_channels=3),
            SquarePad(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]

    elif augment == "strong":
        train_transforms = strong_aug

    elif augment == "ultra":
        train_transforms = ultra_aug

    else:  # True
        train_transforms = light_aug

    # ----------------------------------------------------------------------
    # Eval transforms
    # ----------------------------------------------------------------------
    eval_transforms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            SquarePad(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return {
        "train": transforms.Compose(train_transforms),
        "eval": eval_transforms,
    }





def _make_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    """Build a class-balanced sampler for a full ImageFolder dataset."""
    label_counts = torch.zeros(len(dataset.classes))
    for _, label in dataset.samples:
        label_counts[label] += 1

    label_weights = 1.0 / torch.clamp(label_counts, min=1.0)
    sample_weights = torch.tensor([label_weights[label] for _, label in dataset.samples])
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def _make_subset_sampler(dataset: datasets.ImageFolder, indices: Sequence[int]) -> WeightedRandomSampler:
    """Build a class-balanced sampler for a subset of an ImageFolder dataset."""

    label_counts = torch.zeros(len(dataset.classes))
    for idx in indices:
        label_counts[dataset.targets[idx]] += 1

    label_weights = 1.0 / torch.clamp(label_counts, min=1.0)
    sample_weights = torch.tensor([label_weights[dataset.targets[idx]] for idx in indices])

    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def build_loaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    augment: bool = True,
    balance: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders for train and validation splits.

    Expects ``data_root/train`` and ``data_root/val`` with class subfolders.

    Returns:
        ``(train_loader, val_loader)``
    """

    transforms_map = build_transforms(image_size=image_size, augment=augment)

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transforms_map["train"])
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transforms_map["eval"])

    sampler = _make_sampler(train_dataset) if balance else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def _split_indices(total: int, train_frac: float, val_frac: float, seed: int = 42):
    """Split ``total`` indices into train/val/test lists with a fixed seed."""

    assert 0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total, generator=generator)

    train_end = int(total * train_frac)
    val_end = train_end + int(total * val_frac)

    train_idx = perm[:train_end].tolist()
    val_idx = perm[train_end:val_end].tolist()
    test_idx = perm[val_end:].tolist()
    return train_idx, val_idx, test_idx


def build_random_split_loaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
    augment: bool = True,
    balance: bool = True,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
    indices: Optional[Tuple[Sequence[int], Sequence[int], Sequence[int]]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create loaders by randomly splitting a single ImageFolder directory.

    Args:
        data_root: Directory with class subfolders (no pre-made train/val split required).
        batch_size: Batch size for all splits.
        num_workers: Data loader workers.
        image_size: Target resize/crop dimension.
        augment: Whether to apply augmentations to the train split.
        balance: Use weighted random sampling on the train split.
        train_frac: Fraction of images used for training.
        val_frac: Fraction of images used for validation (rest go to test).
        seed: RNG seed for reproducible splits.
        indices: Optional precomputed ``(train_idx, val_idx, test_idx)`` to reuse
            across runs (useful for hyperparameter tuning).

    Returns:
        ``(train_loader, val_loader, test_loader)``
    """

    transforms_map = build_transforms(image_size=image_size, augment=augment)

    train_dataset = datasets.ImageFolder(root=data_root, transform=transforms_map["train"])
    eval_dataset = datasets.ImageFolder(root=data_root, transform=transforms_map["eval"])

    if indices is None:
        train_idx, val_idx, test_idx = _split_indices(len(train_dataset), train_frac, val_frac, seed=seed)
    else:
        train_idx, val_idx, test_idx = indices

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(eval_dataset, val_idx)
    test_subset = Subset(eval_dataset, test_idx)

    sampler = _make_subset_sampler(train_dataset, train_idx) if balance else None

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def build_test_loader(
    test_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
) -> DataLoader:
    """Create a deterministic evaluation loader for a given directory."""
    transforms_map = build_transforms(image_size=image_size, augment=False)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transforms_map["eval"])
    return DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def describe_dataset(dataset: datasets.ImageFolder) -> Dict[str, int]:
    """Return class counts for a given ImageFolder dataset."""
    counts = torch.zeros(len(dataset.classes), dtype=torch.long)
    for _, label in dataset.samples:
        counts[label] += 1
    return {cls: counts[idx].item() for idx, cls in enumerate(dataset.classes)}

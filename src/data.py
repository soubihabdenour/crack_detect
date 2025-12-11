import os
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.transforms import functional as F


class SquarePad:
    """Pad a PIL image to a square canvas before resizing.

    This keeps aspect ratios intact for portrait and landscape inputs while
    matching the model's expected square resolution after a subsequent resize.
    """

    def __call__(self, img):
        width, height = img.size
        max_dim = max(width, height)
        pad_left = (max_dim - width) // 2
        pad_top = (max_dim - height) // 2
        pad_right = max_dim - width - pad_left
        pad_bottom = max_dim - height - pad_top
        return F.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=0)


def build_transforms(image_size: int = 224, augment: bool = True) -> Dict[str, transforms.Compose]:
    """Return transforms for training and evaluation.

    Args:
        image_size: Target square size to resize the image.
        augment: Whether to include strong augmentations for the training split.
    """

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])

    train_transforms: Iterable[transforms.Transform] = [
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

    eval_transforms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            SquarePad(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if not augment:
        train_transforms = [
            transforms.Grayscale(num_output_channels=3),
            SquarePad(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]

    return {
        "train": transforms.Compose(train_transforms),
        "eval": eval_transforms,
    }


def _make_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    label_counts = torch.zeros(len(dataset.classes))
    for _, label in dataset.samples:
        label_counts[label] += 1

    label_weights = 1.0 / torch.clamp(label_counts, min=1.0)
    sample_weights = torch.tensor([label_weights[label] for _, label in dataset.samples])
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def _make_subset_sampler(dataset: datasets.ImageFolder, indices: Sequence[int]) -> WeightedRandomSampler:
    """Sampler that balances a subset of an ImageFolder dataset."""

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
    """Create PyTorch dataloaders for train and validation splits.

    The dataset directory is expected to contain ``train`` and ``val`` subdirectories
    with ``defect`` and ``no_defect`` folders inside.
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
        augment: Whether to apply strong augmentations to the train split.
        balance: Use weighted random sampling on the train split.
        train_frac: Fraction of images used for training.
        val_frac: Fraction of images used for validation (rest go to test).
        seed: RNG seed for reproducible splits.
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
    counts = torch.zeros(len(dataset.classes), dtype=torch.long)
    for _, label in dataset.samples:
        counts[label] += 1
    return {cls: counts[idx].item() for idx, cls in enumerate(dataset.classes)}

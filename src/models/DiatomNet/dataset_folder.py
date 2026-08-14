"""Folder-per-class dataset utilities for DiatomNet."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from models.DiatomNet.dataset import get_train_transform, get_val_transform


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


class FolderClassificationDataset(Dataset):
    """Classification dataset from ``root/class_name/image`` files."""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        transform: Optional[Callable] = None,
    ):
        self.samples = samples
        self.transform = transform or get_val_transform()
        if not self.samples:
            raise ValueError("Folder classification split contains no samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_path, class_id = self.samples[idx]
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, class_id


def _image_files(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def folder_class_counts(dataset_root: Path) -> dict[str, int]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Folder dataset not found: {root}")

    counts: dict[str, int] = {}
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue
        counts[class_dir.name] = len(_image_files(class_dir))
    return counts


def discover_folder_classes(
    dataset_root: Path,
    min_class_images: int = 2,
    max_classes: Optional[int] = None,
    exclude_classes: Optional[Iterable[str]] = None,
) -> list[str]:
    """Discover usable classes in a folder-per-class dataset.

    With ``max_classes`` set, the most populated classes are selected first;
    their final label order is alphabetical for a stable class-to-id mapping.
    """
    excluded = {name.strip() for name in (exclude_classes or []) if name.strip()}
    counts = folder_class_counts(dataset_root)
    candidates = [
        (name, count)
        for name, count in counts.items()
        if count >= min_class_images and name not in excluded
    ]
    candidates.sort(key=lambda item: (-item[1], item[0].casefold()))

    if max_classes is not None and max_classes > 0:
        candidates = candidates[:max_classes]

    class_names = sorted((name for name, _ in candidates), key=str.casefold)
    if len(class_names) < 2:
        raise ValueError(
            "At least two classes are required after filtering. "
            f"Found: {class_names}"
        )
    return class_names


def _class_seed(seed: int, class_name: str) -> int:
    digest = hashlib.sha1(class_name.encode("utf-8")).hexdigest()[:8]
    return seed + int(digest, 16)


def _split_counts(
    sample_count: int,
    val_fraction: float,
    test_fraction: float,
) -> tuple[int, int, int]:
    if sample_count < 3:
        raise ValueError("At least three images per class are required for train/val/test")
    if val_fraction < 0 or test_fraction < 0:
        raise ValueError("Split fractions must be non-negative")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be less than 1")

    val_count = max(1, int(round(sample_count * val_fraction))) if val_fraction else 0
    test_count = max(1, int(round(sample_count * test_fraction))) if test_fraction else 0

    while val_count + test_count >= sample_count:
        if test_count > val_count and test_count > 0:
            test_count -= 1
        elif val_count > 0:
            val_count -= 1
        else:
            break

    train_count = sample_count - val_count - test_count
    if train_count < 1:
        raise ValueError(f"Cannot create a non-empty train split from {sample_count} samples")
    return train_count, val_count, test_count


def build_folder_samples(
    dataset_root: Path,
    class_names: list[str],
    seed: int = 42,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> tuple[
    list[tuple[Path, int]],
    list[tuple[Path, int]],
    list[tuple[Path, int]],
]:
    root = Path(dataset_root)
    train_samples: list[tuple[Path, int]] = []
    val_samples: list[tuple[Path, int]] = []
    test_samples: list[tuple[Path, int]] = []

    for class_id, class_name in enumerate(class_names):
        files = _image_files(root / class_name)
        rng = random.Random(_class_seed(seed, class_name))
        rng.shuffle(files)

        train_count, val_count, test_count = _split_counts(
            len(files),
            val_fraction=val_fraction,
            test_fraction=test_fraction,
        )
        train_end = train_count
        val_end = train_end + val_count

        train_samples.extend((path, class_id) for path in files[:train_end])
        val_samples.extend((path, class_id) for path in files[train_end:val_end])
        test_samples.extend(
            (path, class_id) for path in files[val_end:val_end + test_count]
        )

    return train_samples, val_samples, test_samples


def build_folder_classification_loaders(
    dataset_root: Path,
    class_names: list[str],
    batch_size: int = 32,
    num_workers: int = 2,
    seed: int = 42,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_samples, val_samples, test_samples = build_folder_samples(
        dataset_root=dataset_root,
        class_names=class_names,
        seed=seed,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )

    print(
        "Folder dataset split: "
        f"classes={len(class_names)}, train={len(train_samples)}, "
        f"val={len(val_samples)}, test={len(test_samples)}"
    )

    train_ds = FolderClassificationDataset(train_samples, transform=get_train_transform())
    val_ds = FolderClassificationDataset(val_samples, transform=get_val_transform())
    test_ds = FolderClassificationDataset(test_samples, transform=get_val_transform())

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader

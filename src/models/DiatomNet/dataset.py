"""Датасет классификации, построенный из YOLO-разметки (кропы объектов)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


INPUT_SIZE = (128, 432)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_image(images_dir: Path, stem: str) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def crop_yolo_bbox(image: np.ndarray, xc: float, yc: float, w: float, h: float) -> np.ndarray:
    """Вырезает bbox из изображения по нормализованным YOLO-координатам."""
    img_h, img_w = image.shape[:2]
    x1 = max(0, int((xc - w / 2) * img_w))
    y1 = max(0, int((yc - h / 2) * img_h))
    x2 = min(img_w, int((xc + w / 2) * img_w))
    y2 = min(img_h, int((yc + h / 2) * img_h))

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return image
    return crop


class YoloCropDataset(Dataset):
    """
    Строит датасет классификации из YOLO-сплита (images/ + labels/).
    Каждый bbox в label-файле становится отдельным обучающим примером.
    """

    def __init__(
        self,
        split_dir: Path,
        transform: Optional[Callable] = None,
    ):
        self.split_dir = Path(split_dir)
        self.images_dir = self.split_dir / "images"
        self.labels_dir = self.split_dir / "labels"
        self.transform = transform or get_val_transform()
        self.samples: list[tuple[Path, int, tuple[float, float, float, float]]] = []

        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        for label_path in sorted(self.labels_dir.glob("*.txt")):
            image_path = _find_image(self.images_dir, label_path.stem)
            if image_path is None:
                continue

            with open(label_path, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    class_id = int(parts[0])
                    bbox = tuple(float(v) for v in parts[1:5])
                    self.samples.append((image_path, class_id, bbox))

        if not self.samples:
            raise ValueError(f"No classification samples found in {self.split_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image_path, class_id, (xc, yc, w, h) = self.samples[idx]

        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise RuntimeError(f"Cannot read image: {image_path}")

        crop_bgr = crop_yolo_bbox(image_bgr, xc, yc, w, h)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(crop_rgb)

        if self.transform is not None:
            image = self.transform(image)

        return image, class_id


def build_classification_loaders(
    dataset_root: Path,
    batch_size: int = 32,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Создаёт train/val/test DataLoader'ы из YOLO-датасета."""
    train_ds = YoloCropDataset(dataset_root / "train", transform=get_train_transform())
    val_ds = YoloCropDataset(dataset_root / "val", transform=get_val_transform())
    test_ds = YoloCropDataset(dataset_root / "test", transform=get_val_transform())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader

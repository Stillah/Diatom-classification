"""Датасет классификации, построенный из YOLO-разметки (кропы объектов)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Union

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from src.config import INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD

def get_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
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


def _resolve_split_dirs(data_source: Union[str, Path, dict[str, Any]]) -> tuple[Path, Path, Path]:
    """Resolves YOLO dataset config into train/val/test image directories."""
    data_cfg: dict[str, Any]
    data_path: Optional[Path] = None

    if isinstance(data_source, dict):
        data_cfg = data_source
        root_path = Path(data_cfg.get("path") or ".")
    else:
        data_path = Path(data_source)
        if data_path.is_dir():
            return (
                data_path / "train",
                data_path / "val",
                data_path / "test",
            )

        if data_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"Unsupported dataset source: {data_source}")

        if not data_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {data_path}")

        with open(data_path, encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f) or {}
        root_path = Path(data_cfg.get("path") or data_path.parent)

    train_path = data_cfg.get("train", "train")
    val_path = data_cfg.get("val", "val")
    test_path = data_cfg.get("test", "test")

    def _to_path(value: Union[str, Path]) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root_path / candidate

    return (
        _to_path(train_path),
        _to_path(val_path),
        _to_path(test_path),
    )


def _resolve_trainvaltest(data_source: Union[str, Path, dict[str, Any]], split: str) -> Path:
    train_path, val_path, test_path = _resolve_split_dirs(data_source)
    split_dirs = {"train": train_path, "val": val_path, "test": test_path}
    if split not in split_dirs:
        raise ValueError(f"Unsupported split '{split}'. Expected one of: {sorted(split_dirs)}")
    return split_dirs[split]


def _read_yolo_data_config(data_source: Union[str, Path, dict[str, Any]]) -> dict[str, Any]:
    if isinstance(data_source, dict):
        return data_source

    data_path = Path(data_source)
    if data_path.is_dir():
        for candidate in (data_path / "dataset_filtered.yaml", data_path / "data.yaml", data_path / "dataset.yaml"):
            if candidate.exists():
                data_path = candidate
                break
        else:
            return {}

    if not data_path.exists():
        return {}

    with open(data_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg if isinstance(cfg, dict) else {}


def _resolve_selected_class_ids(data_source: Union[str, Path, dict[str, Any]], classes: Optional[list[int] | list[str]]) -> Optional[list[int]]:
    if classes is None:
        return None

    selected: list[int] = []
    seen: set[int] = set()
    names = _read_yolo_data_config(data_source).get("names") or []

    for cls in classes:
        if isinstance(cls, str):
            if cls not in names:
                raise ValueError(f"Unknown class name '{cls}' in dataset YAML names: {names}")
            class_id = names.index(cls)
        else:
            class_id = int(cls)

        if class_id in seen:
            continue
        seen.add(class_id)
        selected.append(class_id)

    return selected


class YoloCropDataset(Dataset):
    """
    Строит датасет классификации из YOLO-сплита (images/ + labels/).
    Каждый bbox в label-файле становится отдельным обучающим примером.
    """

    def __init__(
        self,
        split_dir: Path,
        transform: Optional[Callable] = None,
        class_ids: Optional[list[int]] = None,
    ):
        self.split_dir = Path(split_dir)
        self.images_dir = self.split_dir

        if self.split_dir.parent.name == "images":
            self.labels_dir = self.split_dir.parent.parent / "labels" / self.split_dir.name
        else:
            self.labels_dir = self.split_dir / "labels"

        self.class_ids = set(class_ids) if class_ids is not None else None
        self.class_remap = None
        if self.class_ids is not None:
            self.class_remap = {original_id: idx for idx, original_id in enumerate(class_ids)}

        self.transform = transform
        self.samples: list[tuple[Path, int, tuple[float, float, float, float]]] = []

        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        image_class_dirs = {
            child.name: child for child in sorted(self.images_dir.iterdir()) if child.is_dir()
        }
        label_class_dirs = {
            child.name: child for child in sorted(self.labels_dir.iterdir()) if child.is_dir()
        }

        if image_class_dirs and label_class_dirs:
            for class_name, image_dir in image_class_dirs.items():
                label_dir = label_class_dirs.get(class_name)
                if label_dir is None:
                    continue
                for label_path in sorted(label_dir.rglob("*.txt")):
                    image_path = _find_image(image_dir, label_path.stem)
                    if image_path is None:
                        continue
                    with open(label_path, encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            class_id = int(parts[0])
                            if self.class_ids is not None and class_id not in self.class_ids:
                                continue
                            bbox = tuple(float(v) for v in parts[1:5])
                            remapped_class_id = self.class_remap[class_id] if self.class_remap is not None else class_id
                            self.samples.append((image_path, remapped_class_id, bbox))
        else:
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
                        if self.class_ids is not None and class_id not in self.class_ids:
                            continue
                        bbox = tuple(float(v) for v in parts[1:5])
                        remapped_class_id = self.class_remap[class_id] if self.class_remap is not None else class_id
                        self.samples.append((image_path, remapped_class_id, bbox))

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
    dataset_root: Union[str, Path, dict[str, Any]],
    batch_size: int = 32,
    num_workers: int = 4,
    classes: Optional[list[int] | list[str]] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Создаёт train/val/test DataLoader'ы из YOLO-датасета."""
    train_path, val_path, test_path = _resolve_split_dirs(dataset_root)
    selected_classes = _resolve_selected_class_ids(dataset_root, classes)
    train_ds = YoloCropDataset(train_path, transform=get_train_transform(), class_ids=selected_classes)
    val_ds = YoloCropDataset(val_path, transform=get_val_transform(), class_ids=selected_classes)
    test_ds = YoloCropDataset(test_path, transform=get_val_transform(), class_ids=selected_classes)

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

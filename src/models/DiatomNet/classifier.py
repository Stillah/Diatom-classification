"""Обёртка DiatomNet для обучения и инференса классификации."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.DiatomNet.architecture import DiatomNet
from models.DiatomNet.dataset import (
    build_classification_loaders,
    get_val_transform,
)


class DiatomNetClassifier:
    """Классификатор видов диатомов на основе DiatomNet."""

    def __init__(
        self,
        device: str = "cpu",
        num_classes: int = 6,
        class_names: Optional[List[str]] = None,
        weights_path: Optional[Union[str, Path]] = None,
    ):
        self.device = device
        self.num_classes = num_classes
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.model = DiatomNet(num_classes=num_classes).to(device)
        self._transform = get_val_transform()

        if weights_path is not None:
            self.load(weights_path)

    def _build_loaders(self, train_cfg: Dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader]:
        return build_classification_loaders(
            dataset_root=Path(train_cfg["dataset_root"]),
            batch_size=train_cfg.get("batch", 32),
            num_workers=train_cfg.get("num_workers", 2),
        )

    @staticmethod
    def _run_epoch(
        model: DiatomNet,
        loader: DataLoader,
        criterion: nn.Module,
        device: str,
        optimizer: Optional[optim.Optimizer] = None,
    ) -> tuple[float, float]:
        is_train = optimizer is not None
        model.train(is_train)

        running_loss = 0.0
        correct = 0
        total = 0

        context = torch.enable_grad() if is_train else torch.no_grad()
        with context:
            for images, labels in tqdm(loader, leave=False):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * images.size(0)
                predicted = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return running_loss / total, correct / total

    def train(self, train_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Обучает классификатор на кропах из YOLO-датасета."""
        train_loader, val_loader, _ = self._build_loaders(train_cfg)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=train_cfg.get("lr", 0.001),
            weight_decay=train_cfg.get("weight_decay", 1e-4),
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=train_cfg.get("scheduler_patience", 2),
            min_lr=1e-7,
        )

        epochs = train_cfg.get("epochs", 50)
        patience = train_cfg.get("patience", 10)
        save_path = Path(train_cfg.get("save_path", "best_diatomnet.pth"))

        best_val_acc = 0.0
        epochs_without_improvement = 0
        history: Dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }

        for epoch in range(epochs):
            print(f"\n[Classifier] Epoch {epoch + 1}/{epochs}")
            print("-" * 50)

            train_loss, train_acc = self._run_epoch(
                self.model, train_loader, criterion, self.device, optimizer,
            )
            val_loss, val_acc = self._run_epoch(
                self.model, val_loader, criterion, self.device,
            )
            scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                self.save(save_path)
                print(f"Model saved with val_acc: {best_val_acc:.4f}")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping after {epoch + 1} epochs")
                    break

        return {"best_val_acc": best_val_acc, "history": history, "save_path": str(save_path)}

    def validate(
        self,
        dataset_root: Optional[Union[str, Path]] = None,
        split: str = "val",
        batch_size: int = 32,
        num_workers: int = 2,
    ) -> Dict[str, float]:
        """Оценивает модель на указанном сплите YOLO-датасета."""
        if dataset_root is None:
            raise ValueError("dataset_root is required for validation")

        from models.DiatomNet.dataset import YoloCropDataset

        dataset = YoloCropDataset(Path(dataset_root) / split, transform=get_val_transform())
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        criterion = nn.CrossEntropyLoss()

        val_loss, val_acc = self._run_epoch(self.model, loader, criterion, self.device)
        return {"loss": val_loss, "accuracy": val_acc}

    def classify(self, image: Union[str, Path, np.ndarray]) -> str:
        """Классифицирует одно изображение (кроп) и возвращает название вида."""
        result = self.predict(image)
        return result["class_name"]

    def predict(self, image: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """Классифицирует одно изображение."""
        self.model.eval()

        if isinstance(image, (str, Path)):
            image_bgr = cv2.imread(str(image))
            if image_bgr is None:
                raise RuntimeError(f"Cannot read image: {image}")
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image.astype(np.uint8) if image.dtype != np.uint8 else image

        pil_image = Image.fromarray(image_rgb)
        tensor = self._transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            class_id = int(probs.argmax().item())

        return {
            "class_id": class_id,
            "class_name": self.class_names[class_id],
            "confidence": float(probs[class_id].item()),
            "probabilities": probs.cpu().numpy().tolist(),
        }

    def predict_crop(
        self,
        image: Union[str, Path, np.ndarray],
        box: List[float],
    ) -> Dict[str, Any]:
        """Классифицирует кроп по абсолютным координатам bbox [x1, y1, x2, y2]."""
        if isinstance(image, (str, Path)):
            image_bgr = cv2.imread(str(image))
        else:
            image_bgr = image

        if image_bgr is None:
            raise RuntimeError("Cannot read image for crop classification")

        x1, y1, x2, y2 = [int(v) for v in box]
        crop = image_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return self.predict(crop_rgb)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "num_classes": self.num_classes,
                "class_names": self.class_names,
            },
            path,
        )

    def load(self, path: Union[str, Path]) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.num_classes = checkpoint.get("num_classes", self.num_classes)
            self.class_names = checkpoint.get("class_names", self.class_names)
            self.model = DiatomNet(num_classes=self.num_classes).to(self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

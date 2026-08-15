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

from src.models.DiatomNet.modules import DiatomNet
from src.dataset import (
    YoloCropDataset,
    _read_yolo_data_config,
    _resolve_selected_class_ids,
    _resolve_trainvaltest,
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
        self.selected_classes: Optional[List[int]] = None
        self.model = DiatomNet(num_classes=num_classes).to(device)
        self._transform = get_val_transform()

        if weights_path is not None:
            self.load(weights_path)

    def _build_loaders(self, train_cfg: Dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader]:
        dataset_source = train_cfg.get("data", train_cfg.get("dataset_root"))
        if dataset_source is None:
            raise ValueError("Classification config must include 'data' or 'dataset_root'")

        return build_classification_loaders(
            dataset_root=dataset_source,
            batch_size=train_cfg.get("batch", 32),
            num_workers=train_cfg.get("num_workers", 2),
            classes=train_cfg.get("classes"),
        )

    def _resolve_class_names_from_cfg(self, train_cfg: Dict[str, Any]) -> List[str]:
        dataset_source = train_cfg.get("data", train_cfg.get("dataset_root"))
        if dataset_source is None:
            return self.class_names

        names = _read_yolo_data_config(dataset_source).get("names") or []
        selected_classes = _resolve_selected_class_ids(dataset_source, train_cfg.get("classes"))
        if not selected_classes:
            return self.class_names

        return [str(names[i]) for i in selected_classes]

    @staticmethod
    def _macro_metrics(confusion: torch.Tensor) -> Dict[str, float]:
        confusion = confusion.to(torch.float64)
        true_positive = confusion.diag()
        predicted = confusion.sum(dim=0)
        actual = confusion.sum(dim=1)

        precision = torch.where(
            predicted > 0,
            true_positive / predicted,
            torch.zeros_like(true_positive),
        )
        recall = torch.where(
            actual > 0,
            true_positive / actual,
            torch.zeros_like(true_positive),
        )
        f1 = torch.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall),
            torch.zeros_like(precision),
        )

        return {
            "macro_precision": float(precision.mean().item()),
            "macro_recall": float(recall.mean().item()),
            "macro_f1": float(f1.mean().item()),
        }

    @staticmethod
    def _run_epoch(
        model: DiatomNet,
        loader: DataLoader,
        criterion: nn.Module,
        device: str,
        num_classes: int,
        optimizer: Optional[optim.Optimizer] = None,
    ) -> Dict[str, float]:
        is_train = optimizer is not None
        model.train(is_train)

        running_loss = 0.0
        correct = 0
        total = 0
        confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

        context = torch.enable_grad() if is_train else torch.no_grad()
        with context:
            for images, labels in tqdm(loader, leave=False):
                images = images.to(device)
                labels = labels.to(device)

                if labels.numel() == 0:
                    continue

                num_classes = model.fc.out_features
                if labels.min().item() < 0 or labels.max().item() >= num_classes:
                    raise ValueError(
                        f"Invalid class labels for this model: min={labels.min().item()}, max={labels.max().item()}, "
                        f"expected range [0, {num_classes - 1}]"
                    )

                outputs = model(images)
                loss = criterion(outputs, labels)

                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                predicted = outputs.argmax(dim=1)
                batch_size = labels.size(0)
                running_loss += loss.item() * batch_size
                total += batch_size
                correct += (predicted == labels).sum().item()

                indices = (
                    labels.detach().cpu() * num_classes
                    + predicted.detach().cpu()
                )
                confusion += torch.bincount(
                    indices,
                    minlength=num_classes * num_classes,
                ).reshape(num_classes, num_classes)

        if total == 0:
            raise RuntimeError("DataLoader не содержит ни одного примера")

        metrics = {
            "loss": running_loss / total,
            "accuracy": correct / total,
        }
        metrics.update(DiatomNetClassifier._macro_metrics(confusion))
        return metrics

    @staticmethod
    def _clearml_context(
        class_names: List[str],
    ) -> tuple[Any | None, Any | None, Any | None]:
        """Возвращает текущие Task, Logger и OutputModel, если Task инициализирован."""
        try:
            from clearml import OutputModel, Task

            task = Task.current_task()
            if task is None:
                return None, None, None

            logger = task.get_logger()
            output_model = OutputModel(
                task=task,
                name="DiatomNet best checkpoint",
                framework="PyTorch",
                label_enumeration={
                    label: index for index, label in enumerate(class_names)
                },
                config_dict={
                    "architecture": "DiatomNet",
                    "num_classes": len(class_names),
                },
            )
            return task, logger, output_model
        except ImportError:
            return None, None, None

    @staticmethod
    def _report_epoch(
        logger: Any,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        learning_rate: float,
        best_val_accuracy: float,
    ) -> None:
        if logger is None:
            return

        metric_groups = {
            "Loss": "loss",
            "Accuracy": "accuracy",
            "Macro precision": "macro_precision",
            "Macro recall": "macro_recall",
            "Macro F1": "macro_f1",
        }
        for title, key in metric_groups.items():
            logger.report_scalar(
                title=title,
                series="train",
                value=train_metrics[key],
                iteration=epoch,
            )
            logger.report_scalar(
                title=title,
                series="validation",
                value=val_metrics[key],
                iteration=epoch,
            )

        logger.report_scalar(
            title="Optimization",
            series="learning rate",
            value=learning_rate,
            iteration=epoch,
        )
        logger.report_scalar(
            title="Best metrics",
            series="validation accuracy",
            value=best_val_accuracy,
            iteration=epoch,
        )

    @staticmethod
    def _macro_precision_recall_f1(
        y_true: List[int],
        y_pred: List[int],
        num_classes: int,
    ) -> tuple[float, float, float]:
        """Считает precision/recall/f1 по каждому классу и усредняет (macro)."""
        precisions, recalls, f1s = [], [], []

        for c in range(num_classes):
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0
            )

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        return (
            sum(precisions) / num_classes,
            sum(recalls) / num_classes,
            sum(f1s) / num_classes,
        )

    def train(self, train_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Обучает классификатор на кропах из YOLO-датасета."""
        selected_class_names = self._resolve_class_names_from_cfg(train_cfg)
        selected_classes = _resolve_selected_class_ids(
            train_cfg.get("data", train_cfg.get("dataset_root")),
            train_cfg.get("classes"),
        )

        if selected_classes:
            self.selected_classes = selected_classes
            self.num_classes = len(selected_classes)
            self.class_names = selected_class_names
            self.model = DiatomNet(num_classes=self.num_classes).to(self.device)
        else:
            self.selected_classes = None

        dataset_source = train_cfg.get("data", train_cfg.get("dataset_root"))
        train_split = _resolve_trainvaltest(dataset_source, "train")
        val_split = _resolve_trainvaltest(dataset_source, "val")

        train_images_dir = train_split
        val_images_dir = val_split
        train_labels_dir = train_images_dir.parent.parent / "labels" / train_images_dir.name if train_images_dir.parent.name == "images" else train_images_dir / "labels"
        val_labels_dir = val_images_dir.parent.parent / "labels" / val_images_dir.name if val_images_dir.parent.name == "images" else val_images_dir / "labels"

        train_class_dirs = sorted(p for p in train_images_dir.iterdir() if p.is_dir()) if train_images_dir.exists() else []
        val_class_dirs = sorted(p for p in val_images_dir.iterdir() if p.is_dir()) if val_images_dir.exists() else []
        train_label_class_dirs = sorted(p for p in train_labels_dir.iterdir() if p.is_dir()) if train_labels_dir.exists() else []
        val_label_class_dirs = sorted(p for p in val_labels_dir.iterdir() if p.is_dir()) if val_labels_dir.exists() else []

        train_images_total = sum(len(list(p.glob("*"))) for p in train_class_dirs)
        val_images_total = sum(len(list(p.glob("*"))) for p in val_class_dirs)
        train_labels_total = sum(len(list(p.glob("*.txt"))) for p in train_label_class_dirs)
        val_labels_total = sum(len(list(p.glob("*.txt"))) for p in val_label_class_dirs)

        print(f"[Classifier] Train split: {train_images_dir}")
        print(f"[Classifier] Train class folders used: {len(train_class_dirs)}")
        print(f"[Classifier] Train label folders used: {len(train_label_class_dirs)}")
        print(f"[Classifier] Train images counted: {train_images_total}")
        print(f"[Classifier] Train labels counted: {train_labels_total}")
        print(f"[Classifier] Val split: {val_images_dir}")
        print(f"[Classifier] Val class folders used: {len(val_class_dirs)}")
        print(f"[Classifier] Val label folders used: {len(val_label_class_dirs)}")
        print(f"[Classifier] Val images counted: {val_images_total}")
        print(f"[Classifier] Val labels counted: {val_labels_total}")

        train_loader, val_loader, _ = self._build_loaders(train_cfg)

        print(f"[Classifier] Train batches: {len(train_loader)}")
        print(f"[Classifier] Val batches: {len(val_loader)}")

        train_samples = sum(len(dataset) for dataset in [train_loader.dataset])
        val_samples = sum(len(dataset) for dataset in [val_loader.dataset])
        print(f"[Classifier] Loaded train samples: {train_samples}")
        print(f"[Classifier] Loaded val samples: {val_samples}")

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
        task, logger, output_model = self._clearml_context(self.class_names)

        best_val_acc = 0.0
        best_epoch = 0
        epochs_without_improvement = 0
        history: Dict[str, list] = {
            "train_loss": [],
            "train_accuracy": [],
            "train_macro_precision": [],
            "train_macro_recall": [],
            "train_macro_f1": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_macro_precision": [],
            "val_macro_recall": [],
            "val_macro_f1": [],
            "learning_rate": [],
        }

        for epoch_index in range(epochs):
            epoch = epoch_index + 1
            print(f"\n[Classifier] Epoch {epoch}/{epochs}")
            print("-" * 50)

            train_metrics = self._run_epoch(
                self.model,
                train_loader,
                criterion,
                self.device,
                self.num_classes,
                optimizer,
            )
            val_metrics = self._run_epoch(
                self.model,
                val_loader,
                criterion,
                self.device,
                self.num_classes,
            )
            scheduler.step(val_metrics["loss"])
            learning_rate = float(optimizer.param_groups[0]["lr"])

            for key, value in train_metrics.items():
                history[f"train_{key}"].append(value)
            for key, value in val_metrics.items():
                history[f"val_{key}"].append(value)
            history["learning_rate"].append(learning_rate)

            improved = val_metrics["accuracy"] > best_val_acc
            if improved:
                best_val_acc = val_metrics["accuracy"]
                best_epoch = epoch
                epochs_without_improvement = 0
                self.save(save_path)
                print(f"Model saved with val_acc: {best_val_acc:.4f}")

                if output_model is not None:
                    try:
                        output_model.update_weights(
                            weights_filename=str(save_path),
                            iteration=epoch,
                            auto_delete_file=False,
                            async_enable=False,
                        )
                    except Exception as exc:
                        if logger is not None:
                            logger.report_text(
                                f"Не удалось загрузить checkpoint в ClearML: {exc}",
                                print_console=False,
                            )
            else:
                epochs_without_improvement += 1

            self._report_epoch(
                logger=logger,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                learning_rate=learning_rate,
                best_val_accuracy=best_val_acc,
            )

            print(
                "Train: "
                f"loss={train_metrics['loss']:.4f}, "
                f"acc={train_metrics['accuracy']:.4f}, "
                f"macro_f1={train_metrics['macro_f1']:.4f}"
            )
            print(
                "Validation: "
                f"loss={val_metrics['loss']:.4f}, "
                f"acc={val_metrics['accuracy']:.4f}, "
                f"macro_f1={val_metrics['macro_f1']:.4f}"
            )

            if not improved and epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch} epochs")
                break

        if logger is not None:
            logger.report_single_value(
                name="best_validation_accuracy",
                value=best_val_acc,
            )
            logger.report_single_value(name="best_epoch", value=best_epoch)

        if task is not None:
            try:
                task.upload_artifact(
                    name="classification_history",
                    artifact_object=history,
                    wait_on_upload=True,
                )
            except Exception as exc:
                print(f"Не удалось загрузить history в ClearML: {exc}")

        return {
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "history": history,
            "save_path": str(save_path),
        }

    def validate(
        self,
        dataset_root: Optional[Union[str, Path]] = None,
        split: str = "val",
        batch_size: int = 32,
        num_workers: int = 2,
        classes: Optional[List[Union[int, str]]] = None,
    ) -> Dict[str, float]:
        """Оценивает модель на указанном сплите YOLO-датасета."""
        if dataset_root is None:
            raise ValueError("dataset_root is required for validation")

        resolved_classes = classes if classes is not None else self.selected_classes
        if resolved_classes is None:
            resolved_classes = _resolve_selected_class_ids(dataset_root, None)

        split_dir = _resolve_trainvaltest(dataset_root, split)
        dataset = YoloCropDataset(
            split_dir,
            transform=get_val_transform(),
            class_ids=resolved_classes,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        criterion = nn.CrossEntropyLoss()

        if resolved_classes is not None and self.num_classes != len(resolved_classes):
            raise ValueError(
                f"Model num_classes ({self.num_classes}) does not match selected validation classes ({len(resolved_classes)}): "
                f"{resolved_classes}."
            )

        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds: List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for images, labels in tqdm(loader, leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                predicted = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_loss = running_loss / total
        val_acc = correct / total
        precision, recall, f1 = self._macro_precision_recall_f1(
            all_labels, all_preds, self.num_classes,
        )

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

    def predict_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
    ) -> List[Dict[str, Any]]:
        """Классифицирует список изображений одним батчем."""
        if not images:
            return []

        self.model.eval()
        transformed = []
        for image in images:
            if isinstance(image, (str, Path)):
                image_bgr = cv2.imread(str(image))
                if image_bgr is None:
                    raise RuntimeError(f"Cannot read image: {image}")
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image.astype(np.uint8) if image.dtype != np.uint8 else image
            transformed.append(Image.fromarray(image_rgb))

        tensor = torch.stack([self._transform(img) for img in transformed]).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            class_ids = probs.argmax(dim=1).cpu().tolist()

        results: List[Dict[str, Any]] = []
        for index, class_id in enumerate(class_ids):
            class_id = int(class_id)
            probs_row = probs[index]
            results.append(
                {
                    "class_id": class_id,
                    "class_name": self.class_names[class_id],
                    "confidence": float(probs_row[class_id].item()),
                    "probabilities": probs_row.cpu().numpy().tolist(),
                }
            )
        return results

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
        if crop.size == 0:
            raise ValueError(f"Пустой crop для bbox: {box}")
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
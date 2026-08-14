"""
Pipeline: детекция (YOLOv11) + классификация видов (DiatomNet).

Детектор находит объекты на изображении, классификатор уточняет вид
по кропу каждого bbox.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from src.config import CLASSIFICATION_CONFIG, OUTPUT_ROOT, TARGET_CLASSES
from src.models.DiatomNet.model import DiatomNetClassifier
from src.models.SC_Diatomnet.model import YOLOv11Baseline


class DiatomPipeline:
    """Двухэтапный pipeline: детекция → классификация видов."""

    def __init__(
        self,
        device: str = "cpu",
        detector: Optional[YOLOv11Baseline] = None,
        classifier: Optional[DiatomNetClassifier] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.device = device
        self.class_names = class_names or TARGET_CLASSES

        self.detector = detector or YOLOv11Baseline(device=device)
        self.classifier = classifier or DiatomNetClassifier(
            device=device,
            num_classes=len(self.class_names),
            class_names=self.class_names,
        )

    def train_detection(self, train_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Обучает YOLO-детектор."""
        print("=" * 60)
        print("Обучение детектора (YOLOv11)")
        print("=" * 60)
        return self.detector.train(train_cfg)

    def train_classification(self, train_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Обучает DiatomNet-классификатор на кропах из YOLO-датасета."""
        print("=" * 60)
        print("Обучение классификатора (DiatomNet)")
        print("=" * 60)
        return self.classifier.train(train_cfg)

    def train(
        self,
        detection_cfg: Dict[str, Any] | None = None,
        classification_cfg: Dict[str, Any] | None = None,
        train_detector: bool = True,
        train_classifier: bool = True,
    ) -> Dict[str, Any]:
        """Последовательно обучает детектор и классификатор."""
        results: Dict[str, Any] = {}

        if train_detector:
            results["detection"] = self.train_detection(detection_cfg)

        if train_classifier:
            results["classification"] = self.train_classification(classification_cfg)

        return results

    def validate_detection(
        self,
        data: Optional[Union[str, Path]] = None,
        val_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        return self.detector.validate(data=data, val_cfg=val_cfg)

    def validate_classification(
        self,
        dataset_root: Optional[Union[str, Path]] = None,
        split: str = "val",
        classes: Optional[List[Union[int, str]]] = None,
    ) -> Dict[str, float]:
        root = dataset_root or CLASSIFICATION_CONFIG.get("data", OUTPUT_ROOT)
        resolved_classes = classes if classes is not None else CLASSIFICATION_CONFIG.get("classes")
        return self.classifier.validate(dataset_root=root, split=split, classes=resolved_classes)

    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        det_conf: float = 0.25,
        det_iou: float = 0.45,
        det_imgsz: int = 640,
        use_classifier: bool = True,
    ) -> Dict[str, Any]:
        """
        Запускает полный pipeline на одном изображении.

        Returns:
            dict с ключами:
                - boxes: [[x1,y1,x2,y2], ...]
                - detection_class_ids: id класса от детектора
                - detection_confidences: уверенность детектора
                - class_ids: id вида (от классификатора или детектора)
                - class_names: названия видов
                - confidences: итоговая уверенность
        """
        det_result = self.detector.predict(
            image, conf=det_conf, iou=det_iou, imgsz=det_imgsz,
        )

        boxes = det_result["boxes"]
        det_class_ids = det_result["class_ids"]
        det_confidences = det_result["confidences"]

        if not boxes:
            return {
                "boxes": [],
                "detection_class_ids": [],
                "detection_confidences": [],
                "class_ids": [],
                "class_names": [],
                "confidences": [],
            }

        class_ids: List[int] = []
        class_names: List[str] = []
        confidences: List[float] = []

        for i, box in enumerate(boxes):
            if use_classifier and self.classifier is not None:
                cls_result = self.classifier.predict_crop(image, box)
                class_ids.append(cls_result["class_id"])
                class_names.append(cls_result["class_name"])
                confidences.append(cls_result["confidence"])
            else:
                cid = det_class_ids[i]
                class_ids.append(cid)
                class_names.append(
                    self.class_names[cid] if cid < len(self.class_names) else str(cid)
                )
                confidences.append(det_confidences[i])

        return {
            "boxes": boxes,
            "detection_class_ids": det_class_ids,
            "detection_confidences": det_confidences,
            "class_ids": class_ids,
            "class_names": class_names,
            "confidences": confidences,
        }

    def save(
        self,
        detector_path: Union[str, Path, None] = None,
        classifier_path: Union[str, Path, None] = None,
    ) -> None:
        """Сохраняет веса обеих моделей."""
        if detector_path is not None:
            self.detector.save(detector_path)
        if classifier_path is not None:
            self.classifier.save(classifier_path)

    def load(
        self,
        detector_path: Optional[Union[str, Path]] = None,
        classifier_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Загружает веса одной или обеих моделей."""
        if detector_path is not None:
            self.detector.load(detector_path)
        if classifier_path is not None:
            self.classifier.load(classifier_path)

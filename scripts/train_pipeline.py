"""Обучение pipeline: детекция (YOLOv11) + классификация видов (DiatomNet)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import CLASSIFICATION_CONFIG, DEVICE, OUTPUT_ROOT, DETECTION_CONFIG
from src.models.SC_Diatomnet.model import YOLOv11Baseline
from src.models.pipeline import DiatomPipeline

if __name__ == "__main__":
    print("Training started")

    pipeline = DiatomPipeline(
        device=DEVICE,
        detector=YOLOv11Baseline(device=DEVICE, model_path="yolo11n.pt"),
    )

    # 1. Обучение детектора и классификатора
    results = pipeline.train(
        detection_cfg=DETECTION_CONFIG,
        train_classifier=False
    )

    # 2. Валидация детектора
    det_metrics = pipeline.validate_detection()
    print("Detection mAP@50-95:", det_metrics.get("metrics/mAP50-95(B)", "N/A"))
    print("Detection mAP@50:", det_metrics.get("metrics/mAP50(B)", "N/A"))

    # 3. Валидация классификатора
    cls_metrics = pipeline.validate_classification()
    print("Classification val accuracy:", cls_metrics.get("accuracy", "N/A"))

    # 4. Сохранение обеих моделей
    detector_path = OUTPUT_ROOT / "best_diatom_detector.pt"
    classifier_path = OUTPUT_ROOT / "best_diatomnet.pth"
    pipeline.save(detector_path, classifier_path)
    print(f"Detector saved to {detector_path}")
    print(f"Classifier saved to {classifier_path}")

    print("Training finished")
    print(f"Results: {results}")

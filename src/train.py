"""Обучение pipeline: детекция (YOLOv11) + классификация видов (DiatomNet)."""

from config import CLASSIFICATION_CONFIG, DEVICE, OUTPUT_ROOT, TRAIN_CONFIG
from models.SC_Diatomnet.models import YOLOv11Baseline
from pipeline import DiatomPipeline

if __name__ == "__main__":
    print("Training started")

    pipeline = DiatomPipeline(
        device=DEVICE,
        detector=YOLOv11Baseline(device=DEVICE, model_path="yolo11n.pt"),
    )

    # 1. Обучение детектора и классификатора
    results = pipeline.train(
        detection_cfg=TRAIN_CONFIG,
        classification_cfg=CLASSIFICATION_CONFIG,
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

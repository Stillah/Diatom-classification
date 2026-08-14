"""Обучение pipeline: детекция (YOLOv11) + классификация видов (DiatomNet)."""

from src.clearml_tracking import init_clearml_task, register_model_file
from src.config import (
    CLASSIFICATION_CONFIG,
    DETECTION_CONFIG,
    DEVICE,
    OUTPUT_ROOT,
    TARGET_CLASSES,
)
from src.models.SC_Diatomnet.model import YOLOv11Baseline
from src.models.pipeline import DiatomPipeline


def main() -> None:
    clearml_config = {
        "detection": DETECTION_CONFIG,
        "classification": CLASSIFICATION_CONFIG,
    }
    task = init_clearml_task(
        default_task_name="YOLOv11 + DiatomNet pipeline",
        config=clearml_config,
    )

    try:
        print("Training started")
        pipeline = DiatomPipeline(
            device=DEVICE,
            detector=YOLOv11Baseline(device=DEVICE, model_path="yolo11n.pt"),
        )

        results = pipeline.train(
            detection_cfg=DETECTION_CONFIG,
            classification_cfg=CLASSIFICATION_CONFIG,
        )

        det_metrics = pipeline.validate_detection()
        print("Detection mAP@50-95:", det_metrics.get("metrics/mAP50-95(B)", "N/A"))
        print("Detection mAP@50:", det_metrics.get("metrics/mAP50(B)", "N/A"))

        cls_metrics = pipeline.validate_classification()
        print("Classification val metrics:", cls_metrics)

        detector_path = OUTPUT_ROOT / "best_diatom_detector.pt"
        classifier_path = OUTPUT_ROOT / "best_diatomnet.pth"
        pipeline.save(detector_path, classifier_path)
        print(f"Detector saved to {detector_path}")
        print(f"Classifier saved to {classifier_path}")

        if task is not None:
            logger = task.get_logger()
            for key, value in det_metrics.items():
                if isinstance(value, (int, float)):
                    logger.report_single_value(f"detection/{key}", float(value))
            for key, value in cls_metrics.items():
                logger.report_single_value(f"classification/{key}", float(value))

            register_model_file(
                task=task,
                weights_path=detector_path,
                name="YOLOv11 diatom detector",
                class_names=TARGET_CLASSES,
            )
            task.upload_artifact(
                name="training_results",
                artifact_object=results,
                wait_on_upload=True,
            )
            print(f"ClearML Task ID: {task.id}")

        print("Training finished")
        print(f"Results: {results}")
    finally:
        if task is not None:
            task.close()


if __name__ == "__main__":
    main()

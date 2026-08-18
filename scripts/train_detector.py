from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clearml_tracking import init_clearml_task, register_model_file
from src.config import DEVICE, DETECTION_CONFIG, OUTPUT_ROOT, TARGET_CLASSES
from src.models.pipeline import DiatomPipeline

SAVE_PATH = OUTPUT_ROOT / "best_diatom_detector.pt"


def _with_valid_run_name(train_cfg: dict) -> dict:
    cfg = dict(train_cfg)
    run_name = str(cfg.get("name", "")).strip()
    if len(run_name) < 3:
        cfg["name"] = f"run_{run_name or 'detector'}"
    return cfg


def main() -> None:
    # if SAVE_PATH.exists():
    #     print(f"Детектор уже обучен: {SAVE_PATH} — пропускаю обучение.")
    #     return
    
    detection_cfg = _with_valid_run_name(DETECTION_CONFIG)

    task = init_clearml_task(
        default_task_name="YOLOv11 detector training",
        config={"detection": detection_cfg},
    )

    try:
        pipeline = DiatomPipeline(device=DEVICE)

        print("Запуск обучения детектора...")
        train_results = pipeline.train_detection(detection_cfg)
        pipeline.save(detector_path=SAVE_PATH)

        # Валидация через интерфейс pipeline
        det_metrics = pipeline.validate_detection()
        print("\n" + "=" * 60)
        print("Detection mAP@50-95:", det_metrics.get("metrics/mAP50-95(B)", "N/A"))
        print("Detection mAP@50:", det_metrics.get("metrics/mAP50(B)", "N/A"))
        print(f"Веса сохранены в: {SAVE_PATH}")
        print("=" * 60)

        if task is not None:
            logger = task.get_logger()
            for key, value in det_metrics.items():
                if isinstance(value, (int, float)):
                    logger.report_single_value(name=f"detection/{key}", value=float(value))

            register_model_file(
                task=task,
                weights_path=SAVE_PATH,
                name="YOLOv11 diatom detector",
                class_names=TARGET_CLASSES,
            )
            task.upload_artifact(
                name="detection_training_results",
                artifact_object=train_results,
                wait_on_upload=True,
            )
            print(f"ClearML Task ID: {task.id}")
    finally:
        if task is not None:
            task.close()


if __name__ == "__main__":
    main()

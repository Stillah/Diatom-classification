from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DEVICE, DETECTION_CONFIG, OUTPUT_ROOT
from src.models.pipeline import DiatomPipeline

SAVE_PATH = OUTPUT_ROOT / "best_diatom_detector.pt"


def main() -> None:
    # if SAVE_PATH.exists():
    #     print(f"Детектор уже обучен: {SAVE_PATH} — пропускаю обучение.")
    #     return

    pipeline = DiatomPipeline(device=DEVICE)

    print("Запуск обучения детектора...")
    pipeline.train_detection(DETECTION_CONFIG)
    pipeline.save(detector_path=SAVE_PATH)

    # Валидация через интерфейс pipeline
    det_metrics = pipeline.validate_detection()
    print("\n" + "=" * 60)
    print("Detection mAP@50-95:", det_metrics.get("metrics/mAP50-95(B)", "N/A"))
    print("Detection mAP@50:", det_metrics.get("metrics/mAP50(B)", "N/A"))
    print(f"Веса сохранены в: {SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()

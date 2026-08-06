from __future__ import annotations

from pathlib import Path

from src.config import CLASSIFICATION_CONFIG, DEVICE, OUTPUT_ROOT
from src.models.pipeline import DiatomPipeline

SAVE_PATH = Path(CLASSIFICATION_CONFIG["save_path"])


def main() -> None:
    if SAVE_PATH.exists():
        print(f"Классификатор уже обучен: {SAVE_PATH} — пропускаю обучение.")
        return

    pipeline = DiatomPipeline(device=DEVICE)

    print("Запуск обучения классификатора (DiatomNet)...")
    result = pipeline.train_classification(CLASSIFICATION_CONFIG)
    pipeline.save(classifier_path=SAVE_PATH)

    # Валидация через интерфейс pipeline
    cls_metrics = pipeline.validate_classification()
    print("\n" + "=" * 60)
    print("Classification val accuracy:", cls_metrics.get("accuracy", "N/A"))
    print(f"Веса сохранены в: {SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
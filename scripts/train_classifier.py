from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CLASSIFICATION_CONFIG, DEVICE, OUTPUT_ROOT
from src.models.pipeline import DiatomPipeline

SAVE_PATH = Path(CLASSIFICATION_CONFIG["save_path"])


def main() -> None:
    pipeline = DiatomPipeline(device=DEVICE)
    
    if not SAVE_PATH.exists():
        print("Запуск обучения классификатора (DiatomNet)...")
        pipeline.train_classification(CLASSIFICATION_CONFIG)
        pipeline.save(classifier_path=SAVE_PATH)
        
    else:
        print(f"Классификатор уже обучен: {SAVE_PATH} — пропускаю обучение.")
        pipeline.load(classifier_path=SAVE_PATH)

    # Валидация через интерфейс pipeline
    cls_metrics = pipeline.validate_classification()
    print("\n" + "=" * 60)
    print("Classification val accuracy:", cls_metrics.get("accuracy", "N/A"))
    print("Classification val F1_macro:", cls_metrics.get("f1_macro", "N/A"))
    print("Classification val precision:", cls_metrics.get("precision_macro", "N/A"))
    print("Classification val recall:", cls_metrics.get("recall_macro", "N/A"))
    print(f"Веса сохранены в: {SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
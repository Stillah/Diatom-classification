"""Обучение классификатора видов (DiatomNet) отдельно от детектора."""

from __future__ import annotations

from pathlib import Path

from config import CLASSIFICATION_CONFIG, DEVICE, OUTPUT_ROOT
from pipeline import DiatomPipeline

SAVE_PATH = Path(CLASSIFICATION_CONFIG["save_path"])


def main() -> None:
    if SAVE_PATH.exists():
        print(f"Классификатор уже обучен: {SAVE_PATH} — пропускаю обучение.")
        return

    pipeline = DiatomPipeline(device=DEVICE)

    print("Запуск обучения классификатора (DiatomNet)...")
    result = pipeline.train_classification(CLASSIFICATION_CONFIG)

    print("\n" + "=" * 60)
    print(f"Обучение завершено. Лучшая val_acc: {result['best_val_acc']:.4f}")
    print(f"Веса сохранены в: {result['save_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

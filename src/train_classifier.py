"""Обучение классификатора видов (DiatomNet) отдельно от детектора."""

from __future__ import annotations

from pathlib import Path

from clearml_tracking import init_clearml_task
from config import CLASSIFICATION_CONFIG, DEVICE
from pipeline import DiatomPipeline

SAVE_PATH = Path(CLASSIFICATION_CONFIG["save_path"])


def main() -> None:
    if SAVE_PATH.exists():
        print(f"Классификатор уже обучен: {SAVE_PATH} — пропускаю обучение.")
        return

    task = init_clearml_task(
        default_task_name="DiatomNet classifier",
        config=CLASSIFICATION_CONFIG,
    )

    try:
        pipeline = DiatomPipeline(device=DEVICE)

        print("Запуск обучения классификатора (DiatomNet)...")
        result = pipeline.train_classification(CLASSIFICATION_CONFIG)

        print("\n" + "=" * 60)
        print(f"Обучение завершено. Лучшая val_acc: {result['best_val_acc']:.4f}")
        print(f"Лучшая эпоха: {result['best_epoch']}")
        print(f"Веса сохранены в: {result['save_path']}")
        if task is not None:
            print(f"ClearML Task ID: {task.id}")
        print("=" * 60)
    finally:
        if task is not None:
            task.close()


if __name__ == "__main__":
    main()

"""Обучение классификатора видов (DiatomNet) отдельно от детектора."""

from __future__ import annotations

import os
from pathlib import Path

from clearml_tracking import init_clearml_task
from config import CLASSIFICATION_CONFIG, DEVICE, OUTPUT_ROOT
from pipeline import DiatomPipeline


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolved_config() -> dict:
    cfg = dict(CLASSIFICATION_CONFIG)

    if os.getenv("DIATOM_EPOCHS"):
        cfg["epochs"] = int(os.environ["DIATOM_EPOCHS"])
    if os.getenv("DIATOM_PATIENCE"):
        cfg["patience"] = int(os.environ["DIATOM_PATIENCE"])
    if os.getenv("DIATOM_NUM_WORKERS"):
        cfg["num_workers"] = int(os.environ["DIATOM_NUM_WORKERS"])
    if os.getenv("DIATOM_SAVE_PATH"):
        cfg["save_path"] = os.environ["DIATOM_SAVE_PATH"]
    elif _env_flag("DIATOM_SMOKE_TEST"):
        cfg["save_path"] = str(
            OUTPUT_ROOT / "smoke" / "best_diatomnet_smoke.pth"
        )

    return cfg


def main() -> None:
    train_config = _resolved_config()
    save_path = Path(train_config["save_path"])
    force_train = _env_flag("DIATOM_FORCE_TRAIN")

    if save_path.exists() and not force_train:
        print(f"Классификатор уже обучен: {save_path} — пропускаю обучение.")
        return

    task = init_clearml_task(
        default_task_name="DiatomNet classifier",
        config=train_config,
    )

    try:
        pipeline = DiatomPipeline(device=DEVICE)

        print("Запуск обучения классификатора (DiatomNet)...")
        print(f"Параметры запуска: {train_config}")
        result = pipeline.train_classification(train_config)

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

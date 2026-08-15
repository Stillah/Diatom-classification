"""Smoke/full training of DiatomNet on a folder-per-class dataset."""

from __future__ import annotations

import os
from pathlib import Path

from clearml_tracking import init_clearml_task
from config import DEVICE, OUTPUT_ROOT
from models.DiatomNet.dataset_folder import (
    discover_folder_classes,
    folder_class_counts,
)
from models.DiatomNet.folder_classifier_fixed import FolderDiatomNetClassifier


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


def _build_config() -> dict:
    smoke = _env_flag("DIATOM_SMOKE_TEST")
    default_save = OUTPUT_ROOT / "smoke" / "best_diatomnet_smoke.pth"
    return {
        "dataset_root": _required_env("DIATOM_DATASET_ROOT"),
        "dataset_format": "folder",
        "epochs": int(os.getenv("DIATOM_EPOCHS", "1" if smoke else "50")),
        "batch": int(os.getenv("DIATOM_BATCH", "8" if smoke else "32")),
        "lr": float(os.getenv("DIATOM_LR", "0.001")),
        "weight_decay": float(os.getenv("DIATOM_WEIGHT_DECAY", "0.0001")),
        "patience": int(os.getenv("DIATOM_PATIENCE", "1" if smoke else "10")),
        "scheduler_patience": int(os.getenv("DIATOM_SCHEDULER_PATIENCE", "2")),
        "num_workers": int(os.getenv("DIATOM_NUM_WORKERS", "0" if smoke else "2")),
        "seed": int(os.getenv("DIATOM_SEED", "42")),
        "min_class_images": int(os.getenv("DIATOM_MIN_CLASS_IMAGES", "5")),
        "max_classes": int(os.getenv("DIATOM_MAX_CLASSES", "0")) or None,
        "val_fraction": float(os.getenv("DIATOM_VAL_FRACTION", "0.2")),
        "test_fraction": float(os.getenv("DIATOM_TEST_FRACTION", "0.2")),
        "exclude_classes": [
            item.strip()
            for item in os.getenv("DIATOM_EXCLUDE_CLASSES", "").split(",")
            if item.strip()
        ],
        "save_path": os.getenv("DIATOM_SAVE_PATH", str(default_save)),
    }


def main() -> None:
    config = _build_config()
    dataset_root = Path(config["dataset_root"])
    class_names = discover_folder_classes(
        dataset_root=dataset_root,
        min_class_images=config["min_class_images"],
        max_classes=config["max_classes"],
        exclude_classes=config["exclude_classes"],
    )
    counts = folder_class_counts(dataset_root)
    config["class_names"] = class_names
    config["class_counts"] = {name: counts[name] for name in class_names}

    print("Selected classes:")
    for name in class_names:
        print(f"  - {name}: {counts[name]} images")

    save_path = Path(config["save_path"])
    if save_path.exists() and not _env_flag("DIATOM_FORCE_TRAIN"):
        print(f"Classifier already exists: {save_path}; skipping training.")
        return

    task = init_clearml_task(
        default_task_name="DiatomNet folder dataset",
        config=config,
    )
    try:
        classifier = FolderDiatomNetClassifier(
            device=DEVICE,
            num_classes=len(class_names),
            class_names=class_names,
        )
        print(f"Device: {DEVICE}")
        result = classifier.train(config)
        print("\n" + "=" * 60)
        print(f"Training finished. Best val_acc: {result['best_val_acc']:.4f}")
        print(f"Best epoch: {result['best_epoch']}")
        print(f"Weights saved to: {result['save_path']}")
        if task is not None:
            print(f"ClearML Task ID: {task.id}")
        print("=" * 60)
    finally:
        if task is not None:
            task.close()


if __name__ == "__main__":
    main()

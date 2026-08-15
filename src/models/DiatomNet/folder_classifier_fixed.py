"""DiatomNet classifier for folder-per-class datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from torch.utils.data import DataLoader

from models.DiatomNet.classifier import DiatomNetClassifier
from models.DiatomNet.dataset_folder import build_folder_classification_loaders


class FolderDiatomNetClassifier(DiatomNetClassifier):
    """Use the existing ClearML-aware training loop with folder loaders."""

    def _build_loaders(
        self,
        train_cfg: Dict[str, Any],
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        return build_folder_classification_loaders(
            dataset_root=Path(train_cfg["dataset_root"]),
            class_names=self.class_names,
            batch_size=train_cfg.get("batch", 32),
            num_workers=train_cfg.get("num_workers", 2),
            seed=train_cfg.get("seed", 42),
            val_fraction=train_cfg.get("val_fraction", 0.2),
            test_fraction=train_cfg.get("test_fraction", 0.2),
        )

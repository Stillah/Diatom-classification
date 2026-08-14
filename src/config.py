import os
from pathlib import Path

import numpy as np
import torch

# ID S3-коннектора можно переопределить через переменную окружения.
# Значение по умолчанию соответствует текущему датасету проекта.
STORAGE_ID = os.getenv("DIATOM_STORAGE_ID", "bt10d2p35vtasuqqfkps")
ROOT = Path(os.getenv("DIATOM_STORAGE_ROOT", f"/job/s3/{STORAGE_ID}"))
DATASET_ROOT = ROOT / "raw"  # Folder containing images and annotations
OUTPUT_ROOT = ROOT  # Where YOLO dataset will be created
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(SEED)

# Six species used in the paper
TARGET_CLASSES = [
    "Encyonema silesiacum",
    "Fragilaria recapitellata",
    "Gomphonema olivaceum",
    "Navicula cryptotenella",
    "Navicula reichardtiana",
    "Planothidium lanceolatum",
]

# Create class-to-id mapping
class_to_id = {name: idx for idx, name in enumerate(TARGET_CLASSES)}

DETECTION_CONFIG = {
    "data": str(OUTPUT_ROOT / "dataset_filtered.yaml"),  # YOLO format dataset
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "lr0": 0.001,
    "patience": 10,
    "cos_lr": True,
    "warmup_epochs": 3,
    "project": "diatoms",
    "name": "v2",
    # Augmentation settings
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 15.0,
    "scale": 0.3,
    "fliplr": 0.5,
    "flipud": 0.5,
    # Class balancing
    "cls_pw": 0.8,
}

# Backward compatibility for scripts still using TRAIN_CONFIG.
TRAIN_CONFIG = DETECTION_CONFIG

CLASSIFICATION_CONFIG = {
    "dataset_root": str(OUTPUT_ROOT),
    "epochs": 50,
    "batch": 32,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "patience": 10,
    "scheduler_patience": 2,
    "num_workers": 2,
    "save_path": str(OUTPUT_ROOT / "best_diatomnet.pth"),
}

INPUT_SIZE = (128, 432)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

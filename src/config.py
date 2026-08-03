import os
from pathlib import Path

import numpy as np
import torch

# ID S3-коннектора можно переопределить через переменную окружения.
# Значение по умолчанию соответствует текущему DataSphere-проекту.
STORAGE_ID = os.getenv("DIATOM_STORAGE_ID", "bt1cef26io7ofqin11u8")
ROOT = Path(os.getenv("DIATOM_STORAGE_ROOT", f"/job/s3/{STORAGE_ID}"))
DATASET_ROOT = ROOT / "raw"              # Folder containing images and annotations
OUTPUT_ROOT = ROOT / "yolov11"           # Where YOLO dataset will be created
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

TRAIN_CONFIG = {
    "data": str(OUTPUT_ROOT / "data.yaml"),   # YOLO format dataset
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "lr0": 0.001,
    "patience": 10,
    "cos_lr": True,
    "warmup_epochs": 3,
    "project": "diatoms",
    "name": "v1",
}

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

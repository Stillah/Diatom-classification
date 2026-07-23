import numpy as np
from pathlib import Path
import torch 
STORAGE_ID = "bt15mrdpleurdj659o9m"
ROOT = Path(f"/job/s3/{STORAGE_ID}")
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
    "Planothidium lanceolatum"
]

# Create class-to-id mapping
class_to_id = {name: idx for idx, name in enumerate(TARGET_CLASSES)}


TRAIN_CONFIG = {
    "data": str(OUTPUT_ROOT / "data.yaml"),   # YOLO format dataset
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "lr0": 0.001,        # fine-tune
    "patience": 10,      # stop if no improvement for 10 epochs
    "cos_lr": True,
    "warmup_epochs": 3,
    "project": "diatoms",
    "name": "v1",
}

CLASSIFICATION_CONFIG = {
    "dataset_root": str(OUTPUT_ROOT),   # YOLO-сплиты train/val/test с кропами
    "epochs": 50,
    "batch": 32,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "patience": 10,
    "scheduler_patience": 2,
    "num_workers": 2,
    "save_path": str(OUTPUT_ROOT / "best_diatomnet.pth"),
}
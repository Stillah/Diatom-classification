import numpy as np
from pathlib import Path

STORAGE_ID = "bt15mrdpleurdj659o9m"
ROOT = Path(f"/job/s3/{STORAGE_ID}")
DATASET_ROOT = ROOT / "raw"          # Folder containing images and annotations
OUTPUT_ROOT = ROOT / "yolov11"           # Where YOLO dataset will be created
SEED = 42
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
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


OUR_DATASET_CLASSES = ['Amphora affinis', 'Amphora copulata', 'Amphora ovalis', 'Amphora pediculus', 'Aneumastus apiculatus', 'Aneumastus tusculus', 'Aulacoseira ambigua', 'Aulacoseira crassipunctata', 'Aulacoseira granulata', 'Aulacoseira granulata var angustissima', 'Aulacoseira italica', 'Aulacoseira subarctica', 'Cavinula scutelloides', 'Cocconeis placentula', 'Cymatopleuta elliptica', 'Cymbella neocistula', 'Cymbopleura inaequalis', 'Diploneis elliptica', 'Encyonema eglinense', 'Epithemia adnata', 'Epithemia frickei', 'Epithemia turguda', 'Gyrosigma attenuatum', 'Iconella bifrons', 'Iconella hibernica', 'Karayevia clevei', 'Lindavia affinis', 'Lindavia praetermissa', 'Martiy mutants', 'Navicula radiosa', 'Pantoscekiella ocellata', 'Paraplaconeis minor', 'Paraplaconeis placentula', 'Pinnularia gracilloides var triandulata', 'Placoneis gastrum', 'Pseudostaurosira brevistriata', 'Pseudostaurosira parasitica', 'Pseudostaurosira subconstricta', 'Staurosira construens', 'Staurosirella martyi', 'Staurosirella ovata', 'Stephanodiscus alpinus', 'Stephanodiscus neoastrea', 'Surirella librile', 'Ulnaria biceps', 'Ulnaria ulna']
KAGGLE_DATASET_CLASSES = ['Achnanthidium biasolettianum', 'Achnanthidium minutissimum', 'Adlafia minuscula', 'Amphora pediculus', 'Caloneis lancettula', 'Cocconeis pseudolineata', 'Cymbella cantonatii', 'Cymbella excisa', 'Cymbella excisa var. subcapitata', 'Denticula kuetzingii', 'Diatoma mesodon', 'Diatoma moniliformis', 'Encyonema silesiacum', 'Encyonema ventricosum', 'Epithemia argus', 'Fragilaria recapitellata', 'Frustulia vulgaris', 'Gomphonema calcifugum', 'Gomphonema drutelingense', 'Gomphonema micropus', 'Gomphonema minutum', 'Gomphonema olivaceum', 'Gomphonema pumilum', 'Gomphonema pumilum var. rigidum', 'Gomphonema supertergestinum', 'Gomphonema tergestinum', 'Halamphora paraveneta', 'Halamphora veneta', 'Hantzschiana abundans', 'Humidophila contenta', 'Humidophila perpusilla', 'Meridion circulare', 'Navicula capitatoradiata', 'Navicula cryptocephala', 'Navicula cryptotenella', 'Navicula cryptotenelloides', 'Navicula gregaria', 'Navicula moskalii', 'Navicula reichardtiana', 'Navicula tripunctata', 'Navicula trivialis', 'Navicula upsaliensis', 'Nitzschia archibaldii', 'Nitzschia hantzschiana', 'Nitzschia linearis', 'Pantocsekiella ocellata', 'Pinnularia brebissonii', 'Planothidium frequentissimum', 'Planothidium lanceolatum', 'Rhoicosphenia abbreviata', 'Surirella brebissonii var. kuetzingii']
DEFAULT_DETECTABLE_CLASSES = ['Amphora copulata', 'Amphora ovalis', 'Aulacoseira ambigua', 'Aulacoseira crassipunctata', 'Aulacoseira granulata', 'Cavinula scutelloides', 'Diploneis elliptica', 'Epithemia frickei', 'Iconella bifrons', 'Lindavia praetermissa', 'Pantoscekiella ocellata', 'Pseudostaurosira brevistriata', 'Staurosira construens', 'Staurosirella martyi', 'Stephanodiscus alpinus', 'Stephanodiscus neoastrea']
# Change if using kaggle dataset
TARGET_CLASSES = KAGGLE_DATASET_CLASSES

# Create class-to-id mapping
class_to_id = {name: idx for idx, name in enumerate(TARGET_CLASSES)}

DETECTION_CONFIG = {
    "data": str(OUTPUT_ROOT / "dataset_synthetic.yaml"),  # YOLO format dataset
    "epochs": 100,
    "imgsz": 640,
    "batch": 64,
    "lr0": 0.001,
    "patience": 3,
    "cos_lr": True,
    "warmup_epochs": 1,
    "project": "diatoms",
    "name": "v3",
    # Augmentation settings
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 15.0,
    "scale": 0.3,
    "fliplr": 0.5,
    "flipud": 0.5,
    "cls_pw": 0.0, # Class balancing
    "freeze": 2, # Freeze N backbone layers
    # "classes": [1, 2, 6, 7, 8, 12, 17, 20, 23, 27, 30, 35, 38, 39, 41, 42] # Classes to consider
}

CLASSIFICATION_CONFIG = {
    "data": str(OUTPUT_ROOT / "data.yaml"),   # YOLO format dataset
    "epochs": 50,
    "batch": 32,
    "lr": 0.001,
    "weight_decay": 1e-4,
    "patience": 10,
    "scheduler_patience": 2,
    "num_workers": 4,
    # Classes to consider
    # "classes": [1, 2, 6, 7, 8, 12, 17, 20, 23, 27, 30, 35, 38, 39, 41, 42],
    "save_path": str(OUTPUT_ROOT / "kaggle_diatomnet_classifier.pth"),
}

INPUT_SIZE = (128, 432)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


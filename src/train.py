"""Train and save SC-Diatomnet model."""

from src.models.SC_Diatomnet.models import YOLOv11Baseline
from .config import DATASET_ROOT, DEVICE, OUTPUT_ROOT


if __name__ == '__main__':
    model = YOLOv11Baseline(
    weights="yolo11n.pt",
    device=DEVICE,
    )

    model.train(
        data=DATASET_ROOT / "data.yaml",
        epochs=200,
        imgsz=640,
        batch=16,
    )
"""Train and save SC-Diatomnet model."""

from src.models.YOLO.model import YOLOv11Baseline

model = YOLOv11Baseline(
    weights="yolo11n.pt",
    device="cuda",
)

model.train(
    data="data.yaml",
    epochs=200,
    imgsz=640,
    batch=16,
)
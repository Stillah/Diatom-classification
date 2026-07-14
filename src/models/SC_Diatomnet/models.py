from src.models.base import SegmentModel

from pathlib import Path

from ultralytics import YOLO

from src.models.base import SegmentModel




class YOLOv11Baseline(SegmentModel):
    """
    Wrapper around pretrained Ultralytics YOLO11.
    """

    def __init__(
        self,
        weights: str = "yolo11n.pt",
        device: str = "cuda",
    ):
        self.device = device
        self.model = YOLO(weights)

    def train(
        self,
        data: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        project: str = "runs",
        name: str = "yolo11_baseline",
    ):
        return self.model.train(
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            project=project,
            name=name,
        )

    def validate(self):
        return self.model.val(device=self.device)

    def segment(self, image):
        return self.model.predict(
            source=image,
            device=self.device,
            verbose=False,
        )

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def load(self, weights: str):
        self.model = YOLO(weights)
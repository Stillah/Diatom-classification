from pathlib import Path
from typing import Union, Optional, Dict, Any, List
import numpy as np
from ultralytics import YOLO
from src.models.base import DetectModel

class YOLOv11Baseline(DetectModel):
    """
    Wrapper around pretrained Ultralytics YOLO11 for diatom detection.
    Supports training, validation, inference, save/load.
    """

    def __init__(
        self,
        device: str = "cpu",
        task: str = "detect",
        model_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the YOLO detection model.

        Args:
            model_path: Path to a YOLO weights file (.pt) or a pretrained string
                        like 'yolo11n.pt'. If None, model is not loaded until
                        load() is called.
            device: Device to run on ('cpu', 'cuda', 'mps', etc.).
            task: Should be 'detect' for detection.
        """
        super().__init__()
        self.device = device
        self.task = task
        self.model: Optional[YOLO] = None

        if model_path is not None:
            self.load(model_path)

    def train(self, train_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the YOLO detection model.

        Args:
            train_cfg: Dictionary with training parameters. Common keys:
                - data (str): Path to dataset YAML file.
                - epochs (int): Number of epochs.
                - imgsz (int): Input image size.
                - batch (int): Batch size.
                - device (str, optional): Override device.
                - project (str): Project name.
                - name (str): Experiment name.
                - ... (any other Ultralytics training argument)

        Returns:
            Dictionary with training results (metrics, etc.).
        """
        if self.model is None:
            # Load a default detection model
            self.load("yolo11m.pt")   # or yolo11s.pt, etc.

        if "device" not in train_cfg:
            train_cfg["device"] = self.device

        results = self.model.train(**train_cfg)
        return {
            "metrics": results.results_dict if hasattr(results, "results_dict") else {},
            "save_dir": str(results.save_dir) if hasattr(results, "save_dir") else None,
        }

    def validate(
        self,
        data: Optional[Union[str, Path]] = None,
        val_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Run validation on the validation set.

        Args:
            data: Path to dataset YAML (if not already set in model).
            val_cfg: Additional validation arguments (batch, imgsz, etc.).

        Returns:
            Dictionary of validation metrics (e.g., mAP, precision, recall).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        cfg = val_cfg or {}
        if data is not None:
            cfg["data"] = str(data)
        if "device" not in cfg:
            cfg["device"] = self.device

        results = self.model.val(**cfg)
        metrics = results.results_dict if hasattr(results, "results_dict") else {}
        return metrics

    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run detection inference on a single image.

        Args:
            image: Path to image file or numpy array (H,W,3) in BGR or RGB.
            conf: Confidence threshold.
            iou: IoU threshold for NMS.
            imgsz: Inference image size.
            **kwargs: Additional prediction arguments.

        Returns:
            Dictionary with:
                - 'boxes': list of bounding boxes [x1, y1, x2, y2] (absolute pixel coords).
                - 'confidences': list of confidence scores.
                - 'class_ids': list of class IDs (integers).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = self.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            **kwargs,
        )

        if not results:
            return {"boxes": [], "confidences": [], "class_ids": []}

        result = results[0]  # single image
        if result.boxes is None:
            return {"boxes": [], "confidences": [], "class_ids": []}

        boxes = [box.xyxy.cpu().numpy().squeeze().tolist() for box in result.boxes]
        confidences = result.boxes.conf.cpu().numpy().tolist()
        class_ids = result.boxes.cls.cpu().numpy().astype(int).tolist()

        return {
            "boxes": boxes,
            "confidences": confidences,
            "class_ids": class_ids,
        }

    def predict_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Run batch inference on multiple images.

        Args:
            images: List of image paths or numpy arrays.
            **kwargs: Prediction arguments (conf, iou, etc.).

        Returns:
            List of dictionaries, each containing boxes, confidences, class_ids.
        """
        results = self.model.predict(source=images, device=self.device, **kwargs)
        outputs = []
        for result in results:
            if result.boxes is None:
                outputs.append({"boxes": [], "confidences": [], "class_ids": []})
                continue
            outputs.append({
                "boxes": [box.xyxy.cpu().numpy().squeeze().tolist() for box in result.boxes],
                "confidences": result.boxes.conf.cpu().numpy().tolist(),
                "class_ids": result.boxes.cls.cpu().numpy().astype(int).tolist(),
            })
        return outputs

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the model weights to a .pt file.

        Args:
            path: Output file path (should end with .pt).
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Nothing to save.")
        self.model.save(str(path))

    def load(self, weights: Union[str, Path]) -> None:
        """
        Load model weights from a .pt file or a pretrained string.

        Args:
            weights: Path to weights file or a pretrained model name
                     (e.g., 'yolo11n.pt').
        """
        self.model = YOLO(str(weights), task=self.task)
        if self.device != "cpu":
            self.model.to(self.device)

    def get_class_names(self) -> List[str]:
        """Get the names of the classes the model can predict."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        return list(self.model.names.values())
        

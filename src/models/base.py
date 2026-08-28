
from typing import Protocol
from abc import ABC, abstractmethod 


# models/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Union, Optional, List
from pathlib import Path
import numpy as np

class DetectModel(ABC):
    """
    Abstract base class for object detection models.
    All detection wrappers (e.g., YOLOv11Baseline) must implement these methods.
    """

    @abstractmethod
    def train(self, train_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the detection model.

        Args:
            train_cfg: Dictionary of training hyperparameters.

        Returns:
            Dictionary with training results (metrics, save paths, etc.).
        """
        pass

    @abstractmethod
    def validate(
        self,
        data: Optional[Union[str, Path]] = None,
        val_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on a validation set.

        Args:
            data: Path to dataset YAML file.
            val_cfg: Additional validation arguments.

        Returns:
            Dictionary of evaluation metrics (e.g., mAP, precision, recall).
        """
        pass

    @abstractmethod
    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run inference on a single image.

        Args:
            image: Path to image or numpy array (H,W,3).
            **kwargs: Inference parameters (conf, iou, imgsz, etc.).

        Returns:
            Dictionary with at least 'boxes', 'confidences', 'class_ids'.
        """
        pass

    @abstractmethod
    def predict_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of images.

        Args:
            images: List of image paths or numpy arrays.
            **kwargs: Inference parameters.

        Returns:
            List of dictionaries, each with 'boxes', 'confidences', 'class_ids'.
        """
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Save model weights to a file."""
        pass

    @abstractmethod
    def load(self, weights: Union[str, Path]) -> None:
        """Load model weights from a file or pretrained identifier."""
        pass

    @abstractmethod
    def get_class_names(self) -> List[str]:
        """Get the names of the classes the model can predict."""
        pass

class ClassificationModel(Protocol):

    @abstractmethod
    def classify(self, image) -> str:
        pass
    
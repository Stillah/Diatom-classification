"""Train and save a YOLOv11 detection model (e.g., SC‑DiatomNet baseline)."""

from models.SC_Diatomnet.models import YOLOv11Baseline
from config import DEVICE, OUTPUT_ROOT, TRAIN_CONFIG

if __name__ == "__main__":
    print("Training started")

    # 1. Initialise the model (detection variant)
    model = YOLOv11Baseline(device=DEVICE, model_path="yolo11n.pt")

    # 2. Train (uses TRAIN_CONFIG from config.py)
    model.train(TRAIN_CONFIG)

    # 3. Validate on the validation set
    metrics = model.validate()
    print("Validation mAP@50-95:", metrics.get("mAP50-95", "N/A"))
    print("Validation mAP@50:", metrics.get("mAP50", "N/A"))


    # 5. Save the trained weights
    save_path = OUTPUT_ROOT / "best_diatom.pt"
    model.save(save_path)
    print(f"Model saved to {save_path}")

    # 6. (Optional) demonstrate loading the saved model
    # new_model = YOLOv11Baseline()
    # new_model.load(save_path)
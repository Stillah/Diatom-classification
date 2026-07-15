"""Тестирование обученной модели YOLOv11 на train, val и test наборах."""

from pathlib import Path
import sys
# sys.path.insert(0, str(Path(__file__).parent.parent))

from models.SC_Diatomnet.models import YOLOv11Baseline
from config import DEVICE, OUTPUT_ROOT, DATASET_ROOT

# Пути к модели и файлу с данными (измените при необходимости)
MODEL_PATH = OUTPUT_ROOT / "best_diatom.pt"
DATA_YAML = OUTPUT_ROOT / "data.yaml"

# Загружаем модель (автоматически определяет задачу как detect)
model = YOLOv11Baseline(device=DEVICE, model_path=MODEL_PATH)

# Метрики, которые нас интересуют
# metric_keys = ["mAP50-95", "mAP50", "precision", "recall"]

# Проверяем каждый сплит
for split in ["train", "val", "test"]:
    print(f"\n{'='*50}")
    print(f"Оценка на наборе: {split.upper()}")
    print("="*50)

    try:
        # Запускаем валидацию с указанием сплита
        metrics = model.validate(
            data=DATA_YAML,
            val_cfg={"split": split, "verbose": True}  # verbose=False скрывает лишние логи
        )

        # Выводим интересующие метрики
        for key in metrics.keys():
            value = metrics.get(key, "N/A")
            if isinstance(value, float):
                print(f"{key:>12}: {value:.4f}")
            else:
                print(f"{key:>12}: {value}")

    except Exception as e:
        print(f"Ошибка при оценке {split}: {e}")
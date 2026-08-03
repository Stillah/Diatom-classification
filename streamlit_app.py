from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Root-level Streamlit secrets are also copied to environment variables here
# so third-party libraries such as ClearML can read them reliably.
try:
    for secret_name in (
        "CLEARML_API_ACCESS_KEY",
        "CLEARML_API_SECRET_KEY",
        "CLEARML_API_HOST",
        "CLEARML_WEB_HOST",
        "CLEARML_FILES_HOST",
        "CLEARML_TASK_ID",
        "DIATOM_DETECTOR",
        "DIATOM_CLASSIFIER",
    ):
        if secret_name in st.secrets and secret_name not in os.environ:
            os.environ[secret_name] = str(st.secrets[secret_name])
except FileNotFoundError:
    pass

from clearml_utils import get_metrics  # noqa: E402

st.set_page_config(
    page_title="Diatom classifier",
    page_icon="🔬",
    layout="wide",
)


def _model_path(env_name: str, default: str) -> Path:
    value = os.getenv(env_name, default)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"Файл модели не найден: {path}. "
            f"Добавьте файл в приложение или задайте секрет {env_name}."
        )
    return path


@st.cache_resource(show_spinner=False)
def load_pipeline() -> Any:
    # Heavy computer-vision imports are intentionally lazy. This lets the
    # ClearML metrics page start even when model weights are not configured yet.
    from config import DEVICE
    from pipeline import DiatomPipeline

    detector = _model_path(
        "DIATOM_DETECTOR",
        "runs/detect/train/weights/best.pt",
    )
    classifier = _model_path(
        "DIATOM_CLASSIFIER",
        "best_diatomnet.pth",
    )
    pipeline = DiatomPipeline(device=DEVICE)
    pipeline.load(
        detector_path=detector,
        classifier_path=classifier,
    )
    return pipeline


@st.cache_data(ttl=60, show_spinner=False)
def load_metrics(task_id: str) -> pd.DataFrame:
    return get_metrics(task_id)


def draw_predictions(
    image_rgb: np.ndarray,
    result: dict,
) -> np.ndarray:
    annotated = Image.fromarray(image_rgb.astype(np.uint8, copy=False)).copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size

    for box, name, confidence in zip(
        result["boxes"],
        result["class_names"],
        result["confidences"],
    ):
        x1, y1, x2, y2 = [int(value) for value in box]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        label = f"{name}: {confidence:.2f}"
        draw.rectangle((x1, y1, x2, y2), outline=(0, 220, 0), width=3)
        draw.text(
            (x1, max(0, y1 - 18)),
            label,
            fill=(0, 255, 0),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

    return np.asarray(annotated)


st.title("🔬 Automatic Diatom Classification")
st.write("YOLOv11 detection → DiatomNet species classification")

mode = st.sidebar.radio(
    "Режим",
    ["Инференс", "Метрики обучения"],
)

if mode == "Инференс":
    det_conf = st.sidebar.slider(
        "Порог confidence детектора",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
    )
    det_iou = st.sidebar.slider(
        "IoU для NMS",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
    )
    use_classifier = st.sidebar.checkbox(
        "Использовать DiatomNet",
        value=True,
    )

    uploaded = st.file_uploader(
        "Загрузите микроскопическое изображение",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image_rgb = np.asarray(image)
        # Pipeline historically expects OpenCV-style BGR arrays.
        image_bgr = image_rgb[:, :, ::-1].copy()

        try:
            with st.spinner("Выполняется инференс..."):
                pipeline = load_pipeline()
                result = pipeline.predict(
                    image_bgr,
                    det_conf=det_conf,
                    det_iou=det_iou,
                    use_classifier=use_classifier,
                )
        except Exception as exc:
            st.error("Не удалось запустить инференс")
            st.exception(exc)
        else:
            if not result["boxes"]:
                st.warning("Диатомеи не обнаружены")
                st.image(image_rgb, caption="Исходное изображение")
            else:
                annotated = draw_predictions(image_rgb, result)
                left, right = st.columns(2)
                with left:
                    st.image(image_rgb, caption="Исходное изображение")
                with right:
                    st.image(annotated, caption="Результат pipeline")

                rows = []
                for index, (name, confidence, det_confidence, box) in enumerate(
                    zip(
                        result["class_names"],
                        result["confidences"],
                        result["detection_confidences"],
                        result["boxes"],
                    ),
                    start=1,
                ):
                    rows.append(
                        {
                            "object": index,
                            "species": name,
                            "classification_confidence": confidence,
                            "detection_confidence": det_confidence,
                            "bbox": [round(value, 1) for value in box],
                        }
                    )
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True,
                )

else:
    default_task_id = os.getenv("CLEARML_TASK_ID", "")
    task_id = st.text_input(
        "ClearML Task ID",
        value=default_task_id,
    ).strip()

    if not task_id:
        st.info("Введите ClearML Task ID, чтобы загрузить графики обучения.")
    else:
        try:
            metrics = load_metrics(task_id)
        except Exception as exc:
            st.error("Не удалось загрузить метрики ClearML")
            st.exception(exc)
        else:
            if metrics.empty:
                st.warning("В эксперименте пока нет scalar-метрик")
            else:
                for metric_name in metrics["metric"].drop_duplicates():
                    metric_frame = metrics[
                        metrics["metric"] == metric_name
                    ].pivot_table(
                        index="step",
                        columns="series",
                        values="value",
                        aggfunc="last",
                    )
                    st.subheader(metric_name)
                    st.line_chart(metric_frame)

                with st.expander("Сырые значения"):
                    st.dataframe(metrics, use_container_width=True)

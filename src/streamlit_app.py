from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
        path = PROJECT_ROOT / path
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
    from src.config import DEVICE
    from src.models.pipeline import DiatomPipeline

    detector = _model_path(
        "DIATOM_DETECTOR",
        "model_weights/best_detector.pt",
    )
    classifier = _model_path(
        "DIATOM_CLASSIFIER",
        "model_weights/best_classifier.pth",
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
        draw.rectangle((x1, y1, x2, y2), outline=(0, 220, 0), width=15)
        draw.text(
            (x1, max(0, y1 - 100)),
            label,
            font=ImageFont.truetype("arial.ttf", 100),
            fill=(155, 55, 0),
            stroke_width=1,
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
        "Intersection over Union для Non-Maximum Suppression",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
    )
    use_classifier = st.sidebar.checkbox(
        "Использовать отдельный DiatomNet (обычно работает хуже)",
        value=False,
    )


    uploaded = st.file_uploader(
        "Загрузите одно или несколько изображений диатомий",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
    )
    run_batch = st.button("Запустить прогноз", width="stretch", type="primary")

    if uploaded and run_batch:
        batch_images = []
        for uploaded_file in uploaded:
            image = Image.open(uploaded_file).convert("RGB")
            image_rgb = np.asarray(image)
            batch_images.append(image_rgb[:, :, ::-1].copy())

        try:
            with st.spinner("Выполняется предсказание..."):
                pipeline = load_pipeline()
                results = pipeline.predict_batch(
                    batch_images,
                    det_conf=det_conf,
                    det_iou=det_iou,
                    use_classifier=use_classifier,
                )
        except Exception as exc:
            st.error("Не удалось запустить предсказание")
            st.exception(exc)
        else:
            detection_rows = []
            image_views = []

            for index, (uploaded_file, result) in enumerate(zip(uploaded, results), start=1):
                image_rgb = np.asarray(Image.open(uploaded_file).convert("RGB"))
                if result["boxes"]:
                    annotated = draw_predictions(image_rgb, result)
                else:
                    annotated = image_rgb

                image_views.append(
                    {
                        "index": index,
                        "name": uploaded_file.name,
                        "original": image_rgb,
                        "annotated": annotated,
                        "has_detections": bool(result["boxes"]),
                    }
                )

                for box_index, (name, confidence, det_confidence, box) in enumerate(
                    zip(
                        result["class_names"],
                        result["confidences"],
                        result["detection_confidences"],
                        result["boxes"],
                    ),
                    start=1,
                ):
                    detection_rows.append(
                        {
                            "image": uploaded_file.name,
                            "object": box_index,
                            "class_name": name,
                            "classification_confidence": confidence,
                            "detection_confidence": det_confidence,
                            "bbox": [round(value, 1) for value in box],
                        }
                    )

            detection_df = pd.DataFrame(detection_rows)
            if detection_df.empty:
                summary_df = pd.DataFrame(
                    columns=[
                        "class_name",
                        "count",
                        "average classification confidence",
                        "average detection confidence",
                    ]
                )
            else:
                summary_df = (
                    detection_df.groupby("class_name", as_index=False)
                    .agg(
                        count=("class_name", "size"),
                        average_classification_confidence=("classification_confidence", "mean"),
                        average_detection_confidence=("detection_confidence", "mean"),
                    )
                    .rename(
                        columns={
                            "average_classification_confidence": "average classification confidence",
                            "average_detection_confidence": "average detection confidence",
                        }
                    )
                    .sort_values("count", ascending=False)
                )

            st.subheader("Class summary")
            st.dataframe(summary_df, width='stretch', hide_index=True)

            st.subheader("All detections")
            st.dataframe(detection_df, width='stretch', hide_index=True)

            st.subheader("Images")
            for start in range(0, len(image_views), 2):
                row_items = image_views[start:start + 2]
                cols = st.columns(2)
                for col_index, item in enumerate(row_items):
                    with cols[col_index]:
                        st.image(item["annotated"], caption=f"{item['index']}. {item['name']}")
                        if not item["has_detections"]:
                            st.caption("No diatoms detected")

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
                    st.dataframe(metrics, width='stretch')


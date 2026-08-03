from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from clearml_utils import get_metrics  # noqa: E402
from config import DEVICE  # noqa: E402
from pipeline import DiatomPipeline  # noqa: E402

st.set_page_config(
    page_title="Diatom classifier",
    page_icon="🔬",
    layout="wide",
)


def _model_path(env_name: str, default: str) -> Path:
    value = os.getenv(env_name, default)
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Файл модели не найден: {path}. Задайте переменную {env_name}."
        )
    return path


@st.cache_resource(show_spinner=False)
def load_pipeline() -> DiatomPipeline:
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
    annotated = image_rgb.copy()
    height, width = annotated.shape[:2]

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
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 3)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


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
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

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

    if task_id:
        try:
            metrics = load_metrics(task_id)
        except Exception as exc:
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

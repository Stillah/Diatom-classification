from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any
from collections import defaultdict

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

# ---------- Environment & paths ----------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv()

# ---------- Imports from project ----------
from src.clearml_utils import get_metrics
from src.config import OUR_DATASET_CLASSES, TARGET_CLASSES

# Copy Streamlit secrets to environment for third‑party libraries
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


# ---------- Helper functions ----------
def _model_path(env_name: str, default: str) -> Path:
    """Resolve model path from environment variable or fallback to default."""
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


def get_available_detectors() -> list[Path]:
    """Return sorted list of .pt files in model_weights/detectors/."""
    detector_dir = PROJECT_ROOT / "model_weights" / "detectors"
    if not detector_dir.exists():
        return []
    return sorted(detector_dir.glob("*.pt"))


@st.cache_resource(show_spinner=False)
def load_pipeline(detector_path: Path | None = None) -> Any:
    """
    Load the detection + classification pipeline (cached by detector path).
    If detector_path is None, use the default from environment / fallback.
    """
    from src.config import DEVICE
    from src.models.pipeline import DiatomPipeline

    if detector_path is None:
        detector_path = _model_path(
            "DIATOM_DETECTOR",
            "model_weights/best_detector.pt",
        )
    else:
        detector_path = detector_path.resolve()
        if not detector_path.exists():
            raise FileNotFoundError(f"Выбранный детектор не найден: {detector_path}")

    classifier_path = _model_path(
        "DIATOM_CLASSIFIER",
        "model_weights/best_classifier.pth",
    )

    print(f"Loading detector from {detector_path}")
    print(f"Loading classifier from {classifier_path}")

    pipeline = DiatomPipeline(device=DEVICE, class_names=OUR_DATASET_CLASSES)
    pipeline.load(
        detector_path=detector_path,
        classifier_path=classifier_path,
    )
    return pipeline


@st.cache_data(ttl=60, show_spinner=False)
def load_metrics(task_id: str) -> pd.DataFrame:
    """Fetch scalar metrics from ClearML (cached for 60 seconds)."""
    return get_metrics(task_id)


def draw_predictions(
    image_rgb: np.ndarray,
    result: dict,
) -> np.ndarray:
    """Draw bounding boxes and labels on an RGB image."""
    annotated = Image.fromarray(image_rgb.astype(np.uint8, copy=False)).copy()
    draw = ImageDraw.Draw(annotated)
    width, height = annotated.size

    try:
        font = ImageFont.truetype("arial.ttf", 100)
    except IOError:
        font = ImageFont.load_default()

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
            font=font,
            fill=(155, 55, 0),
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )

    return np.asarray(annotated)


# ---------- Main UI ----------
def main() -> None:
    st.set_page_config(
        page_title="Diatom classifier",
        page_icon="🔬",
        layout="wide",
    )
    st.title("🔬 Автоматическая классификация диатомий")

    # Initialise session state variables
    if "detection_rows" not in st.session_state:
        st.session_state.detection_rows = []
    if "image_views" not in st.session_state:
        st.session_state.image_views = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "image_classes" not in st.session_state:
        st.session_state.image_classes = []
    if "class_counts" not in st.session_state:
        st.session_state.class_counts = {}
    if "count_keys_initialized" not in st.session_state:
        st.session_state.count_keys_initialized = False

    mode = st.sidebar.radio(
        "Режим",
        ["Предсказание", "Метрики обучения"],
    )

    if mode == "Предсказание":
        # ---------- Sidebar controls ----------
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
            "Использовать DiatomNet как классификатор (обычно работает хуже)",
            value=False,
        )

        # Detector selection
        available_detectors = get_available_detectors()
        if available_detectors:
            default_env = os.getenv("DIATOM_DETECTOR", "model_weights/best_detector.pt")
            default_path = Path(default_env).expanduser()
            if not default_path.is_absolute():
                default_path = PROJECT_ROOT / default_path
            default_index = 0
            for i, p in enumerate(available_detectors):
                if p == default_path:
                    default_index = i
                    break
            detector_names = [p.name for p in available_detectors]
            selected_name = st.sidebar.selectbox(
                "Модель детектора",
                options=detector_names,
                index=default_index,
            )
            selected_detector_path = available_detectors[detector_names.index(selected_name)]
        else:
            st.sidebar.warning(
                "Папка model_weights/detectors/ не найдена или не содержит .pt файлов. "
                "Используется детектор по умолчанию."
            )
            selected_detector_path = None
            
        with st.expander("Распозноваемые классы", expanded=False):
            st.markdown("\n".join(f"- **{c}**" for c in TARGET_CLASSES))

        # ---------- File upload & inference ----------
        uploaded = st.file_uploader(
            "Загрузите одно или несколько изображений диатомий",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            accept_multiple_files=True,
        )
        run_batch = st.button("Запустить прогноз", width="stretch", type="primary")

        # If a new batch is run, clear the previous results
        if run_batch and uploaded:
            # Remove all old count widget keys to avoid interference
            for key in list(st.session_state.keys()):
                if key.startswith("count_"):
                    del st.session_state[key]

            st.session_state.detection_rows = []
            st.session_state.image_views = []
            st.session_state.uploaded_files = []
            st.session_state.image_classes = []
            st.session_state.class_counts = {}
            st.session_state.count_keys_initialized = False

            # Prepare images
            batch_images = []
            for uploaded_file in uploaded:
                image = Image.open(uploaded_file).convert("RGB")
                image_rgb = np.asarray(image)
                batch_images.append(image_rgb[:, :, ::-1].copy())  # RGB → BGR

            # Run inference
            try:
                with st.spinner("Выполняется предсказание..."):
                    pipeline = load_pipeline(detector_path=selected_detector_path)
                    results = pipeline.predict_batch(
                        batch_images,
                        det_conf=det_conf,
                        det_iou=det_iou,
                        use_classifier=use_classifier,
                    )
            except Exception as exc:
                st.error("Не удалось запустить предсказание")
                st.exception(exc)
                st.stop()

            # Build detection rows and image views
            detection_rows = []
            image_views = []
            image_classes = []  # list of lists: classes per image

            for idx, (uploaded_file, result) in enumerate(zip(uploaded, results), start=1):
                image_rgb = np.asarray(Image.open(uploaded_file).convert("RGB"))
                if result["boxes"]:
                    annotated = draw_predictions(image_rgb, result)
                else:
                    annotated = image_rgb

                image_views.append(
                    {
                        "index": idx,
                        "name": uploaded_file.name,
                        "original": image_rgb,
                        "annotated": annotated,
                        "has_detections": bool(result["boxes"]),
                    }
                )

                # Get unique class names in this image
                unique_classes = []
                class_counts_this = defaultdict(int)
                for name, conf, det_conf_val, box in zip(
                    result["class_names"],
                    result["confidences"],
                    result["detection_confidences"],
                    result["boxes"],
                ):
                    if name not in class_counts_this:
                        unique_classes.append(name)
                    class_counts_this[name] += 1
                    detection_rows.append(
                        {
                            "image": uploaded_file.name,
                            "image_index": idx,
                            "object": len(detection_rows) + 1,
                            "class_name": name,
                            "classification_confidence": conf,
                            "detection_confidence": det_conf_val,
                            "bbox": [round(value, 1) for value in box],
                        }
                    )
                image_classes.append(unique_classes)

                # Store counts and initialise widget keys
                for cls, cnt in class_counts_this.items():
                    img_idx_0 = idx - 1
                    st.session_state.class_counts[(img_idx_0, cls)] = cnt
                    widget_key = f"count_{img_idx_0}_{cls}"
                    st.session_state[widget_key] = cnt

            # Store in session state
            st.session_state.detection_rows = detection_rows
            st.session_state.image_views = image_views
            st.session_state.uploaded_files = uploaded
            st.session_state.image_classes = image_classes
            st.session_state.count_keys_initialized = True

        # ---------- Display results from session state (if any) ----------
        if st.session_state.detection_rows:
            image_views = st.session_state.image_views
            detection_rows = st.session_state.detection_rows
            image_classes = st.session_state.image_classes

            st.subheader("Images")
            for start in range(0, len(image_views), 2):
                row_items = image_views[start:start + 2]
                cols = st.columns(2)
                for col_idx, item in enumerate(row_items):
                    with cols[col_idx]:
                        idx_0 = item["index"] - 1
                        st.image(
                            item["annotated"],
                            caption=f"{item['index']}. {item['name']}",
                        )
                        if not item["has_detections"]:
                            st.caption("No diatoms detected")

                        # Show class count controls for this image
                        if st.session_state.count_keys_initialized:
                            classes_this = image_classes[idx_0] if idx_0 < len(image_classes) else []
                            if classes_this:
                                st.markdown(
                                    '<span style="font-size: 1.85em; color: #A1ABB2;">**Редактировать количество**</span>',
                                    unsafe_allow_html=True,
                                )
                                # st.divider()
                                # Compact layout: each class in a row with label + number input
                                for class_name in sorted(classes_this):
                                    col_label, col_input = st.columns([2, 1])
                                    with col_label:
                                        st.markdown(f"**{class_name}**")
                                    with col_input:
                                        widget_key = f"count_{idx_0}_{class_name}"
                                        st.number_input(
                                            "Count",  # Non‑empty label (hidden)
                                            min_value=0,
                                            step=1,
                                            key=widget_key,
                                            label_visibility="collapsed",
                                        )
                                    # Sync widget value back to class_counts
                                    if widget_key in st.session_state:
                                        st.session_state.class_counts[(idx_0, class_name)] = st.session_state[widget_key]

                            # ---------- Add class section ----------
                            st.markdown(
                                '<span style="font-size: 1.5em; color: #86888A;">**Добавить класс**</span>',
                                unsafe_allow_html=True,
                            )
                            available_add = [c for c in TARGET_CLASSES if c not in classes_this]
                            if not available_add:
                                st.info("Все классы уже добавлены для этого изображения.")
                            else:
                                # Compact row for add: dropdown + button
                                col_drop, col_btn = st.columns([3, 1])
                                with col_drop:
                                    selected_add_class = st.selectbox(
                                        "Выберите класс для добавления",
                                        options=available_add,
                                        key=f"add_select_{idx_0}",
                                        label_visibility="collapsed",
                                        placeholder="Выберите класс...",
                                    )
                                with col_btn:
                                    if st.button("➕ Добавить", key=f"add_button_{idx_0}", use_container_width=True):
                                        if selected_add_class not in image_classes[idx_0]:
                                            image_classes[idx_0].append(selected_add_class)
                                        count_key = (idx_0, selected_add_class)
                                        if count_key in st.session_state.class_counts:
                                            st.session_state.class_counts[count_key] += 1
                                        else:
                                            st.session_state.class_counts[count_key] = 1
                                        widget_key = f"count_{idx_0}_{selected_add_class}"
                                        st.session_state[widget_key] = st.session_state.class_counts[count_key]
                                        st.rerun()
                            st.divider()

            # ---------- Compute summary from adjusted counts ----------
            adjusted_counts = defaultdict(int)

            # Aggregate counts from class_counts dict (all images included)
            for (_, class_name), count in st.session_state.class_counts.items():
                adjusted_counts[class_name] += count

            # Compute average confidences from all detections
            if detection_rows:
                df_all = pd.DataFrame(detection_rows)
                avg_class_conf = df_all.groupby("class_name")["classification_confidence"].mean()
                avg_det_conf = df_all.groupby("class_name")["detection_confidence"].mean()
            else:
                avg_class_conf = pd.Series(dtype=float)
                avg_det_conf = pd.Series(dtype=float)

            # Build summary DataFrame – for added classes, set confidences to 1.0
            if adjusted_counts:
                summary_data = []
                for class_name, count in adjusted_counts.items():
                    if count == 0:
                        continue
                    class_conf = avg_class_conf.get(class_name, 1.0)
                    det_conf = avg_det_conf.get(class_name, 1.0)
                    summary_data.append({
                        "class_name": class_name,
                        "count": count,
                        "average classification confidence": class_conf,
                        "average detection confidence": det_conf,
                    })
                if summary_data:
                    summary_df = pd.DataFrame(summary_data).sort_values("class_name", ascending=True)
                else:
                    summary_df = pd.DataFrame(
                        columns=[
                            "class_name",
                            "count",
                            "average classification confidence",
                            "average detection confidence",
                        ]
                    )
            else:
                summary_df = pd.DataFrame(
                    columns=[
                        "class_name",
                        "count",
                        "average classification confidence",
                        "average detection confidence",
                    ]
                )

            st.subheader("Class summary")
            st.dataframe(summary_df, width="stretch", hide_index=True)

            # Show all detections (only model detections, unchanged)
            st.subheader("All detections")
            detection_df = pd.DataFrame(detection_rows)
            st.dataframe(detection_df, width="stretch", hide_index=True)

    else:
        # ---------- Metrics mode ----------
        default_task_id = os.getenv("CLEARML_TASK_ID", "")
        task_id = st.text_input(
            "ClearML Task ID",
            value=default_task_id,
        ).strip()

        if not task_id:
            st.info("Введите ClearML Task ID, чтобы загрузить графики обучения.")
        elif re.fullmatch(r"[a-fA-F0-9]{32}", task_id) is None:
            st.error("Некорректный ClearML Task ID. Ожидается 32-символьный hex-идентификатор.")
        else:
            try:
                with st.spinner("Загружаю метрики из ClearML..."):
                    metrics = load_metrics(task_id)
            except Exception as exc:
                st.error("Не удалось загрузить метрики ClearML")
                st.exception(exc)
                st.stop()

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
                    st.dataframe(metrics, width="stretch")


if __name__ == "__main__":
    main()
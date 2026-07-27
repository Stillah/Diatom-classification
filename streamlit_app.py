from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.clearml_utils import get_metrics, load_clearml_model
from src.pipeline import DiatomPipeline
from src.config import TARGET_CLASSES, DEVICE

st.set_page_config(page_title="Diatom classifier", page_icon="🔬", layout="wide")


@st.cache_resource

def load_pipeline():
    detector = os.getenv("DIATOM_DETECTOR", "runs/detect/train/weights/best.pt")
    classifier = os.getenv("DIATOM_CLASSIFIER", "best_diatomnet.pth")
    pipeline = DiatomPipeline(device=DEVICE)
    pipeline.load(detector_path=detector, classifier_path=classifier)
    return pipeline


st.title("🔬 Automatic Diatom Classification")
st.write("YOLOv11 detection → DiatomNet species classification")

mode = st.sidebar.radio("Mode", ["Inference", "Training metrics"])

if mode == "Inference":
    uploaded = st.file_uploader("Upload microscope image", type=["png", "jpg", "jpeg", "tif", "tiff"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Input image", use_container_width=True)

        pipeline = load_pipeline()
        result = pipeline.predict(image)

        if not result["boxes"]:
            st.warning("No diatoms detected")
        else:
            rows = []
            for name, conf, box in zip(result["class_names"], result["confidences"], result["boxes"]):
                rows.append({"species": name, "confidence": conf, "bbox": box})
            st.dataframe(pd.DataFrame(rows))

else:
    task_id = st.text_input("ClearML task id")
    if task_id:
        df = get_metrics(task_id)
        st.line_chart(df.set_index("step"))

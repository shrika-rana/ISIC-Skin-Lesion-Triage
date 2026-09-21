import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "deployment_model.keras"
CONFIG_PATH = PROJECT_DIR / "artifacts" / "deployment_config.json"
RESULTS_PATH = PROJECT_DIR / "artifacts" / "final_test_results.csv"
LATENCY_PATH = PROJECT_DIR / "artifacts" / "latency.json"

st.set_page_config(
    page_title="ISIC Skin Lesion Triage Assistant",
    layout="wide",
)

st.title("ISIC Skin Lesion Triage Assistant")
st.caption(
    "Educational/research prototype. It is not a medical device and must not "
    "be used as a substitute for diagnosis by a qualified clinician."
)

if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
    st.error(
        "Trained model/configuration not found. Run train_isic.py first and "
        "keep the generated models/ and artifacts/ folders beside app.py."
    )
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

model = load_model()
config = load_config()

threshold = float(config["threshold"])
img_size = tuple(config["image_size"])

left, right = st.columns([1, 1])

with left:
    uploaded = st.file_uploader(
        "Upload a dermoscopic skin-lesion image",
        type=["jpg", "jpeg", "png", "webp"],
    )

with right:
    st.subheader("Deployment settings")
    st.write(f"Model: **{config['model_name']}**")
    st.write(f"Malignant triage threshold: **{threshold:.3f}**")
    st.write(
        "The threshold was selected from validation data to prioritize "
        "malignant recall/sensitivity."
    )

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    display_image = image.copy()

    image = image.resize(img_size)
    arr = np.asarray(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)

    probability = float(model.predict(arr, verbose=0)[0][0])
    pred_malignant = probability >= threshold

    c1, c2 = st.columns([1, 1])

    with c1:
        st.image(display_image, caption="Uploaded image", use_container_width=True)

    with c2:
        st.metric("Predicted malignant probability", f"{probability:.1%}")
        st.progress(min(max(probability, 0.0), 1.0))

        if pred_malignant:
            st.error(
                "TRIAGE FLAG: model score is at/above the malignant-review threshold."
            )
        else:
            st.success(
                "Below the malignant-review threshold used by this research model."
            )

        st.write(
            "This output indicates only the trained model's score. "
            "It does not establish or exclude cancer."
        )

st.divider()
st.subheader("Performance dashboard")

if RESULTS_PATH.exists():
    results = pd.read_csv(RESULTS_PATH)
    display_cols = [
        c for c in [
            "model",
            "threshold",
            "accuracy",
            "precision",
            "recall_sensitivity",
            "specificity",
            "f1",
            "roc_auc",
            "pr_auc",
            "fn",
            "fp",
        ]
        if c in results.columns
    ]
    st.dataframe(results[display_cols], use_container_width=True)

if LATENCY_PATH.exists():
    with open(LATENCY_PATH, "r") as f:
        latency = json.load(f)
    st.write(
        f"Measured mean inference time: "
        f"**{latency['mean_ms_per_image']:.2f} ms/image** "
        "(hardware-dependent)."
    )

st.info(
    "Recommended academic deployment framing: decision-support/triage only, "
    "human review required, threshold monitored on the target population, "
    "external validation required before any real clinical use."
)

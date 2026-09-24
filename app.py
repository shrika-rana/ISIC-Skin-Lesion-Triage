import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "deployment_model.keras"
MODEL_CANDIDATES = [
    MODEL_PATH,
    PROJECT_DIR / "models" / "resnet50_head.keras",
    PROJECT_DIR / "models" / "resnet50_finetuned.keras",
    PROJECT_DIR / "models" / "final_regularized_cnn.keras",
]
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
    last_error = None
    for candidate in MODEL_CANDIDATES:
        if not candidate.exists():
            continue
        try:
            return tf.keras.models.load_model(candidate)
        except Exception as exc:  # pragma: no cover - runtime fallback logic
            last_error = exc
    raise ValueError(
        "No valid saved model could be loaded. Checked: "
        + ", ".join(str(p) for p in MODEL_CANDIDATES)
    ) from last_error

@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

model = load_model()
config = load_config()

threshold = float(config["threshold"])
img_size = tuple(config["image_size"])

selected_model_metrics = {}
if RESULTS_PATH.exists():
    results = pd.read_csv(RESULTS_PATH)
    matches = results[results["model"].astype(str).str.lower() == str(config["model_name"]).lower()]
    if not matches.empty:
        selected_model_metrics = matches.iloc[0].to_dict()
    elif not results.empty:
        selected_model_metrics = results.iloc[0].to_dict()

left, right = st.columns([1, 1])

with left:
    uploaded = st.file_uploader(
        "Upload a dermoscopic skin-lesion image",
        type=["jpg", "jpeg", "png", "webp"],
    )

with right:
    st.subheader("Deployment settings")
    st.write(f"Selected deployment model: **{config['model_name']}**")
    st.write(f"Malignant triage threshold: **{threshold:.3f}**")
    st.write(
        "Deployment decision: the saved ResNet-50 head was selected as the best "
        "verified candidate because it maintains recall while improving the "
        "specificity/accuracy trade-off relative to the regularized CNN baseline."
    )

    if selected_model_metrics:
        st.write("Verified metric snapshot:")
        metric_cols = st.columns(4)
        metric_values = [
            ("Accuracy", selected_model_metrics.get("accuracy", np.nan)),
            ("Recall", selected_model_metrics.get("recall_sensitivity", np.nan)),
            ("Specificity", selected_model_metrics.get("specificity", np.nan)),
            ("PR-AUC", selected_model_metrics.get("pr_auc", np.nan)),
        ]
        for idx, (label, value) in enumerate(metric_values):
            metric_cols[idx].metric(label, f"{float(value):.3f}" if pd.notna(value) else "n/a")

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
st.info(
    "Current deployment choice: ResNet-50 head (saved model) with threshold 0.37. "
    "The regularized CNN is retained only as a baseline comparison and is not the "
    "selected deployment model."
)

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
    comparison = results[display_cols].sort_values(
        ["recall_sensitivity", "pr_auc", "f1", "accuracy"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    st.markdown("### Model comparison")
    for _, row in comparison.iterrows():
        st.caption(
            f"{row['model']}: accuracy={row['accuracy']:.3f}, recall={row['recall_sensitivity']:.3f}, "
            f"specificity={row['specificity']:.3f}, PR-AUC={row['pr_auc']:.3f}, threshold={row['threshold']:.2f}"
        )
    st.dataframe(comparison, use_container_width=True, height=220)

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

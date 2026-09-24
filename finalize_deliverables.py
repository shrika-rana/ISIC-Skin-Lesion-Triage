from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TARGET_RECALL = 0.92
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def folder_has_images(folder: Path) -> bool:
    if not folder.exists() or not folder.is_dir():
        return False
    return any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in folder.rglob("*"))


def find_dataset_root() -> Path:
    for base in [
        PROJECT_DIR / "data",
        PROJECT_DIR,
        Path("./data"),
        Path("."),
        Path("/kaggle/input/isic-skin-lesion-dataset"),
        Path("/kaggle/input"),
    ]:
        if not base.exists():
            continue
        if folder_has_images(base / "Benign") and folder_has_images(base / "Malignant"):
            return base
        for benign_dir in base.rglob("Benign"):
            malignant_dir = benign_dir.parent / "Malignant"
            if folder_has_images(benign_dir) and folder_has_images(malignant_dir):
                return benign_dir.parent
    raise FileNotFoundError("Dataset folders with Benign/Malignant images were not found.")


def collect_images(root: Path) -> pd.DataFrame:
    rows = []
    class_map = {"Benign": 0, "Malignant": 1}
    for class_name, label in class_map.items():
        folder = root / class_name
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({"filepath": str(path), "label": int(label), "class_name": class_name})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No images were found in the dataset.")
    return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def load_image(path, label):
    data = tf.io.read_file(path)
    image = tf.io.decode_image(data, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE, antialias=True)
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    return image, label


def make_dataset(frame: pd.DataFrame, training=False, batch_size=BATCH_SIZE):
    paths = frame["filepath"].astype(str).values
    labels = frame["label"].astype(np.float32).values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=min(len(frame), 5000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def get_labels(dataset):
    labels = []
    for _, batch_labels in dataset:
        labels.extend(batch_labels.numpy().astype(int).tolist())
    return np.asarray(labels)


def predict_probs(model, dataset):
    return model.predict(dataset, verbose=0).reshape(-1)


def choose_threshold_for_recall(y_true, probs, target_recall=0.92):
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for threshold in thresholds:
        pred = (probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "recall": recall_score(y_true, pred, zero_division=0),
                "precision": precision_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
                "accuracy": accuracy_score(y_true, pred),
                "specificity": specificity,
            }
        )
    table = pd.DataFrame(rows)
    eligible = table[table["recall"] >= target_recall].copy()
    if not eligible.empty:
        best = eligible.sort_values(["specificity", "accuracy", "f1", "precision", "threshold"], ascending=[False, False, False, False, False]).iloc[0]
    else:
        best = table.sort_values(["recall", "specificity", "accuracy", "f1", "precision"], ascending=[False, False, False, False, False]).iloc[0]
    return float(best["threshold"]), table


def validation_profile(model, model_name, val_ds, y_val):
    val_probs = predict_probs(model, val_ds)
    threshold, threshold_table = choose_threshold_for_recall(y_val, val_probs, TARGET_RECALL)
    threshold_table.to_csv(ARTIFACT_DIR / f"{model_name}_threshold_search.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(threshold_table["threshold"], threshold_table["recall"], label="Recall / Sensitivity", linewidth=2)
    ax.plot(threshold_table["threshold"], threshold_table["specificity"], label="Specificity", linewidth=2)
    ax.plot(threshold_table["threshold"], threshold_table["accuracy"], label="Accuracy", linewidth=2)
    ax.axvspan(0.35, 0.40, color="lightgreen", alpha=0.30, label="Balanced operating band")
    ax.axvline(threshold, color="black", linestyle="--", label=f"Selected threshold = {threshold:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_title(f"{model_name} validation threshold trade-off")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / f"{model_name}_threshold_curve.png", dpi=200)
    plt.close(fig)

    val_pred = (val_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, val_pred, labels=[0, 1]).ravel()
    return {
        "model": model_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_val, val_pred),
        "precision": precision_score(y_val, val_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_val, val_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_val, val_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_val, val_probs),
        "pr_auc": average_precision_score(y_val, val_probs),
        "fn": int(fn),
        "fp": int(fp),
    }


def evaluate_binary_model(model, model_name, threshold, test_ds, y_test):
    test_probs = predict_probs(model, test_ds)
    test_pred = (test_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()
    return {
        "model": model_name,
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_test, test_pred),
        "precision": precision_score(y_test, test_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_test, test_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_test, test_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, test_probs),
        "pr_auc": average_precision_score(y_test, test_probs),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def measure_latency(model, dataset, batches=20):
    times = []
    count = 0
    for images, _ in dataset:
        if count == 0:
            _ = model(images, training=False).numpy()
        start = time.perf_counter()
        _ = model(images, training=False).numpy()
        elapsed = time.perf_counter() - start
        times.append(elapsed / images.shape[0])
        count += 1
        if count >= batches:
            break
    return {
        "mean_ms_per_image": float(np.mean(times) * 1000),
        "median_ms_per_image": float(np.median(times) * 1000),
    }


def main():
    data_root = find_dataset_root()
    df = collect_images(data_root)
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED)

    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df)
    test_ds = make_dataset(test_df)

    y_val = get_labels(val_ds)
    y_test = get_labels(test_ds)

    candidate_names = ["final_regularized_cnn", "resnet50_head"]
    models = {}
    for name in candidate_names:
        path = MODEL_DIR / f"{name}.keras"
        if path.exists():
            models[name] = tf.keras.models.load_model(path)

    if not models:
        raise RuntimeError("No saved candidate models were found in the models folder.")

    validation_rows = []
    for model_name, model in models.items():
        validation_rows.append(validation_profile(model, model_name, val_ds, y_val))

    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(ARTIFACT_DIR / "validation_model_selection.csv", index=False)

    selected_row = validation_df.sort_values(["recall_sensitivity", "pr_auc", "f1"], ascending=False).iloc[0]
    deploy_model_name = str(selected_row["model"])
    deploy_threshold = float(selected_row["threshold"])
    deploy_model = models[deploy_model_name]
    deploy_model.save(MODEL_DIR / "deployment_model.keras")

    deployment_config = {
        "model_name": deploy_model_name,
        "image_size": list(IMG_SIZE),
        "threshold": deploy_threshold,
        "positive_class": "Malignant",
        "negative_class": "Benign",
        "research_only": True,
        "target_recall_requested": TARGET_RECALL,
        "selection_source": "validation_only",
    }
    with open(ARTIFACT_DIR / "deployment_config.json", "w") as f:
        json.dump(deployment_config, f, indent=2)

    test_rows = []
    for model_name, model in models.items():
        model_threshold = float(validation_df.loc[validation_df["model"] == model_name, "threshold"].iloc[0])
        test_rows.append(evaluate_binary_model(model, model_name, model_threshold, test_ds, y_test))

    results_df = pd.DataFrame(test_rows)
    results_df.to_csv(ARTIFACT_DIR / "final_test_results.csv", index=False)

    latency = measure_latency(deploy_model, test_ds)
    with open(ARTIFACT_DIR / "latency.json", "w") as f:
        json.dump(latency, f, indent=2)

    project_summary = {
        "dataset_root": str(data_root),
        "deployment_model": deploy_model_name,
        "deployment_threshold": deploy_threshold,
        "selection_source": "validation_only",
        "latency": latency,
        "validation_selection": validation_df.to_dict(orient="records"),
        "test_metrics": results_df.to_dict(orient="records"),
    }
    with open(ARTIFACT_DIR / "project_summary.json", "w") as f:
        json.dump(project_summary, f, indent=2, default=float)

    print("Selected deployment model:", deploy_model_name)
    print("Deployment threshold:", deploy_threshold)
    print("Validation selection:\n", validation_df.sort_values(["recall_sensitivity", "pr_auc", "f1"], ascending=False).to_string(index=False))
    print("\nFinal test metrics:\n", results_df.sort_values(["recall_sensitivity", "pr_auc", "f1"], ascending=False).to_string(index=False))
    print("\nLatency:", latency)


if __name__ == "__main__":
    main()

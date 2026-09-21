# ISIC Skin Lesion Triage Assistant
# Binary classification: Benign (0) vs Malignant (1)
# Designed for Kaggle: https://www.kaggle.com/datasets/prabhjitsingh2401/isic-skin-lesion-dataset

import os
import json
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# First use QUICK_MODE=True to verify that the whole notebook/script runs.
# Set QUICK_MODE=False for your final project experiments/report.
QUICK_MODE = False

RUN_ABLATIONS = True
RUN_OPTIMIZER_SWEEP = True
RUN_TRANSFER_LEARNING = True

ABLATION_EPOCHS = 4 if QUICK_MODE else 12
SWEEP_EPOCHS = 3 if QUICK_MODE else 7
BASELINE_FINAL_EPOCHS = 6 if QUICK_MODE else 25
TRANSFER_HEAD_EPOCHS = 5 if QUICK_MODE else 12
TRANSFER_FINETUNE_EPOCHS = 4 if QUICK_MODE else 12

# For faster experimental comparison. Final models still use all training data.
EXPERIMENT_FRACTION = 0.20 if QUICK_MODE else 1.00

# Target recall used to choose triage threshold on validation data.
TARGET_RECALL = 0.95

PROJECT_DIR = Path(".")
MODEL_DIR = PROJECT_DIR / "models"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
MODEL_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))


# ============================================================
# 2. LOCATE THE KAGGLE DATASET AUTOMATICALLY
# ============================================================
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def folder_has_images(folder: Path) -> bool:
    if not folder.exists() or not folder.is_dir():
        return False
    return any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
               for p in folder.rglob("*"))

def find_dataset_root() -> Path:
    candidates = [
        Path("/kaggle/input/isic-skin-lesion-dataset"),
        Path("/kaggle/input"),
        Path("./isic-skin-lesion-dataset"),
        Path("./data"),
        Path("."),
    ]

    for base in candidates:
        if not base.exists():
            continue

        # Direct Benign/Malignant layout
        if folder_has_images(base / "Benign") and folder_has_images(base / "Malignant"):
            return base

        # Search descendants
        try:
            for benign_dir in base.rglob("Benign"):
                malignant_dir = benign_dir.parent / "Malignant"
                if folder_has_images(benign_dir) and folder_has_images(malignant_dir):
                    return benign_dir.parent
        except PermissionError:
            pass

    raise FileNotFoundError(
        "Could not find folders named 'Benign' and 'Malignant'. "
        "On Kaggle, add the dataset as notebook input first."
    )

DATA_ROOT = find_dataset_root()
print("Dataset root:", DATA_ROOT)


# ============================================================
# 3. CREATE DATAFRAME + STRATIFIED TRAIN/VAL/TEST SPLIT
# ============================================================
def collect_images(root: Path) -> pd.DataFrame:
    rows = []
    class_map = {"Benign": 0, "Malignant": 1}

    for class_name, label in class_map.items():
        folder = root / class_name
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append(
                    {
                        "filepath": str(path),
                        "label": int(label),
                        "class_name": class_name,
                    }
                )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise RuntimeError("No image files were found.")
    return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

df = collect_images(DATA_ROOT)
print("\nTotal images:", len(df))
print(df["class_name"].value_counts())

# 70 / 15 / 15 stratified split
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=SEED,
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=SEED,
)

for name, part in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
    print(f"\n{name}: {len(part)} images")
    print(part["class_name"].value_counts())
    print(part["class_name"].value_counts(normalize=True).round(4))

split_summary = pd.DataFrame({
    "split": ["train", "validation", "test"],
    "n_images": [len(train_df), len(val_df), len(test_df)],
    "benign": [
        int((train_df.label == 0).sum()),
        int((val_df.label == 0).sum()),
        int((test_df.label == 0).sum())
    ],
    "malignant": [
        int((train_df.label == 1).sum()),
        int((val_df.label == 1).sum()),
        int((test_df.label == 1).sum())
    ],
})
split_summary.to_csv(ARTIFACT_DIR / "split_summary.csv", index=False)

plt.figure(figsize=(6, 4))
df["class_name"].value_counts().plot(kind="bar")
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of images")
plt.tight_layout()
plt.savefig(ARTIFACT_DIR / "class_distribution.png", dpi=160)
plt.show()


# ============================================================
# 4. TF.DATA PIPELINE
# ============================================================
AUTOTUNE = tf.data.AUTOTUNE

def load_image(path, label):
    data = tf.io.read_file(path)
    image = tf.io.decode_image(data, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE, antialias=True)
    image = tf.cast(image, tf.float32)  # keep 0..255; models preprocess internally
    label = tf.cast(label, tf.float32)
    return image, label

def make_dataset(frame: pd.DataFrame, training=False, batch_size=BATCH_SIZE):
    paths = frame["filepath"].astype(str).values
    labels = frame["label"].astype(np.float32).values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(
            buffer_size=min(len(frame), 5000),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds

train_ds = make_dataset(train_df, training=True)
val_ds = make_dataset(val_df)
test_ds = make_dataset(test_df)

# Smaller/same-data experimental subset for ablation and optimizer search
if EXPERIMENT_FRACTION < 1.0:
    exp_train_df, _ = train_test_split(
        train_df,
        train_size=EXPERIMENT_FRACTION,
        stratify=train_df["label"],
        random_state=SEED,
    )
else:
    exp_train_df = train_df.copy()

exp_train_ds = make_dataset(exp_train_df, training=True)
print("\nExperimental training images:", len(exp_train_df))


# ============================================================
# 5. VISUAL SANITY CHECK
# ============================================================
def show_samples(dataset, n=9):
    images, labels = next(iter(dataset))
    n = min(n, len(images))
    plt.figure(figsize=(9, 9))
    for i in range(n):
        plt.subplot(3, 3, i + 1)
        plt.imshow(tf.cast(images[i], tf.uint8))
        plt.title("Malignant" if int(labels[i].numpy()) == 1 else "Benign")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "sample_images.png", dpi=160)
    plt.show()

show_samples(train_ds)


# ============================================================
# 6. CLASS WEIGHTS
# ============================================================
classes = np.array([0, 1])
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["label"].values,
)
CLASS_WEIGHT = {0: float(weights[0]), 1: float(weights[1])}
print("Class weights:", CLASS_WEIGHT)


# ============================================================
# 7. DATA AUGMENTATION
# ============================================================
def build_augmentation():
    aug = [
        layers.RandomFlip("horizontal"),
        # Keras rotation factor is fraction of a full turn: 10/360 ~= 0.0278
        layers.RandomRotation(0.028, fill_mode="reflect"),
        layers.RandomZoom(
            height_factor=(-0.10, 0.10),
            width_factor=(-0.10, 0.10),
            fill_mode="reflect",
        ),
        layers.RandomContrast(0.10),
    ]

    # Available in modern tf.keras / Keras; guarded for compatibility.
    if hasattr(layers, "RandomBrightness"):
        aug.append(layers.RandomBrightness(0.10, value_range=(0, 255)))

    return keras.Sequential(aug, name="augmentation")


# ============================================================
# 8. METRICS + OPTIMIZERS
# ============================================================
def binary_metrics():
    return [
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="roc_auc", curve="ROC"),
        keras.metrics.AUC(name="pr_auc", curve="PR"),
    ]

def make_optimizer(name="adam", learning_rate=1e-3):
    name = name.lower()
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=learning_rate)
    if name == "sgd":
        return keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9,
            nesterov=True,
        )
    if name == "rmsprop":
        return keras.optimizers.RMSprop(
            learning_rate=learning_rate,
            momentum=0.9,
        )
    raise ValueError(f"Unknown optimizer: {name}")


# ============================================================
# 9. BASELINE CNN BUILDER
# ============================================================
def build_baseline_cnn(
    use_dropout=False,
    use_batchnorm=False,
    use_l2=False,
    use_augmentation=False,
):
    reg = regularizers.l2(1e-4) if use_l2 else None

    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="image")
    x = inputs

    if use_augmentation:
        x = build_augmentation()(x)

    # Scratch CNN uses [0,1] normalization.
    x = layers.Rescaling(1.0 / 255.0)(x)

    for filters in [32, 64, 128, 256]:
        x = layers.Conv2D(
            filters,
            3,
            padding="same",
            use_bias=not use_batchnorm,
            kernel_regularizer=reg,
        )(x)
        if use_batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(
            filters,
            3,
            padding="same",
            use_bias=not use_batchnorm,
            kernel_regularizer=reg,
        )(x)
        if use_batchnorm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.MaxPooling2D()(x)

        if use_dropout:
            x = layers.SpatialDropout2D(0.10 if filters < 128 else 0.20)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)

    if use_dropout:
        x = layers.Dropout(0.40)(x)

    outputs = layers.Dense(1, activation="sigmoid", name="malignant_probability")(x)

    return keras.Model(inputs, outputs, name="baseline_cnn")


def compile_binary_model(model, optimizer_name="adam", learning_rate=1e-3):
    model.compile(
        optimizer=make_optimizer(optimizer_name, learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=binary_metrics(),
    )
    return model


# ============================================================
# 10. CALLBACKS
# ============================================================
def get_callbacks(run_name):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / f"{run_name}.keras"),
            monitor="val_pr_auc",
            mode="max",
            save_best_only=True,
            verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            str(ARTIFACT_DIR / f"{run_name}_history.csv")
        ),
    ]


# ============================================================
# 11. HELPER: TRAIN + SUMMARIZE
# ============================================================
def history_best_row(history, run_name):
    h = pd.DataFrame(history.history)
    if "val_pr_auc" in h.columns:
        idx = h["val_pr_auc"].idxmax()
    else:
        idx = h["val_loss"].idxmin()

    row = h.loc[idx].to_dict()
    row["run_name"] = run_name
    row["best_epoch"] = int(idx + 1)
    return row

def train_model(
    model,
    train_data,
    run_name,
    epochs,
    optimizer_name="adam",
    learning_rate=1e-3,
    use_class_weights=True,
):
    compile_binary_model(model, optimizer_name, learning_rate)
    history = model.fit(
        train_data,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=CLASS_WEIGHT if use_class_weights else None,
        callbacks=get_callbacks(run_name),
        verbose=1,
    )
    return model, history


# ============================================================
# 12. REGULARIZATION ABLATION
# Required variants:
# (a) no regularization
# (b) +Dropout
# (c) +BatchNorm
# (d) +Dropout +BatchNorm +L2 +augmentation
# ============================================================
ablation_results = []
ablation_histories = {}

ABLATION_CONFIGS = {
    "A_no_regularization": dict(
        use_dropout=False,
        use_batchnorm=False,
        use_l2=False,
        use_augmentation=False,
    ),
    "B_dropout": dict(
        use_dropout=True,
        use_batchnorm=False,
        use_l2=False,
        use_augmentation=False,
    ),
    "C_batchnorm": dict(
        use_dropout=False,
        use_batchnorm=True,
        use_l2=False,
        use_augmentation=False,
    ),
    "D_full_regularization": dict(
        use_dropout=True,
        use_batchnorm=True,
        use_l2=True,
        use_augmentation=True,
    ),
}

if RUN_ABLATIONS:
    for run_name, cfg in ABLATION_CONFIGS.items():
        print("\n" + "=" * 80)
        print("ABLATION:", run_name, cfg)

        tf.keras.backend.clear_session()
        model = build_baseline_cnn(**cfg)

        model, history = train_model(
            model=model,
            train_data=exp_train_ds,
            run_name=run_name,
            epochs=ABLATION_EPOCHS,
            optimizer_name="adam",
            learning_rate=1e-3,
        )

        ablation_histories[run_name] = history.history
        ablation_results.append(history_best_row(history, run_name))

    ablation_df = pd.DataFrame(ablation_results)
    ablation_df.to_csv(ARTIFACT_DIR / "ablation_results.csv", index=False)
    print("\nAblation summary:")
    cols = [c for c in [
        "run_name", "best_epoch", "val_accuracy", "val_precision",
        "val_recall", "val_roc_auc", "val_pr_auc", "val_loss"
    ] if c in ablation_df.columns]
    print(ablation_df[cols].sort_values("val_pr_auc", ascending=False))

    # Overlay validation loss
    plt.figure(figsize=(9, 5))
    for run_name, hist in ablation_histories.items():
        plt.plot(hist["val_loss"], label=run_name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Regularization Ablation - Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "ablation_val_loss.png", dpi=160)
    plt.show()

    # Overlay validation PR-AUC
    plt.figure(figsize=(9, 5))
    for run_name, hist in ablation_histories.items():
        if "val_pr_auc" in hist:
            plt.plot(hist["val_pr_auc"], label=run_name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation PR-AUC")
    plt.title("Regularization Ablation - Validation PR-AUC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "ablation_val_pr_auc.png", dpi=160)
    plt.show()


# ============================================================
# 13. OPTIMIZER + LEARNING-RATE SWEEP
# ============================================================
sweep_results = []

if RUN_OPTIMIZER_SWEEP:
    sweep_grid = {
        "adam": [1e-4, 3e-4, 1e-3],
        "sgd": [1e-3, 3e-3, 1e-2],
        "rmsprop": [1e-4, 3e-4, 1e-3],
    }

    full_reg_cfg = ABLATION_CONFIGS["D_full_regularization"]

    for optimizer_name, learning_rates in sweep_grid.items():
        for lr in learning_rates:
            run_name = f"sweep_{optimizer_name}_{lr:g}".replace(".", "p")
            print("\n" + "=" * 80)
            print("SWEEP:", optimizer_name, lr)

            tf.keras.backend.clear_session()
            model = build_baseline_cnn(**full_reg_cfg)
            model, history = train_model(
                model=model,
                train_data=exp_train_ds,
                run_name=run_name,
                epochs=SWEEP_EPOCHS,
                optimizer_name=optimizer_name,
                learning_rate=lr,
            )

            row = history_best_row(history, run_name)
            row["optimizer"] = optimizer_name
            row["learning_rate"] = lr
            sweep_results.append(row)

    sweep_df = pd.DataFrame(sweep_results)
    sweep_df.to_csv(ARTIFACT_DIR / "optimizer_sweep_results.csv", index=False)

    print("\nOptimizer sweep summary:")
    cols = [c for c in [
        "optimizer", "learning_rate", "best_epoch",
        "val_accuracy", "val_recall", "val_roc_auc", "val_pr_auc", "val_loss"
    ] if c in sweep_df.columns]
    print(sweep_df[cols].sort_values("val_pr_auc", ascending=False))


# ============================================================
# 14. CHOOSE BEST SCRATCH-CNN OPTIMIZER CONFIG
# ============================================================
if RUN_OPTIMIZER_SWEEP and len(sweep_results) > 0:
    sweep_df = pd.DataFrame(sweep_results)
    best_sweep = sweep_df.sort_values(
        ["val_pr_auc", "val_recall"],
        ascending=False
    ).iloc[0]
    BEST_OPTIMIZER = str(best_sweep["optimizer"])
    BEST_LR = float(best_sweep["learning_rate"])
else:
    BEST_OPTIMIZER = "adam"
    BEST_LR = 3e-4

print("\nBest scratch-CNN optimizer:", BEST_OPTIMIZER)
print("Best scratch-CNN learning rate:", BEST_LR)


# ============================================================
# 15. TRAIN FINAL FULL-REGULARIZED SCRATCH CNN ON ALL TRAIN DATA
# ============================================================
tf.keras.backend.clear_session()

final_cnn = build_baseline_cnn(
    use_dropout=True,
    use_batchnorm=True,
    use_l2=True,
    use_augmentation=True,
)

final_cnn, final_cnn_history = train_model(
    model=final_cnn,
    train_data=train_ds,
    run_name="final_regularized_cnn",
    epochs=BASELINE_FINAL_EPOCHS,
    optimizer_name=BEST_OPTIMIZER,
    learning_rate=BEST_LR,
)

final_cnn.save(MODEL_DIR / "final_regularized_cnn.keras")


# ============================================================
# 16. RESNET50 TRANSFER LEARNING
# ============================================================
def build_resnet50_transfer():
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="image")

    x = build_augmentation()(inputs)
    x = tf.keras.applications.resnet50.preprocess_input(x)

    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
    )
    base_model.trainable = False

    # training=False keeps BatchNorm inference statistics stable.
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.40)(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
    )(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(
        1,
        activation="sigmoid",
        name="malignant_probability",
    )(x)

    model = keras.Model(inputs, outputs, name="resnet50_transfer")
    return model, base_model


transfer_model = None
transfer_history_combined = {}

if RUN_TRANSFER_LEARNING:
    tf.keras.backend.clear_session()
    transfer_model, base_model = build_resnet50_transfer()

    # Stage 1: train classifier head
    compile_binary_model(transfer_model, "adam", 1e-3)

    head_history = transfer_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=TRANSFER_HEAD_EPOCHS,
        class_weight=CLASS_WEIGHT,
        callbacks=get_callbacks("resnet50_head"),
        verbose=1,
    )

    # Stage 2: fine-tune the last ~40 backbone layers at low LR
    base_model.trainable = True

    for layer in base_model.layers[:-40]:
        layer.trainable = False

    # Keep BatchNorm frozen during fine-tuning to avoid unstable statistics.
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    compile_binary_model(transfer_model, "adam", 1e-5)

    fine_history = transfer_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=TRANSFER_FINETUNE_EPOCHS,
        class_weight=CLASS_WEIGHT,
        callbacks=get_callbacks("resnet50_finetuned"),
        verbose=1,
    )

    # Combine histories for plotting
    all_keys = set(head_history.history) | set(fine_history.history)
    for key in all_keys:
        transfer_history_combined[key] = (
            head_history.history.get(key, [])
            + fine_history.history.get(key, [])
        )

    transfer_model.save(MODEL_DIR / "resnet50_finetuned.keras")


# ============================================================
# 17. PLOTTING TRAINING BEHAVIOUR
# ============================================================
def plot_history(history_dict, title_prefix, filename_prefix):
    if hasattr(history_dict, "history"):
        h = history_dict.history
    else:
        h = history_dict

    if "loss" in h and "val_loss" in h:
        plt.figure(figsize=(8, 5))
        plt.plot(h["loss"], label="Train loss")
        plt.plot(h["val_loss"], label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{title_prefix} - Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(ARTIFACT_DIR / f"{filename_prefix}_loss.png", dpi=160)
        plt.show()

    if "accuracy" in h and "val_accuracy" in h:
        plt.figure(figsize=(8, 5))
        plt.plot(h["accuracy"], label="Train accuracy")
        plt.plot(h["val_accuracy"], label="Validation accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"{title_prefix} - Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(ARTIFACT_DIR / f"{filename_prefix}_accuracy.png", dpi=160)
        plt.show()

    if "recall" in h and "val_recall" in h:
        plt.figure(figsize=(8, 5))
        plt.plot(h["recall"], label="Train recall")
        plt.plot(h["val_recall"], label="Validation recall")
        plt.xlabel("Epoch")
        plt.ylabel("Recall")
        plt.title(f"{title_prefix} - Recall")
        plt.legend()
        plt.tight_layout()
        plt.savefig(ARTIFACT_DIR / f"{filename_prefix}_recall.png", dpi=160)
        plt.show()

plot_history(final_cnn_history, "Final Regularized CNN", "final_cnn")

if RUN_TRANSFER_LEARNING and transfer_history_combined:
    plot_history(
        transfer_history_combined,
        "ResNet50 Transfer Learning",
        "resnet50",
    )


# ============================================================
# 18. PREDICTION + THRESHOLD TUNING
# ============================================================
def get_labels(dataset):
    y = []
    for _, labels_batch in dataset:
        y.extend(labels_batch.numpy().astype(int).tolist())
    return np.asarray(y)

def predict_probs(model, dataset):
    return model.predict(dataset, verbose=1).reshape(-1)

y_val = get_labels(val_ds)
y_test = get_labels(test_ds)

def choose_threshold_for_recall(y_true, probs, target_recall=0.95):
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []

    for t in thresholds:
        pred = (probs >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "recall": recall_score(y_true, pred, zero_division=0),
            "precision": precision_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "accuracy": accuracy_score(y_true, pred),
        })

    table = pd.DataFrame(rows)

    eligible = table[table["recall"] >= target_recall].copy()

    if len(eligible) > 0:
        # Among thresholds meeting the desired sensitivity, prefer better F1,
        # then precision. This limits false alarms without sacrificing target recall.
        best = eligible.sort_values(
            ["f1", "precision", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
    else:
        # If the requested recall is not achievable, choose highest recall,
        # then best F1.
        best = table.sort_values(
            ["recall", "f1", "precision"],
            ascending=False,
        ).iloc[0]

    return float(best["threshold"]), table


# ============================================================
# 19. VALIDATION-BASED MODEL/THRESHOLD SELECTION
# The test set is NOT used to choose the model or threshold.
# ============================================================
def validation_profile(model, model_name):
    val_probs = predict_probs(model, val_ds)
    threshold, threshold_table = choose_threshold_for_recall(
        y_val,
        val_probs,
        TARGET_RECALL,
    )
    threshold_table.to_csv(
        ARTIFACT_DIR / f"{model_name}_threshold_search.csv",
        index=False,
    )

    val_pred = (val_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_val, val_pred, labels=[0, 1]).ravel()

    row = {
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
    return row

candidate_models = {
    "final_regularized_cnn": final_cnn,
}
if RUN_TRANSFER_LEARNING and transfer_model is not None:
    candidate_models["resnet50_finetuned"] = transfer_model

validation_rows = [
    validation_profile(model, name)
    for name, model in candidate_models.items()
]
validation_df = pd.DataFrame(validation_rows)
validation_df.to_csv(
    ARTIFACT_DIR / "validation_model_selection.csv",
    index=False,
)

print("\nVALIDATION MODEL SELECTION")
print(validation_df.sort_values(
    ["recall_sensitivity", "pr_auc", "f1"],
    ascending=False,
))

# Priority: sensitivity/recall, then PR-AUC, then F1.
# This rule is fixed before looking at final test results.
selected_validation_row = validation_df.sort_values(
    ["recall_sensitivity", "pr_auc", "f1"],
    ascending=False,
).iloc[0]

DEPLOY_MODEL_NAME = str(selected_validation_row["model"])
DEPLOY_THRESHOLD = float(selected_validation_row["threshold"])
deploy_model = candidate_models[DEPLOY_MODEL_NAME]


# ============================================================
# 20. FINAL TEST EVALUATION
# The threshold for each model was already fixed using validation data.
# ============================================================
def evaluate_binary_model(model, model_name, threshold):
    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION:", model_name)

    test_probs = predict_probs(model, test_ds)
    test_pred = (test_probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()

    sensitivity = recall_score(y_test, test_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    metrics = {
        "model": model_name,
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_test, test_pred),
        "precision": precision_score(y_test, test_pred, zero_division=0),
        "recall_sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1_score(y_test, test_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, test_probs),
        "pr_auc": average_precision_score(y_test, test_probs),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    print("\nFixed validation-derived threshold:", round(float(threshold), 3))
    print(pd.Series(metrics))
    print("\nClassification report:")
    print(classification_report(
        y_test,
        test_pred,
        target_names=["Benign", "Malignant"],
        digits=4,
        zero_division=0,
    ))

    cm = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xticks([0, 1], ["Benign", "Malignant"])
    plt.yticks([0, 1], ["Benign", "Malignant"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(
        ARTIFACT_DIR / f"{model_name}_confusion_matrix.png",
        dpi=160,
    )
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, test_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={metrics['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Sensitivity")
    plt.title(f"{model_name} - ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / f"{model_name}_roc.png", dpi=160)
    plt.show()

    precision, recall, _ = precision_recall_curve(y_test, test_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP={metrics['pr_auc']:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} - Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / f"{model_name}_pr_curve.png", dpi=160)
    plt.show()

    pred_df = test_df.copy().reset_index(drop=True)
    pred_df["malignant_probability"] = test_probs
    pred_df["prediction"] = test_pred
    pred_df["prediction_name"] = np.where(
        pred_df["prediction"] == 1,
        "Malignant",
        "Benign",
    )
    pred_df.to_csv(
        ARTIFACT_DIR / f"{model_name}_test_predictions.csv",
        index=False,
    )

    return metrics

threshold_lookup = dict(
    zip(validation_df["model"], validation_df["threshold"])
)

all_test_results = []
for model_name, model in candidate_models.items():
    all_test_results.append(
        evaluate_binary_model(
            model,
            model_name,
            float(threshold_lookup[model_name]),
        )
    )

results_df = pd.DataFrame(all_test_results)
results_df.to_csv(ARTIFACT_DIR / "final_test_results.csv", index=False)

print("\nFINAL TEST COMPARISON (reporting only; not used for model selection)")
print(results_df.sort_values(
    ["recall_sensitivity", "pr_auc", "f1"],
    ascending=False,
))

# Save the deployment candidate selected from validation only.
deploy_model.save(MODEL_DIR / "deployment_model.keras")

deployment_config = {
    "model_name": DEPLOY_MODEL_NAME,
    "image_size": list(IMG_SIZE),
    "threshold": DEPLOY_THRESHOLD,
    "positive_class": "Malignant",
    "negative_class": "Benign",
    "research_only": True,
    "target_recall_requested": TARGET_RECALL,
    "selection_source": "validation_only",
}
with open(ARTIFACT_DIR / "deployment_config.json", "w") as f:
    json.dump(deployment_config, f, indent=2)

print("\nDeployment model selected on validation data:", DEPLOY_MODEL_NAME)
print("Deployment threshold:", DEPLOY_THRESHOLD)


# ============================================================
# 21. INFERENCE LATENCY
# ============================================================
def measure_latency(model, dataset, batches=20):
    times = []
    count = 0

    for images, _ in dataset:
        # Warm-up
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

latency = measure_latency(deploy_model, test_ds)
with open(ARTIFACT_DIR / "latency.json", "w") as f:
    json.dump(latency, f, indent=2)

print("Latency:", latency)


# ============================================================
# 22. ERROR ANALYSIS
# ============================================================
pred_path = ARTIFACT_DIR / f"{DEPLOY_MODEL_NAME}_test_predictions.csv"
pred_analysis = pd.read_csv(pred_path)

false_negatives = pred_analysis[
    (pred_analysis["label"] == 1) &
    (pred_analysis["prediction"] == 0)
].copy()

false_positives = pred_analysis[
    (pred_analysis["label"] == 0) &
    (pred_analysis["prediction"] == 1)
].copy()

false_negatives.to_csv(ARTIFACT_DIR / "false_negatives.csv", index=False)
false_positives.to_csv(ARTIFACT_DIR / "false_positives.csv", index=False)

print("False negatives:", len(false_negatives))
print("False positives:", len(false_positives))


# ============================================================
# 23. REPORT-READY SUMMARY
# ============================================================
selected_test_row = results_df[
    results_df["model"] == DEPLOY_MODEL_NAME
].iloc[0]

summary = {
    "dataset_root": str(DATA_ROOT),
    "n_total": int(len(df)),
    "n_train": int(len(train_df)),
    "n_validation": int(len(val_df)),
    "n_test": int(len(test_df)),
    "class_weights": CLASS_WEIGHT,
    "best_scratch_optimizer": BEST_OPTIMIZER,
    "best_scratch_learning_rate": BEST_LR,
    "deployment_model": DEPLOY_MODEL_NAME,
    "deployment_threshold": DEPLOY_THRESHOLD,
    "selection_source": "validation_only",
    "latency": latency,
    "test_metrics": selected_test_row.to_dict(),
}

with open(ARTIFACT_DIR / "project_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)

print("\nSaved project outputs to:")
print("Models:", MODEL_DIR.resolve())
print("Artifacts:", ARTIFACT_DIR.resolve())

from pathlib import Path

import numpy as np
import pandas as pd
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
)

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / 'models'
ARTIFACT_DIR = ROOT / 'artifacts'
MODEL_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def folder_has_images(folder: Path) -> bool:
    return folder.exists() and folder.is_dir() and any(
        p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in folder.rglob('*')
    )


def find_dataset_root() -> Path:
    for base in [ROOT / 'data', ROOT, Path('./data'), Path('.')]:
        if not base.exists():
            continue
        if folder_has_images(base / 'Benign') and folder_has_images(base / 'Malignant'):
            return base
        for benign_dir in base.rglob('Benign'):
            malignant_dir = benign_dir.parent / 'Malignant'
            if folder_has_images(benign_dir) and folder_has_images(malignant_dir):
                return benign_dir.parent
    raise FileNotFoundError('Dataset folders not found')


def collect_images(root: Path) -> pd.DataFrame:
    rows = []
    class_map = {'Benign': 0, 'Malignant': 1}
    for class_name, label in class_map.items():
        for path in (root / class_name).rglob('*'):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({'filepath': str(path), 'label': int(label), 'class_name': class_name})
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def load_image(path, label):
    data = tf.io.read_file(path)
    image = tf.io.decode_image(data, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE, antialias=True)
    image = tf.cast(image, tf.float32)
    return image, tf.cast(label, tf.float32)


def make_dataset(frame: pd.DataFrame, training: bool = False):
    paths = frame['filepath'].astype(str).values
    labels = frame['label'].astype(np.float32).values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=min(len(frame), 5000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_augmentation():
    aug = [
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.04, fill_mode='reflect'),
        layers.RandomZoom(height_factor=(-0.08, 0.08), width_factor=(-0.08, 0.08), fill_mode='reflect'),
        layers.RandomContrast(0.08),
    ]
    if hasattr(layers, 'RandomBrightness'):
        aug.append(layers.RandomBrightness(0.08, value_range=(0, 255)))
    return keras.Sequential(aug, name='augmentation')


def build_tuned_cnn():
    reg = regularizers.l2(1e-4)
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name='image')
    x = build_augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255.0)(x)

    for filters in [32, 64, 128, 256]:
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False, kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False, kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.SpatialDropout2D(0.10 if filters < 128 else 0.20)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(1, activation='sigmoid', name='malignant_probability')(x)
    return keras.Model(inputs, outputs, name='tuned_regularized_cnn')


def predict_probs(model, ds):
    return model.predict(ds, verbose=0).reshape(-1)


def get_labels(ds):
    labels = []
    for _, batch_labels in ds:
        labels.extend(batch_labels.numpy().astype(int).tolist())
    return np.asarray(labels)


def select_threshold(y_true, probs):
    rows = []
    for threshold in np.linspace(0.05, 0.95, 91):
        pred = (probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append(
            {
                'threshold': float(threshold),
                'recall': recall_score(y_true, pred, zero_division=0),
                'precision': precision_score(y_true, pred, zero_division=0),
                'f1': f1_score(y_true, pred, zero_division=0),
                'accuracy': accuracy_score(y_true, pred),
                'specificity': specificity,
            }
        )
    table = pd.DataFrame(rows)
    target = table[(table['recall'] >= 0.92) & (table['specificity'] >= 0.80)]
    if not target.empty:
        best = target.sort_values(
            ['specificity', 'accuracy', 'f1', 'precision', 'threshold'],
            ascending=[False, False, False, False, False],
        ).iloc[0]
        return float(best['threshold']), table
    best = table.sort_values(
        ['recall', 'specificity', 'accuracy', 'f1', 'precision'],
        ascending=[False, False, False, False, False],
    ).iloc[0]
    return float(best['threshold']), table


def evaluate_model(model, threshold, ds, y_true):
    probs = predict_probs(model, ds)
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        'model': 'final_regularized_cnn_tuned',
        'threshold': float(threshold),
        'accuracy': accuracy_score(y_true, pred),
        'precision': precision_score(y_true, pred, zero_division=0),
        'recall_sensitivity': recall_score(y_true, pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) else 0.0,
        'f1': f1_score(y_true, pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, probs),
        'pr_auc': average_precision_score(y_true, probs),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }


def main():
    print('Finding dataset...')
    data_root = find_dataset_root()
    df = collect_images(data_root)
    print('Images:', len(df), df['class_name'].value_counts().to_dict())

    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['label'], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['label'], random_state=SEED)

    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(val_df)
    test_ds = make_dataset(test_df)

    y_val = get_labels(val_ds)
    y_test = get_labels(test_ds)

    weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=train_df['label'].values)
    class_weight = {0: float(weights[0]), 1: float(weights[1])}
    print('Class weights:', class_weight)

    model = build_tuned_cnn()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='roc_auc', curve='ROC'),
            keras.metrics.AUC(name='pr_auc', curve='PR'),
        ],
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / 'final_regularized_cnn.keras'),
            monitor='val_pr_auc',
            mode='max',
            save_best_only=True,
            verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_pr_auc',
            mode='max',
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.4,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(ARTIFACT_DIR / 'final_regularized_cnn_tuned_history.csv')),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    selected_threshold, threshold_df = select_threshold(y_val, predict_probs(model, val_ds))
    threshold_df.to_csv(ARTIFACT_DIR / 'final_regularized_cnn_tuned_thresholds.csv', index=False)
    print('Selected tuned threshold:', selected_threshold)
    print(threshold_df[(threshold_df['threshold'] >= 0.2) & (threshold_df['threshold'] <= 0.5)].sort_values('threshold').head(10).to_string(index=False))

    summary = evaluate_model(model, selected_threshold, test_ds, y_test)
    pd.DataFrame([summary]).to_csv(ARTIFACT_DIR / 'final_regularized_cnn_tuned_test.csv', index=False)
    print('\nTuned CNN test summary:')
    print(pd.DataFrame([summary]).to_string(index=False))

    print('\nTraining history best val PR-AUC:', max(history.history.get('val_pr_auc', [0])))


if __name__ == '__main__':
    main()

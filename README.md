# Intelligent Visual Recognition Suite — ISIC Skin Lesion Triage

This project adapts the supplied CNN/transfer-learning assignment to the Kaggle dataset:

**ISIC skin lesion dataset — Prabhjit Singh**
https://www.kaggle.com/datasets/prabhjitsingh2401/isic-skin-lesion-dataset

The dataset is organized into two image classes:

- `Benign` → label 0
- `Malignant` → label 1

## What the project implements

- Automatic Kaggle dataset discovery
- Stratified 70/15/15 image split
- Image resizing to 224×224 RGB
- Class-weight handling for imbalance
- Anatomically safe-ish augmentation for dermoscopic images:
  horizontal flip, small rotation, zoom, contrast, brightness
- Baseline CNN from scratch
- Four regularization ablations:
  1. none
  2. Dropout
  3. Batch Normalization
  4. Dropout + BatchNorm + L2 + augmentation
- Optimizer sweep:
  - Adam
  - SGD + momentum
  - RMSprop
  - 3 learning rates per optimizer
- ResNet50 ImageNet transfer learning
- Frozen-head training followed by low-learning-rate fine tuning
- Metrics:
  - accuracy
  - precision
  - malignant recall/sensitivity
  - specificity
  - F1
  - ROC-AUC
  - PR-AUC
  - confusion matrix
- Validation-only threshold tuning to prioritize malignant recall
- Training/validation curves
- False-negative / false-positive exports
- Inference latency measurement
- Saved `.keras` models
- Streamlit upload-and-predict dashboard

## Kaggle run instructions

1. Create a Kaggle notebook.
2. Add the dataset as notebook input.
3. Upload `train_isic.py` or paste the notebook code.
4. Enable a GPU accelerator.
5. For a quick end-to-end smoke test, change:
   `QUICK_MODE = True`
6. For final experiments/report, use:
   `QUICK_MODE = False`
7. Run:
   `python train_isic.py`

Generated outputs are written to:

- `models/`
- `artifacts/`

Important files include:

- `models/deployment_model.keras`
- `artifacts/final_test_results.csv`
- `artifacts/ablation_results.csv`
- `artifacts/optimizer_sweep_results.csv`
- `artifacts/deployment_config.json`
- confusion-matrix, ROC, PR and training-curve PNG files

## Streamlit demo

After training, copy the entire project folder (including generated `models/`
and `artifacts/`) to a local machine or deployment environment.

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## Suggested report narrative

### Problem
Binary dermoscopic image triage: classify lesions as benign or malignant and
prioritize sensitivity/recall for malignant cases.

### Architecture rationale
The scratch CNN establishes a first-principles benchmark and makes the effect
of regularization visible. ResNet50 provides a strong transfer-learning
baseline because ImageNet-pretrained convolutional filters can be adapted to
medical texture/shape patterns with less data and faster convergence than a
large network trained entirely from scratch.

### Regularization story
Compare validation loss/PR-AUC across the four ablation variants. Discuss the
train–validation gap, whether Dropout reduces memorization, whether BatchNorm
stabilizes optimization, and whether L2 + augmentation improves generalization.

### Optimizer story
Compare Adam, SGD and RMSprop at multiple learning rates while keeping model,
data split and regularization fixed.

### Clinical-style metric story
Do not rely only on accuracy. Emphasize malignant recall/sensitivity, number
of false negatives, PR-AUC and the confusion matrix. The probability threshold
is chosen on validation data rather than fixed blindly at 0.5.

### Deployment recommendation
Treat this as a research decision-support system only. A high-sensitivity
threshold may flag more cases for human review, trading additional false
positives for fewer missed malignant cases. Before any real-world clinical use,
perform external validation, calibration, subgroup/fairness analysis,
prospective evaluation, monitoring for data drift, and regulatory/clinical
review.

## Important limitation

The Kaggle copy does not expose patient identifiers in the simple two-folder
layout. Therefore this project performs an image-level split. For a clinical
study, use patient-level splitting whenever patient IDs are available, to
reduce leakage from multiple images of the same patient/lesion.

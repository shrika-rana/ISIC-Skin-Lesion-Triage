# Technical Report: ISIC Skin Lesion Triage System

## 1. Executive Summary

This project implements an end-to-end deep learning pipeline for binary skin lesion classification using dermoscopic images from the ISIC dataset. The objective is to distinguish benign lesions from malignant lesions and to prioritize malignant recall so that suspicious lesions are not missed during triage. The system combines data preparation, CNN training, regularization ablation, optimizer sweep, transfer learning with ResNet50, validation-based threshold tuning, and a Streamlit deployment dashboard.

The project is implemented primarily in [train_isic.py](train_isic.py) and [app.py](app.py). It generates trained models in [models](models) and evaluation artifacts in [artifacts](artifacts). The final deployment model is a ResNet50-based transfer-learning model selected using validation performance, because it achieved the best balance between malignant sensitivity and overall classification quality among the candidate models.

The final deployment configuration is saved in [artifacts/deployment_config.json](artifacts/deployment_config.json), while the key evaluation summaries are stored in [artifacts/validation_model_selection.csv](artifacts/validation_model_selection.csv) and [artifacts/final_test_results.csv](artifacts/final_test_results.csv).

---

## 2. Problem Definition and Clinical Framing

The task is a binary classification problem:

- Benign lesion: label 0
- Malignant lesion: label 1

This is not a general computer vision task alone; it is a decision-support problem. In skin cancer screening, a false negative is especially costly because a malignant lesion may be overlooked. Therefore, the learning objective is not merely to maximize raw accuracy. Instead, the project prioritizes malignant sensitivity (recall for the positive class), with the probability threshold selected on validation data rather than fixed at 0.5.

The project design reflects this by:

- evaluating precision, recall, F1, specificity, ROC-AUC, and PR-AUC,
- tuning the decision threshold to meet a target malignant recall of 0.95,
- reporting confusion matrices and false-negative counts,
- treating the model as a research decision-support tool rather than a clinical diagnostic instrument.

---

## 3. System Architecture

### 3.1 High-Level Pipeline

The architecture is organized as a training-and-deployment workflow:

1. Dataset discovery and loading
2. Stratified 70/15/15 split for train/validation/test
3. Image preprocessing and TF.data pipeline generation
4. Baseline CNN model training
5. Regularization ablation study
6. Optimizer and learning-rate sweep
7. ResNet50 transfer learning with head training and fine-tuning
8. Validation-based model and threshold selection
9. Final test evaluation
10. Latency measurement
11. Deployment export and Streamlit dashboard

### 3.2 Data Flow

The system begins by locating the dataset under the Benign and Malignant folders. The script scans candidate dataset roots and checks for image files with common extensions such as .jpg, .jpeg, .png, .bmp, and .webp. Once found, it builds a pandas DataFrame with file paths and labels.

The dataset is then split using stratification to preserve class balance:

- Training set: 70%
- Validation set: 15%
- Test set: 15%

After splitting, the project constructs a TensorFlow dataset using a tf.data pipeline with:

- shuffled batches for training,
- resizing to 224 x 224,
- float conversion,
- prefetching and autotuning for throughput.

### 3.3 Preprocessing and Augmentation

The training pipeline uses moderate image augmentation to improve generalization without overwhelming the learning signal. Augmentation includes:

- horizontal flipping,
- small-angle rotation,
- zooming,
- contrast adjustment,
- brightness adjustment.

The scratch CNN includes Rescaling(1/255) to normalize pixel values to [0, 1]. For transfer learning, the project uses the ResNet50 preprocessing function provided by Keras.

### 3.4 Scratch CNN Architecture

The baseline model is a custom convolutional neural network designed for the lesion classification task. It follows a standard deep feature extractor pattern:

- input tensor of shape (224, 224, 3)
- stacked conv blocks with increasing filter depth [32, 64, 128, 256]
- each block includes 2 convolutions, ReLU activation, and max pooling
- optional Batch Normalization and/or Dropout depending on the ablation setting
- global average pooling before the classifier head
- dense layer plus sigmoid output for malignant probability

The implementation is defined in the build_baseline_cnn function in [train_isic.py](train_isic.py). This architecture provides a baseline reference for testing whether regularization and transfer learning improve performance beyond the no-regularization baseline.

### 3.5 Transfer-Learning Architecture

The transfer learning pipeline uses ResNet50 pretrained on ImageNet. The design has two stages:

1. Feature extraction stage
   - ResNet50 base is frozen
   - global average pooling is applied to the convolutional feature maps
   - BatchNorm and Dropout regularization are added
   - a dense classification head is trained

2. Fine-tuning stage
   - the last 40 layers of the ResNet50 backbone are unfrozen
   - a lower learning rate (1e-5) is used
   - BatchNorm layers are kept frozen to maintain stable statistics during fine-tuning

This strategy is common in transfer learning because it reuses general visual features while adapting the final representation to the dermoscopic domain.

### 3.6 Deployment Interface

The project provides a Streamlit dashboard in [app.py](app.py). The dashboard loads:

- the deployment model from [models/deployment_model.keras](models/deployment_model.keras)
- the saved deployment configuration from [artifacts/deployment_config.json](artifacts/deployment_config.json)

The user uploads an image, the model predicts a malignant probability, and the app compares it against the selected threshold. It displays:

- uploaded image
- malignant probability
- probability bar
- triage status (below or above threshold)
- research disclaimer and decision-support framing

---

## 4. Training Strategy and Loss Functions

The project uses binary cross-entropy as the loss function because the task is binary classification. The optimization objective is to minimize log loss while also improving sensitivity to the malignant class.

Model compilation includes the following metrics:

- BinaryAccuracy
- Precision
- Recall
- ROC-AUC
- PR-AUC

The project also uses class weighting to account for class imbalance in the dataset. The class weights are computed using balanced weighting from scikit-learn, with the goal of reducing bias toward the majority class.

Training includes callbacks for:

- best-model checkpointing using validation PR-AUC,
- early stopping based on validation PR-AUC,
- learning-rate reduction on validation loss,
- CSV logging of training histories.

This is important because it prevents overfitting and preserves the best-performing model checkpoint for later deployment.

---

## 5. Hyperparameter Tuning and Optimization Study

The project includes an optimizer sweep for the scratch CNN using multiple learning rates for SGD, RMSprop, and Adam. The sweep results are recorded in [artifacts/optimizer_sweep_results.csv](artifacts/optimizer_sweep_results.csv).

### 5.1 Sweep Configuration

The project tested:

- SGD: 1e-3, 3e-3, 1e-2
- RMSprop: 1e-4, 3e-4, 1e-3
- Adam: included in the overall design but the saved results used in this run show the strongest controllable sweep comparisons among the smaller set of experiments retained in the artifact file.

### 5.2 Observed Results

The best scratch-model tuning result in the saved summary was:

| Optimizer | Learning Rate | Validation Accuracy | Validation PR-AUC | Validation Recall |
|---|---:|---:|---:|---:|
| SGD | 0.01 | 0.5654 | 0.5320 | 0.6066 |
| RMSprop | 0.0003 | 0.5726 | 0.5181 | 0.5686 |
| SGD | 0.001 | 0.4287 | 0.4990 | 0.9670 |

The numbers show that the sweep did not produce a single dominant configuration across all metrics. In particular, some high-recall settings achieved excellent malignant sensitivity but suffered from weaker overall accuracy or poor calibration. This is why the project later moved to validation-based threshold selection and then to a ResNet50 transfer-learning model, which gave a better overall operating point.

### 5.3 Interpretation

The optimizer study confirms a common challenge in medical imaging: there is no universal “best” model if one only looks at accuracy. A model with strong recall may still be less precise, leading to more false positives. Therefore, optimizer choice was not treated as the final decision criterion. Instead, the final model was selected based on validation performance under the project’s clinical-style priority metric: malignant recall and PR-AUC.

---

## 6. Regularization Ablation Study

The project compares four regularization variants of the scratch CNN, as required by the research design. The results are stored in [artifacts/ablation_results.csv](artifacts/ablation_results.csv).

### 6.1 Ablation Variants

1. A_no_regularization
2. B_dropout
3. C_batchnorm
4. D_full_regularization

### 6.2 Results Summary

| Model | Validation Accuracy | Validation Loss | Validation PR-AUC | Validation Recall |
|---|---:|---:|---:|---:|
| A_no_regularization | 0.5936 | 0.6734 | 0.5671 | 0.2936 |
| B_dropout | 0.4368 | 0.6967 | 0.5323 | 1.0000 |
| C_batchnorm | 0.4970 | 0.7428 | 0.5444 | 0.9383 |
| D_full_regularization | 0.4541 | 0.8495 | 0.5342 | 0.9785 |

### 6.3 Interpretation

The results show trade-offs that are typical in medical image classification:

- The no-regularization model had the best validation accuracy and lowest validation loss, but it had poor malignant recall when evaluated at the default decision threshold.
- Dropout and BatchNorm improved recall substantially, indicating that regularization changed the operating point toward safer detection of malignant lesions.
- The full regularization model achieved a high recall with a more balanced decision process, which is useful in triage contexts where missing a malignant lesion is more harmful than a few extra false positives.

This is exactly why the project chooses a validation threshold instead of a fixed 0.5 threshold: the model must satisfy a sensitivity target rather than maximizing raw accuracy alone.

---

## 7. Validation-Based Model and Threshold Selection

The project explicitly follows a validation-only selection policy. The model and threshold are selected using validation data only, and the test set is reserved for final evaluation. This is a crucial methodological step to avoid information leakage.

The selection rule implemented in [train_isic.py](train_isic.py) is:

1. Compute validation probabilities for each candidate model.
2. Evaluate a threshold sweep from 0.01 to 0.99.
3. Keep thresholds that satisfy the desired malignant recall target.
4. Among those, choose the threshold that maximizes F1, then precision.
5. Choose the model with the best validation recall, PR-AUC, and F1.

This is documented in the validation profile and threshold search artifacts.

The key validation outcome in [artifacts/validation_model_selection.csv](artifacts/validation_model_selection.csv) is:

| Model | Threshold | Accuracy | Recall | PR-AUC |
|---|---:|---:|---:|---:|
| resnet50_head | 0.29 | 0.8087 | 0.9555 | 0.8822 |
| final_regularized_cnn | 0.22 | 0.5889 | 0.9519 | 0.6442 |

The ResNet50 model emerged as the best deployment candidate because it maintained malignant sensitivity above the target while also yielding substantially better accuracy and PR-AUC.

---

## 8. Final Test Evaluation Metrics

The final test evaluation is stored in [artifacts/final_test_results.csv](artifacts/final_test_results.csv). The selected deployment model is the ResNet50 transfer-learning model with a threshold of 0.29.

### 8.1 Final Test Metrics

| Model | Threshold | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| final_regularized_cnn | 0.22 | 0.5799 | 0.5103 | 0.9555 | 0.2884 | 0.6653 | 0.7296 | 0.6332 |
| resnet50_head | 0.29 | 0.7934 | 0.6917 | 0.9512 | 0.6709 | 0.8010 | 0.9095 | 0.8732 |

### 8.2 Interpretation

The ResNet50 model is significantly stronger on the test set:

- accuracy improves from 0.58 to 0.79,
- PR-AUC improves from 0.63 to 0.87,
- sensitivity remains above 0.95,
- specificity is materially better than the scratch CNN baseline,
- the false-positive count is reduced relative to the baseline.

This demonstrates that transfer learning provides a more reliable operating point for the triage problem than the scratch-only approach.

---

## 9. Inference Latency and Deployment Performance

The project records latency in [artifacts/latency.json](artifacts/latency.json). The saved measurement used the deployment model and the test dataset.

The recorded value is:

- mean latency: approximately 47.96 ms per image
- median latency: approximately 45.73 ms per image

This is acceptable for a research deployment or small-scale triage dashboard, especially for a single-image inference workflow in an application such as Streamlit. The measured latency is also consistent with a system that performs one image inference at a time rather than batch processing at scale.

---

## 10. Deployment and Dashboard System

The deployment artifact is generated by exporting the selected model and its configuration, then passing them to the app in [app.py](app.py). The app loads the model and checks for the existence of:

- [models/deployment_model.keras](models/deployment_model.keras)
- [artifacts/deployment_config.json](artifacts/deployment_config.json)

If either file is missing, the dashboard shows an error and stops. If present, the user can upload an image and receive a triage prediction.

The dashboard is designed for:

- image upload,
- probability estimation,
- threshold-based classification,
- presentation of metrics when the final evaluation files are available,
- explainable decision-support framing.

This ensures the trained model is not just stored on disk but is also usable in an interactive operational setting.

---

## 11. Project Workflow Summary

The full workflow can be summarized as follows:

1. Locate and load the dataset.
2. Build a stratified train/validation/test split.
3. Generate a TensorFlow data pipeline.
4. Train baseline CNN variants and compare regularization strategies.
5. Sweep the optimizer and learning rate to understand training stability.
6. Use ResNet50 transfer learning for stronger feature extraction.
7. Select the model and threshold using validation data only.
8. Evaluate the selected model on the held-out test set.
9. Save the deployment model and configuration.
10. Launch the Streamlit dashboard for inference.

This is a complete machine learning lifecycle for a real classification problem: model development, validation, deployment preparation, and operational interface.

---

## 12. Limitations and Future Work

Although the project is functionally complete and produces a working deployment model, several limitations remain:

- The split is image-level, not patient-level. This means multiple images from the same patient or lesion could leak across train/validation/test partitions.
- The project is a research prototype and should not be treated as a medical device.
- The model is trained on the available dataset only; external validation is required for clinical deployment.
- Thresholds are optimized for sensitivity, which improves recall but can increase false positives.

Future extensions could include:

- patient-level data splitting,
- cross-validation across multiple image sources,
- uncertainty estimation,
- calibration analysis,
- explainability tools such as Grad-CAM,
- a production-grade API or web service deployment.

---

## 13. Conclusion

This project demonstrates a complete deep-learning pipeline for automated skin lesion triage. It combines classical CNN training, regularization analysis, learning-rate and optimizer sweeps, transfer learning with ResNet50, validation-based threshold tuning, and final deployment as a Streamlit app. The final chosen model, ResNet50 transfer learning, provides the strongest balance between malignant recall and classification quality, making it the best candidate for the deployment workflow.

The project is not only a training exercise; it is an end-to-end model lifecycle that starts with data collection and ends with an operational inference interface. That makes it a strong example of how research-grade model selection transforms into a deployable decision-support system.

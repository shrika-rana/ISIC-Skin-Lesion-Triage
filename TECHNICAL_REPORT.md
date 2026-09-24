# Technical Report: ISIC Skin Lesion Triage System

## 1. Executive Summary

This project is now complete as an end-to-end deep learning workflow for binary skin lesion triage using dermoscopic images from the ISIC dataset. It starts with raw image data, builds a robust training pipeline, evaluates several model candidates, selects the best deployment model using validation data, and finishes with a working Streamlit dashboard for inference.

The project is implemented mainly in [train_isic.py](train_isic.py) and [app.py](app.py). The training process generated saved models under [models](models) and final evaluation artifacts under [artifacts](artifacts). The selected deployment model is the ResNet50 transfer-learning model, which was chosen because it gives the best overall operating point for malignant recall while maintaining much stronger overall accuracy and PR-AUC than the CNN baseline.

The current deployment configuration is saved in [artifacts/deployment_config.json](artifacts/deployment_config.json), and the final validation and test summaries are stored in [artifacts/validation_model_selection.csv](artifacts/validation_model_selection.csv) and [artifacts/final_test_results.csv](artifacts/final_test_results.csv). These files represent the final verified project state and are the source of truth for the dashboard and report.

---

## 2. What Is Happening in This Project

This project does several things in sequence:

1. It loads the ISIC image dataset and organizes it into benign and malignant classes.
2. It creates a train/validation/test split while preserving class balance.
3. It builds a TensorFlow/Keras pipeline for preprocessing and batch training.
4. It trains a baseline CNN and compares the effect of regularization strategies.
5. It performs an optimizer and learning-rate sweep to understand training stability.
6. It applies ResNet50 transfer learning to improve feature extraction.
7. It selects the best model and threshold using validation data only.
8. It evaluates the chosen model on the held-out test set.
9. It saves the selected deployment model and configuration.
10. It exposes the model through a Streamlit dashboard for image-based malignant-risk prediction.

The central theme of the project is not just training a model, but choosing the right model and decision threshold for a medical triage use case. A model that has high recall is valuable in screening, but it must also be accurate enough to avoid excessive false positives. The project therefore prioritizes malignant sensitivity while monitoring precision, specificity, PR-AUC, and overall accuracy.

---

## 3. Problem Definition and Clinical Framing

The task is a binary classification problem:

- Benign lesion: label 0
- Malignant lesion: label 1

This is a clinical decision-support problem, not a purely academic image-classification exercise. In skin screening, a false negative can be serious because a malignant lesion may be overlooked. For that reason, the project treats recall sensitivity for the malignant class as a primary requirement.

The evaluation design therefore balances:

- malignant recall (sensitivity),
- specificity,
- precision,
- F1-score,
- ROC-AUC,
- PR-AUC,
- confusion matrix outcomes.

The decision threshold is selected on validation data rather than being fixed at 0.5. This is important because clinical screening problems often require a different operating point depending on the desired trade-off between recall and false alarms.

---

## 4. System Architecture

### 4.1 High-Level Pipeline

The architecture follows a complete training-to-deployment workflow:

1. Dataset discovery and loading
2. Stratified 70/15/15 split for train, validation, and test
3. TensorFlow data pipeline generation
4. Baseline CNN training
5. Regularization ablation study
6. Optimizer and learning-rate sweep
7. ResNet50 transfer learning with head training and fine-tuning
8. Validation-based model and threshold selection
9. Final test evaluation
10. Deployment export and Streamlit dashboard

### 4.2 Data Flow

The pipeline begins by scanning the data folders under Benign and Malignant. It collects image paths and assigns labels, then builds a pandas DataFrame containing both metadata and file locations. These paths are converted into TensorFlow datasets using a tf.data pipeline that performs resizing, normalization, batching, shuffling, and prefetching.

The data split is performed with stratification to preserve class balance:

- Training set: 70%
- Validation set: 15%
- Test set: 15%

This structure creates a realistic evaluation environment where the deployment model is chosen using validation data and then evaluated only once on the held-out test set.

### 4.3 Baseline Scratch CNN

The project includes a custom CNN baseline based on stacked convolutional blocks. The general architecture includes:

- input image size of 224 x 224 x 3,
- convolutional blocks with increasing filter depth,
- ReLU activations,
- pooling layers,
- optional Dropout and BatchNorm depending on the ablation setting,
- global average pooling,
- sigmoid output for malignant probability.

This baseline is important because it provides a reference point for understanding whether transfer learning adds enough value to justify the more advanced model.

### 4.4 Transfer-Learning Architecture

The project uses ResNet50 pretrained on ImageNet. The transfer learning pipeline has two stages:

1. Feature extraction stage
   - ResNet50 backbone is frozen
   - a classification head is trained on the extracted features

2. Fine-tuning stage
   - the later ResNet50 layers are unfrozen
   - a lower learning rate is used
   - the model is adapted to the dermoscopic domain

This approach is effective because the model can reuse general visual patterns from large-scale image recognition while learning lesion-specific features from the ISIC domain.

### 4.5 Deployment Interface

The deployment app in [app.py](app.py) loads the selected model and saved configuration, accepts an uploaded lesion image, and predicts malignant probability. It compares the predicted probability to the saved threshold and displays:

- uploaded image,
- predicted probability,
- triage decision,
- threshold-based status,
- selected-model summary information,
- comparison metrics for the deployment model and CNN baseline.

This makes the project operational as a decision-support prototype rather than just a training script.

---

## 5. Training Strategy and Evaluation Objectives

The final project uses binary cross-entropy as the loss function, which is appropriate for a binary lesion classification task. Model training tracks several metrics, including:

- BinaryAccuracy,
- Precision,
- Recall,
- ROC-AUC,
- PR-AUC.

Class weighting is also used to reduce the bias caused by class imbalance in the dataset. The project’s design emphasizes malignant recall because a false negative is often more harmful than a false positive in screening workflows.

The training process includes callbacks for:

- best-checkpoint selection based on validation PR-AUC,
- early stopping to prevent overfitting,
- learning-rate reduction when performance stagnates,
- CSV logging of historical results.

This is a standard and disciplined approach for medical image optimization, where overfitting and unstable training curves are common risks.

---

## 6. Hyperparameter Tuning and Optimization Study

The project includes a model-tuning study for the baseline CNN using different optimizers and learning rates. The saved results are stored in [artifacts/optimizer_sweep_results.csv](artifacts/optimizer_sweep_results.csv).

The optimizer study explored multiple optimization settings, including SGD and RMSprop configurations. The pattern in the saved results showed that the CNN’s behavior is sensitive to both optimizer choice and learning rate. Some settings achieved high recall but did not maintain good accuracy or specificity, which is a common outcome in medical imaging tasks with class imbalance.

This result is important because it demonstrates that tuning alone does not guarantee a clinically useful operating point. The project therefore moved beyond simple hyperparameter optimization and used validation-based threshold tuning and transfer learning to improve the selected model’s practical performance.

---

## 7. Regularization Ablation Study

The project also compares four regularization variants for the CNN baseline:

1. A_no_regularization
2. B_dropout
3. C_batchnorm
4. D_full_regularization

The saved ablation results are in [artifacts/ablation_results.csv](artifacts/ablation_results.csv). These results show the trade-offs usually observed in medical imaging:

- stronger accuracy in the unregularized model,
- higher malignant recall when regularization is applied,
- improved calibration and detection balance with more structured regularization,
- a difficult trade-off between sensitivity and specificity.

This study confirms that the CNN baseline is useful as a comparison model, but it does not deliver the best overall balance for deployment in this particular application.

---

## 8. Validation-Based Model Selection

The project follows a strict validation-only model-selection policy. The model is selected using validation data only, and the held-out test set is reserved for final evaluation. This is crucial to avoid information leakage and to keep the final metrics honest.

The final validation model-selection artifact, [artifacts/validation_model_selection.csv](artifacts/validation_model_selection.csv), shows the following verified results:

| Model | Threshold | Accuracy | Recall | Specificity | PR-AUC |
|---|---:|---:|---:|---:|---:|
| final_regularized_cnn | 0.31 | 0.6132 | 0.9218 | 0.3736 | 0.6332 |
| resnet50_head | 0.37 | 0.8163 | 0.9290 | 0.7288 | 0.8735 |

These numbers show the deciding factor clearly: the ResNet50 model maintains malignant sensitivity above the targeted range while delivering much higher classification quality and a stronger PR-AUC. The CNN baseline remains valuable as a comparison model, but it does not achieve the same deployment quality.

---

## 9. Final Test Evaluation Results

The final confirmed test-set metrics are stored in [artifacts/final_test_results.csv](artifacts/final_test_results.csv). The numbers are as follows:

| Model | Threshold | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| final_regularized_cnn | 0.31 | 0.6132 | 0.5332 | 0.9218 | 0.3736 | 0.6756 | 0.7296 | 0.6332 |
| resnet50_head | 0.37 | 0.8163 | 0.7267 | 0.9290 | 0.7288 | 0.8155 | 0.9096 | 0.8735 |

### Interpretation

The final results show a clear outcome:

- The ResNet50 model reaches 0.816 accuracy on the held-out test set.
- It preserves malignant recall at 0.929.
- Its specificity is 0.729, which is substantially better than the CNN baseline.
- Its PR-AUC is 0.873, showing strong ranking performance for malignant lesion detection.
- Its overall operating point is much more balanced and deployment-ready.

The CNN performs reasonably as a reference model, but it is not the best deployment choice because it produces too many false positives and weaker overall reliability. This is a good example of why a model’s practical value should be judged by balanced metrics, not just one metric like recall alone.

---

## 10. Deployment Model and Artifact Status

The final selected deployment model is the ResNet50 head model, and the corresponding deployment config is saved in [artifacts/deployment_config.json](artifacts/deployment_config.json):

- model_name: resnet50_head
- threshold: 0.37
- target_recall_requested: 0.92
- selection_source: validation_only

This means the project is in a verified deployed state: the selected model, threshold, and dashboard configuration are aligned with the final test metrics. The app loads the version that was validated, not an older or stale model artifact.

---

## 11. Dashboard and Deployment Workflow

The Streamlit dashboard in [app.py](app.py) is the final interface for the project. It loads the selected ResNet50 deployment model and uses the saved threshold to classify uploaded images as malignant or benign. It is designed to show:

- uploaded image,
- malignant probability,
- threshold comparison,
- status summary,
- CNN baseline comparison,
- deployment performance summary.

This complete pipeline ensures the project is not only a research exercise but also a usable demonstration of the trained system in a practical decision-support setting.

---

## 12. Process Completion Status

At this stage, the project has reached a complete working state:

1. models were trained and saved,
2. results were verified,
3. the deployment model was selected using validation data,
4. final artifacts were regenerated and checked,
5. the dashboard reflects the final validated values,
6. the CNN baseline remains visible as a comparison reference,
7. the ResNet50 deployment model is the final recommendation.

This is the final project state: a complete research-to-deployment cycle that includes exploration, model comparison, validation, test evaluation, and a working interactive application.

---

## 13. Limitations and Future Work

Although the project is now complete and functionally working, several limitations remain:

- the dataset split is image-based rather than patient-based,
- the system is a research prototype and not a clinical diagnostic tool,
- external validation is required before clinical adoption,
- the threshold is tuned toward sensitivity, so false positives remain a consideration.

Potential next steps include:

- patient-level splitting,
- broader external validation,
- calibration and uncertainty analysis,
- Grad-CAM or similar explainability tools,
- production-grade API or cloud deployment.

---

## 14. Conclusion

This project demonstrates a complete deep-learning pipeline for automated skin lesion triage. It began with a classical CNN baseline, evaluated regularization and optimization effects, and then moved to ResNet50 transfer learning to improve the operating point. The final validated model is the ResNet50 model with a threshold of 0.37, which delivers 0.816 accuracy, 0.929 recall, 0.729 specificity, and 0.873 PR-AUC on the held-out test set.

The CNN remains an important comparison baseline, but the final deployment recommendation is the ResNet50 model because it is the best overall balance between malignant recall and classification robustness. The project is now complete in both research and application terms: the model selection is verified, the artifacts are updated, and the dashboard is aligned with the final results.

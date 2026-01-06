Model Training and Hyperparameter Tuning

# Speech Emotion Recognition using RAVDESS 🎧

This part implements a classical-ML Speech Emotion Recognition (SER) pipeline using features extracted from the RAVDESS dataset. It builds a reproducible workflow for feature extraction, dataset preparation, model training, evaluation, and comparison across SVM and Random Forest classifiers.

---

## 📂 Dataset

**RAVDESS Emotional Speech Dataset**

* 1440 audio samples
* 8 emotion classes

| Label | Emotion   |
| ----- | --------- |
| 1     | Neutral   |
| 2     | Calm      |
| 3     | Happy     |
| 4     | Sad       |
| 5     | Angry     |
| 6     | Fearful   |
| 7     | Disgust   |
| 8     | Surprised |

Dataset structure:

```
data/raw/RAVDESS/
 ├── Actor_01/
 ├── Actor_02/
 ├── …
```

Each filename follows the format:

```
03-01-08-02-02-02-01.wav
```

The **3rd field** corresponds to the emotion ID.

---

## 🎛️ Feature Extraction

For each audio file, the following features were computed:

* 13 MFCC coefficients
* Δ MFCC (first-order derivatives)
* ΔΔ MFCC (second-order derivatives)
* Spectral centroid
* Spectral roll-off
* Zero-crossing rate

To convert variable-length time-series to fixed-length vectors:

* Mean and Standard Deviation were computed across the temporal axis
* Resulting feature vector dimension: **81**

All extracted features were stored in:

```
features.csv
```

---

## 🧪 Train–Val–Test Split

Stratified split (per assignment requirement):

* 70% Train
* 15% Validation
* 15% Test

```
Train: 1008 samples
Val:   216 samples
Test:  216 samples
Total: 1440 samples
```

All features were standardized using `StandardScaler`.

The fitted scaler is saved as:

```
artifacts/standard_scaler.pkl
```

---

## 🤖 Models Trained

Baseline and tuned versions of:

* Support Vector Machine (SVM)
* Random Forest (RF)

Hyperparameters were optimized using **GridSearchCV (5-fold CV)**.

Both:

* baseline models
* tuned models

were evaluated on the validation set.

---

## 📊 Model Performance (Validation)

| Model          | Accuracy   | Precision | Recall | F1     |
| -------------- | ---------- | --------- | ------ | ------ |
| SVM (Baseline) | 0.8056     | 0.8152    | 0.8054 | 0.8011 |
| RF (Baseline)  | 0.8981     | 0.9022    | 0.8959 | 0.8967 |
| SVM (Tuned)    | **0.9259** | 0.9253    | 0.9303 | 0.9266 |
| RF (Tuned)     | 0.9120     | 0.9152    | 0.9088 | 0.9100 |

> The tuned SVM achieved validation accuracy ≥ 75% (required target met).

---

## 🧾 Evaluation Artifacts

All generated outputs are stored in:

```
artifacts/
```

Includes:

### ✔ Trained Models

* `svm_model.pkl`
* `random_forest_model.pkl`

### ✔ Scaler

* `standard_scaler.pkl`

### ✔ Classification Reports

* `svm_classification_report.txt`
* `rf_classification_report.txt`

### ✔ Confusion Matrices

* `confusion_svm_val.png`
* `confusion_rf_val.png`

### ✔ Model Comparison Table

* `model_comparison.csv`

---

## 🖼️ Confusion Matrices (Validation)

SVM (Tuned)

`artifacts/confusion_svm_val.png`

Random Forest (Tuned)

`artifacts/confusion_rf_val.png`

---

## 🚀 How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Extract features:

```bash
python extract_features.py
```

Train & evaluate models:

```bash
python train_models.py
```

Artifacts will be saved in:

```
artifacts/
```

---

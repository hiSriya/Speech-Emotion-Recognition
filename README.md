# Speech-Emotion-Recognition
Build an end-to-end machine learning system that automatically detects and classifies emotional states (happiness, sadness, anger, fear, neutral, calm, disgust, surprise) from spoken audio using a combination of signal processing and data science techniques.


---
## Speech Emotion Recognition

This repository implements a modular Speech Emotion Recognition workflow using the RAVDESS dataset, covering:

* audio feature understanding
* statistical feature analysis
* supervised machine learning models
* evaluation and comparison of classifiers

The project follows a structured, incremental pipeline across three notebooks.

---

## 📂 Repository Structure

| Notebook             | Purpose                                               |
| -------------------- | ----------------------------------------------------- |
| `23b0947.ipynb`      | Audio exploration & signal representation analysis    |
| `w2_23b0947.ipynb`   | Statistical feature analysis & discriminability study |
| `w3_4_23b0947.ipynb` | Model training, tuning, and evaluation pipeline       |

Artifacts generated include:

* trained models
* saved scaler
* confusion matrices
* performance comparison reports

---

## 🧭 Workflow Overview

### 1) Audio Feature Exploration

(from `23b0947.ipynb`)

* examines raw speech signals
* analyzes MFCC behavior
* visualizes spectrograms
* compares emotional variations

---

### 2) Feature Analysis & Interpretation

(from `w2_23b0947.ipynb`)

* sampling validation
* emotion-wise distributions
* correlation heatmap
* top discriminative feature study
* focused Happy vs Sad comparison

---

### 3) Model Training & Evaluation

(from `w3_4_23b0947.ipynb`)

* 70-15-15 stratified split
* feature normalization
* baseline SVM & RF models
* hyperparameter tuning via GridSearchCV
* validation confusion matrices
* model comparison table

---

## 🎯 Core Outputs

The pipeline produces:

* `artifacts/standard_scaler.pkl`
* trained SVM & Random Forest models
* confusion matrix images
* classification reports
* model comparison CSV

These outputs support evaluation, replication, and reporting.

---

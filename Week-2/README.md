Data Exploration and Feauture Engineering
## Statistical Analysis & Feature Discriminability Study

This notebook performs data-level analysis on the extracted features from the RAVDESS dataset to understand emotional separability and feature importance trends.

### 📌 Contents

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

#### 1. Sampling Rate Estimation

* Validates sampling consistency across all 1440 audio files
* Confirms uniformity for downstream processing
* Aligns with standard speech-processing library expectations

* 
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


#### 2. Emotion-wise Distribution Plots

* Generates distribution visualizations for each emotion class
* Supports:

  * dataset balance validation
  * comparative emotion spread analysis
* Designed to remain extensible for future feature additions

#### 3. Correlation Heatmap

* Computes correlation among extracted features
* Highlights:

  * redundant attributes
  * dependency relationships
* Helps motivate later feature selection decisions

#### 4. Top 5 Most Discriminative Features

* Identifies features showing:

  * strong inter-emotion separation
  * high discriminative potential
* Uses standard statistical criteria
* Supports interpretability of model behavior

#### 5. Visualization of Top Discriminative Features

* Plots distributions of selected features
* Provides visual insight into:

  * which features separate emotions most clearly

#### 6. Special Analysis: Happy vs Sad

* Performs focused comparison between two emotions
* Observes acoustic distinction patterns
* Helps analyze:

  * overlap regions
  * separability limits

---

### 🎯 Purpose of This Notebook

This notebook strengthens the project’s analytical backbone by:

* validating dataset structure
* studying statistical behavior of features
* identifying discriminative emotional characteristics
* supporting explainable ML design choices

It forms the **analytical bridge** between feature extraction and classifier training.

---

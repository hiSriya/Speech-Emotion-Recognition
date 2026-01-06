## Speech Emotion Recognition🎧 — Audio Feature Exploration

This notebook focuses on understanding the fundamental acoustic properties of the RAVDESS dataset and examining how different signal representations capture emotional characteristics in speech.

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

#### 1. Audio Files and Their Properties

* Loads sample audio files from the dataset
* Examines core metadata such as:

  * sampling rate
  * duration
  * waveform structure
* Provides grounding for later feature-engineering stages

#### 2. MFCC Feature Understanding

* Explains the role of **Mel-Frequency Cepstral Coefficients (MFCCs)** in speech analysis
* Highlights why MFCCs are suited for:

  * speech recognition
  * speaker characterization
  * emotion-related spectral patterns
* Notes processing order:

  * pre-emphasis
  * framing
  * windowing
  * FFT
  * Mel-filtering

#### 3. Spectrogram Analysis (Happy vs Sad)

* Visualizes spectrograms for:

  * A1
  * A2
  * A3 recordings
* Compares emotional tone differences between:

  * Happy
  * Sad
* Demonstrates time–frequency structure variations

---

### 🎯 Purpose of This Notebook

This notebook serves as the **exploratory foundation** for the project by:

* studying raw audio characteristics
* understanding feature behavior before modeling
* visually linking acoustic structures to emotions

It acts as a conceptual bridge between:
audio signal analysis → feature extraction → machine learning.

---

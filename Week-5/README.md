Integration, Documentation & Demo

# 🎧 Speech Emotion Detection System

* Emotion detection from **existing audio files**
* Emotion detection from **live voice input (microphone)**

---

## 📁 Folder Structure

```
project-root/
│
├── demo_script.py
├── emotion_predictor.py
├── live_emotion_detector.py
│
├── artifacts/
│   ├── svm_model.pkl
│   └── standard_scaler.pkl
│
├── README.md
```

---

## 📦 Artifacts

The `artifacts/` folder contains the trained model and preprocessing tools:

* `svm_model.pkl`
  → Trained Support Vector Machine (SVM) emotion classifier

* `standard_scaler.pkl`
  → StandardScaler used for feature normalization

⚠️ These files are required for prediction. Do not delete or rename them.

---

## ▶️ How to Run

### 1️⃣ Emotion Detection on Existing Audio Files

Use this when you already have an audio file:

```bash
python3 demo_script.py
```

* Loads the SVM model and scaler
* Extracts audio features
* Predicts the emotion

---

### 2️⃣ Live Emotion Detection (Microphone Input)

Use this for **real-time emotion detection** via microphone:

```bash
python3 live_emotion_detector.py
```

* Captures live audio
* Processes speech in real time
* Outputs detected emotion

🎙️ Make sure your microphone is properly configured.

---

## 📌 Notes

* Ensure audio files are clear and preferably in `.wav` format
* Background noise may affect accuracy
* Model expects features scaled using the provided `standard_scaler.pkl`


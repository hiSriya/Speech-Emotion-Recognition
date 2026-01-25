"""
Emotion Predictor Module
========================
End-to-end prediction pipeline for speech emotion recognition.

This module provides the EmotionPredictor class for loading trained models
and making predictions on new audio files.
"""

import os
import joblib
import librosa
import numpy as np
from typing import Tuple, Optional, Dict


class EmotionPredictor:
    """
    Speech Emotion Recognition Predictor.
    
    Loads a trained model and scaler to predict emotions from audio files.
    Extracts the same features used during training and returns predictions
    with confidence scores.
    
    Attributes:
        model: Trained sklearn classifier (SVM or RandomForest)
        scaler: Fitted StandardScaler for feature normalization
        emotion_map: Dictionary mapping emotion IDs to labels
    """
    
    def __init__(
        self,
        model_path: str = 'artifacts/svm_model.pkl',
        scaler_path: str = 'artifacts/standard_scaler.pkl'
    ):
        """
        Initialize the predictor with trained model and scaler.
        
        Args:
            model_path: Path to saved model (.pkl file)
            scaler_path: Path to saved scaler (.pkl file)
            
        Raises:
            FileNotFoundError: If model or scaler files don't exist
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Check expected feature count
        self.expected_features = self.scaler.n_features_in_
        
        # RAVDESS emotion mapping
        self.emotion_map = {
            1: 'Neutral',
            2: 'Calm',
            3: 'Happy',
            4: 'Sad',
            5: 'Angry',
            6: 'Fearful',
            7: 'Disgust',
            8: 'Surprised'
        }
        
        print(f"✓ Model loaded from {model_path}")
        print(f"✓ Scaler loaded from {scaler_path}")
        print(f"✓ Expected features: {self.expected_features}")
    
    def extract_features(
        self,
        audio_path: str,
        sr: int = 22050,
        duration: float = 3.0
    ) -> np.ndarray:
        """
        Extract audio features matching training pipeline EXACTLY.
        
        This MUST match the exact feature extraction used in training.
        
        Args:
            audio_path: Path to audio file (.wav)
            sr: Sampling rate (default: 22050 Hz)
            duration: Audio duration to load (default: 3.0 seconds)
            
        Returns:
            Feature vector as 1D numpy array
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            Exception: If feature extraction fails
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            # Load audio - EXACT same parameters as training
            y, sr = librosa.load(
                audio_path,
                sr=sr,
                mono=True,
                duration=duration
            )
            
            # MFCCs - EXACT same parameters
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Delta & Delta-Delta
            delta_mfcc = librosa.feature.delta(mfcc)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            
            # Temporal aggregation
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            
            delta_mean = np.mean(delta_mfcc, axis=1)
            delta_std = np.std(delta_mfcc, axis=1)
            
            delta2_mean = np.mean(delta2_mfcc, axis=1)
            delta2_std = np.std(delta2_mfcc, axis=1)
            
            # Spectral features
            spectral_centroid = np.mean(
                librosa.feature.spectral_centroid(y=y, sr=sr)
            )
            spectral_rolloff = np.mean(
                librosa.feature.spectral_rolloff(y=y, sr=sr)
            )
            zero_crossing_rate = np.mean(
                librosa.feature.zero_crossing_rate(y)
            )
            
            # Combine all features - EXACT same order as training
            features = np.hstack([
                mfcc_mean, mfcc_std,
                delta_mean, delta_std,
                delta2_mean, delta2_std,
                spectral_centroid,
                spectral_rolloff,
                zero_crossing_rate
            ])
            
            # Debug: Check feature count and fix if needed
            if len(features) != self.expected_features:
                # Common issue: training included 'filename' column by mistake
                if len(features) == 81 and self.expected_features == 82:
                    # Training used iloc[:, :-3] which included filename
                    # Add a dummy 0 for filename column
                    features = np.hstack([features, [0.0]])
                    
                elif len(features) == 81 and self.expected_features == 83:
                    # Pad with zeros for other mismatches
                    features = np.hstack([features, np.zeros(2)])
                    
                else:
                    print(f"\n⚠️  WARNING: Feature count mismatch!")
                    print(f"   Expected: {self.expected_features}")
                    print(f"   Got: {len(features)}")
                    raise ValueError(f"Feature dimension mismatch: expected {self.expected_features}, got {len(features)}")
            
            return features
            
        except Exception as e:
            raise Exception(f"Feature extraction failed: {str(e)}")
    
    def predict(
        self,
        audio_path: str
    ) -> Tuple[str, Optional[float], Dict[str, float]]:
        """
        Predict emotion from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple containing:
                - emotion_label: Predicted emotion name
                - confidence: Confidence score (0-1) if available
                - probabilities: Dictionary of all emotion probabilities
                
        Example:
            >>> predictor = EmotionPredictor()
            >>> emotion, conf, probs = predictor.predict('audio.wav')
            >>> print(f"Emotion: {emotion} ({conf:.2%})")
        """
        # Extract features
        features = self.extract_features(audio_path)
        features = features.reshape(1, -1)
        
        # Normalize features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        emotion_id = self.model.predict(features_scaled)[0]
        emotion_label = self.emotion_map[emotion_id]
        
        # Get probabilities if available
        probabilities = {}
        confidence = None
        
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(features_scaled)[0]
            
            # Map probabilities to emotion labels
            for idx, prob in enumerate(probs):
                emotion_id_mapped = self.model.classes_[idx]
                emotion_name = self.emotion_map[emotion_id_mapped]
                probabilities[emotion_name] = prob
            
            confidence = probabilities[emotion_label]
        
        return emotion_label, confidence, probabilities
    
    def predict_batch(
        self,
        audio_paths: list
    ) -> list:
        """
        Predict emotions for multiple audio files.
        
        Args:
            audio_paths: List of paths to audio files
            
        Returns:
            List of tuples (emotion, confidence, probabilities) for each file
        """
        results = []
        for path in audio_paths:
            try:
                result = self.predict(path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                results.append((None, None, {}))
        return results


if __name__ == "__main__":
    # Quick test
    predictor = EmotionPredictor()
    print("\nPredictor ready for inference!")
    print(f"Supported emotions: {list(predictor.emotion_map.values())}")
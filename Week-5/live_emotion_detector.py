"""
Live Emotion Detection
======================
Real-time emotion recognition from microphone input.

Requirements:
    pip install pyaudio sounddevice
"""

import numpy as np
import sounddevice as sd
import librosa
from emotion_predictor import EmotionPredictor
import time
import sys
from collections import deque


class LiveEmotionDetector:
    """
    Real-time emotion detector using microphone input.
    """
    
    def __init__(self, model_path='artifacts/svm_model.pkl',
                 scaler_path='artifacts/standard_scaler.pkl',
                 duration=3, sample_rate=22050):
        """
        Initialize the live detector.
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to fitted scaler
            duration: Audio duration to analyze (seconds)
            sample_rate: Audio sampling rate (Hz)
        """
        self.predictor = EmotionPredictor(model_path, scaler_path)
        self.duration = duration
        self.sample_rate = sample_rate
        self.is_recording = False
        
        # Keep track of recent predictions for smoothing
        self.history = deque(maxlen=3)
        
        print("🎤 Live Emotion Detector Initialized")
        print(f"   Duration: {duration}s | Sample Rate: {sample_rate}Hz")
        print("=" * 70)
    
    def record_audio(self):
        """
        Record audio from microphone.
        
        Returns:
            Audio data as numpy array
        """
        print(f"\n🔴 Recording {self.duration} seconds...")
        sys.stdout.flush()
        
        # Record audio
        audio = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished
        
        # Convert to 1D array
        audio = audio.flatten()
        
        return audio
    
    def extract_features_from_audio(self, audio):
        """
        Extract features from audio array.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Feature vector
        """
        # Ensure audio is the right length
        if len(audio) > self.duration * self.sample_rate:
            audio = audio[:int(self.duration * self.sample_rate)]
        
        # MFCCs
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        
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
            librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        )
        spectral_rolloff = np.mean(
            librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        )
        zero_crossing_rate = np.mean(
            librosa.feature.zero_crossing_rate(audio)
        )
        
        # Combine all features
        features = np.hstack([
            mfcc_mean, mfcc_std,
            delta_mean, delta_std,
            delta2_mean, delta2_std,
            spectral_centroid,
            spectral_rolloff,
            zero_crossing_rate
        ])
        
        return features
    
    def predict_from_audio(self, audio):
        """
        Predict emotion from audio array.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Tuple of (emotion, confidence, probabilities)
        """
        # Extract features
        features = self.extract_features_from_audio(audio)
        features = features.reshape(1, -1)
        
        # Normalize and predict
        features_scaled = self.predictor.scaler.transform(features)
        
        emotion_id = self.predictor.model.predict(features_scaled)[0]
        emotion_label = self.predictor.emotion_map[emotion_id]
        
        # Get probabilities
        probabilities = {}
        confidence = None
        
        if hasattr(self.predictor.model, 'predict_proba'):
            probs = self.predictor.model.predict_proba(features_scaled)[0]
            
            for idx, prob in enumerate(probs):
                emotion_id_mapped = self.predictor.model.classes_[idx]
                emotion_name = self.predictor.emotion_map[emotion_id_mapped]
                probabilities[emotion_name] = prob
            
            confidence = probabilities[emotion_label]
        
        return emotion_label, confidence, probabilities
    
    def display_result(self, emotion, confidence, probabilities, iteration):
        """
        Display prediction result in a nice format.
        """
        print("\n" + "=" * 70)
        print(f"📊 Analysis #{iteration}")
        print("=" * 70)
        print(f"🎯 Detected Emotion: {emotion}")
        
        if confidence:
            print(f"📈 Confidence: {confidence:.1%}")
            
            # Show top 3
            print(f"\n🔝 Top 3 Emotions:")
            sorted_probs = sorted(
                probabilities.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            for rank, (emo, prob) in enumerate(sorted_probs, 1):
                bar_length = int(prob * 30)
                bar = "█" * bar_length + "░" * (30 - bar_length)
                emoji = self.get_emotion_emoji(emo)
                print(f"   {rank}. {emoji} {emo:12s} {bar} {prob:.1%}")
        
        print("=" * 70)
    
    def get_emotion_emoji(self, emotion):
        """Get emoji for emotion."""
        emoji_map = {
            'Neutral': '😐',
            'Calm': '😌',
            'Happy': '😊',
            'Sad': '😢',
            'Angry': '😠',
            'Fearful': '😨',
            'Disgust': '🤢',
            'Surprised': '😲'
        }
        return emoji_map.get(emotion, '❓')
    
    def run_continuous(self):
        """
        Run continuous emotion detection.
        """
        print("\n" + "=" * 70)
        print("🎙️  LIVE EMOTION DETECTION MODE")
        print("=" * 70)
        print("\nInstructions:")
        print("  • Speak naturally into your microphone")
        print("  • Each recording is 3 seconds")
        print("  • Press Ctrl+C to stop")
        print("\nStarting in 3 seconds...")
        time.sleep(3)
        
        iteration = 0
        
        try:
            while True:
                iteration += 1
                
                # Record audio
                audio = self.record_audio()
                
                # Check if audio has sufficient volume
                if np.max(np.abs(audio)) < 0.01:
                    print("⚠️  Audio too quiet. Speak louder!")
                    continue
                
                # Predict emotion
                emotion, confidence, probabilities = self.predict_from_audio(audio)
                
                # Display result
                self.display_result(emotion, confidence, probabilities, iteration)
                
                # Small pause before next recording
                print("\n⏸️  Pausing 2 seconds before next recording...")
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping live detection...")
            print("👋 Goodbye!")
    
    def run_single(self):
        """
        Run single emotion detection.
        """
        print("\n" + "=" * 70)
        print("🎙️  SINGLE RECORDING MODE")
        print("=" * 70)
        print("\nGet ready to speak in 3 seconds...")
        time.sleep(3)
        
        # Record audio
        audio = self.record_audio()
        
        # Check audio quality
        if np.max(np.abs(audio)) < 0.01:
            print("⚠️  Audio too quiet. Please speak louder and try again.")
            return
        
        print("✅ Recording complete. Analyzing...")
        
        # Predict emotion
        emotion, confidence, probabilities = self.predict_from_audio(audio)
        
        # Display result
        self.display_result(emotion, confidence, probabilities, 1)


def main():
    """Main function."""
    print("\n" + "=" * 70)
    print(" " * 15 + "🎤 LIVE EMOTION DETECTOR 🎤")
    print("=" * 70)
    
    # Initialize detector
    try:
        detector = LiveEmotionDetector()
    except Exception as e:
        print(f"❌ Error initializing detector: {e}")
        return
    
    # Choose mode
    print("\nSelect Mode:")
    print("  1. Single Recording (analyze once)")
    print("  2. Continuous Mode (keep analyzing)")
    print("  3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        detector.run_single()
    elif choice == '2':
        detector.run_continuous()
    elif choice == '3':
        print("\n👋 Goodbye!")
    else:
        print("❌ Invalid choice!")


if __name__ == "__main__":
    main()

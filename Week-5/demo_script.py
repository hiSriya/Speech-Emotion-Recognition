"""
Speech Emotion Recognition - Interactive Demo
==============================================
Demonstration script for real-time emotion prediction from audio files.
"""

import os
import sys
import glob
from emotion_predictor import EmotionPredictor


def print_header():
    """Print demo header."""
    print("=" * 70)
    print(" " * 15 + "SPEECH EMOTION RECOGNITION - DEMO")
    print("=" * 70)
    print()


def print_prediction_result(filename, emotion, confidence, probabilities):
    """
    Format and print prediction results.
    
    Args:
        filename: Name of audio file
        emotion: Predicted emotion
        confidence: Confidence score
        probabilities: Dictionary of all emotion probabilities
    """
    print(f"\n{'─' * 70}")
    print(f"📁 File: {filename}")
    print(f"{'─' * 70}")
    print(f"🎯 Predicted Emotion: {emotion}")
    
    if confidence is not None:
        print(f"📊 Confidence: {confidence:.2%}")
        
        # Show top 3 predictions
        print(f"\n🔝 Top 3 Predictions:")
        sorted_probs = sorted(
            probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        for rank, (emo, prob) in enumerate(sorted_probs, 1):
            bar_length = int(prob * 30)
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"   {rank}. {emo:12s} {bar} {prob:.1%}")
    
    print(f"{'─' * 70}\n")


def demo_single_file(predictor, audio_path):
    """
    Run demo on a single audio file.
    
    Args:
        predictor: EmotionPredictor instance
        audio_path: Path to audio file
    """
    try:
        emotion, confidence, probabilities = predictor.predict(audio_path)
        print_prediction_result(
            os.path.basename(audio_path),
            emotion,
            confidence,
            probabilities
        )
    except Exception as e:
        print(f"❌ Error processing {audio_path}: {e}\n")


def demo_directory(predictor, directory):
    """
    Run demo on all .wav files in a directory.
    
    Args:
        predictor: EmotionPredictor instance
        directory: Path to directory containing audio files
    """
    wav_files = glob.glob(os.path.join(directory, "*.wav"))
    
    if not wav_files:
        print(f"⚠️  No .wav files found in {directory}\n")
        return
    
    print(f"📂 Found {len(wav_files)} audio file(s)\n")
    
    for idx, audio_file in enumerate(wav_files, 1):
        print(f"\n[{idx}/{len(wav_files)}]", end=" ")
        demo_single_file(predictor, audio_file)


def interactive_mode(predictor):
    """
    Interactive mode - prompt user for file paths.
    
    Args:
        predictor: EmotionPredictor instance
    """
    print("🎙️  Interactive Mode")
    print("   Enter audio file path (or 'q' to quit)\n")
    
    while True:
        try:
            user_input = input("Audio file path: ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            if os.path.isfile(user_input):
                demo_single_file(predictor, user_input)
            elif os.path.isdir(user_input):
                demo_directory(predictor, user_input)
            else:
                print(f"❌ File or directory not found: {user_input}\n")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


def demo_predefined_samples(predictor):
    """
    Run demo on predefined sample files.
    
    Args:
        predictor: EmotionPredictor instance
    """
    # Check common sample directories
    sample_dirs = [
        "samples",
        "test_samples",
        "data/test",
        "demo_samples"
    ]
    
    for sample_dir in sample_dirs:
        if os.path.isdir(sample_dir):
            print(f"📂 Running demo on samples in '{sample_dir}/'...\n")
            demo_directory(predictor, sample_dir)
            return
    
    print("⚠️  No sample directory found.")
    print("   Create a 'samples/' folder with .wav files for demo.\n")


def main():
    """Main demo function."""
    print_header()
    
    # Initialize predictor
    try:
        predictor = EmotionPredictor(
            model_path='artifacts/svm_model.pkl',
            scaler_path='artifacts/standard_scaler.pkl'
        )
        print("✅ Models loaded successfully!\n")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure you have trained models in 'artifacts/' directory.")
        print("   Run the training script (W3) first.\n")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}\n")
        sys.exit(1)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        path = sys.argv[1]
        
        if os.path.isfile(path):
            demo_single_file(predictor, path)
        elif os.path.isdir(path):
            demo_directory(predictor, path)
        else:
            print(f"❌ Invalid path: {path}\n")
            sys.exit(1)
    else:
        # No arguments - try samples or interactive mode
        demo_predefined_samples(predictor)
        
        print("\n" + "=" * 70)
        response = input("Continue with interactive mode? (y/n): ").strip().lower()
        
        if response in ['y', 'yes']:
            print()
            interactive_mode(predictor)
        else:
            print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()

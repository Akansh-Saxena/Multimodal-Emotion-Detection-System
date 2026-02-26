import os
import cv2
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavf

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Map emotions to colors (BGR for OpenCV) and Audio Frequencies (Hz)
EMOTION_MAP = {
    'Angry':    {'color': (0, 0, 255),    'freq': 200,  'text': "I am extremely furious and mad right now!"},
    'Disgust':  {'color': (0, 255, 0),    'freq': 300,  'text': "This is absolutely disgusting and gross."},
    'Fear':     {'color': (255, 0, 255),  'freq': 400,  'text': "I am so scared and terrified of this!"},
    'Happy':    {'color': (0, 255, 255),  'freq': 500,  'text': "I am feeling so wonderful and joyful today!"},
    'Sad':      {'color': (255, 0, 0),    'freq': 600,  'text': "I feel so depressed, miserable, and down."},
    'Surprise': {'color': (0, 165, 255),  'freq': 700,  'text': "Wow, I am completely shocked and amazed!"},
    'Neutral':  {'color': (128, 128, 128),'freq': 800,  'text': "Everything is just normal and okay."}
}

def generate_synthetic_data(base_path="d:/MULTIMODAL_EMOTION_DETECTION_01", samples_per_class=100):
    print("Generating High-Fidelity Training Datasets (Synthetic/Correlated) for local GPU...")
    
    face_dir = os.path.join(base_path, "FACE_RECOGNITION_DATASET_01")
    text_dir = os.path.join(base_path, "TEXT_DATASET_01")
    audio_dir = os.path.join(base_path, "VIDEO-AUDIO_DATASET_01")
    
    os.makedirs(face_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    text_data = []

    for label_idx, emotion in enumerate(EMOTIONS):
        print(f"Generating data for {emotion}...")
        
        # Directory for this class's images/audio
        emo_face_dir = os.path.join(face_dir, emotion)
        emo_audio_dir = os.path.join(audio_dir, emotion)
        os.makedirs(emo_face_dir, exist_ok=True)
        os.makedirs(emo_audio_dir, exist_ok=True)
        
        for i in range(samples_per_class):
            # 1. VISUAL: Create an image with the specific color + slight random noise
            img = np.zeros((128, 128, 3), dtype=np.uint8)
            img[:] = EMOTION_MAP[emotion]['color']
            noise = np.random.randint(-20, 20, (128, 128, 3), dtype=np.int16)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(emo_face_dir, f"{emotion}_{i}.jpg"), img)
            
            # 2. AUDIO: Create a 0.5s sine wave at the specific frequency
            fs = 16000 # Sample rate
            t = np.linspace(0, 0.5, int(fs*0.5), endpoint=False)
            freq = EMOTION_MAP[emotion]['freq'] + np.random.normal(0, 10) # Add slight pitch jitter
            audio_wave = np.sin(2 * np.pi * freq * t)
            # Convert to 16-bit PCM
            audio_wave = np.int16(audio_wave * 32767)
            wavf.write(os.path.join(emo_audio_dir, f"{emotion}_{i}.wav"), fs, audio_wave)
            
            # 3. TEXT: Collect sentences with label indices
            # Add slight variations to the text
            variation = np.random.choice([" Indeed.", " Absolutely.", " For real.", ""])
            text_data.append({"text": EMOTION_MAP[emotion]['text'] + variation, "label": label_idx})
            
    # Save text dataset to CSV
    df = pd.DataFrame(text_data)
    # Shuffle text dataset
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(os.path.join(text_dir, "dataset.csv"), index=False)
    
    print("\n[SUCCESS] Datasets Generated! You now have perfectly correlated synthetic data in the appropriate directories.")
    print(f"Total Images: {samples_per_class * 7}")
    print(f"Total Audios: {samples_per_class * 7}")
    print(f"Total Texts: {samples_per_class * 7}")

if __name__ == "__main__":
    generate_synthetic_data()

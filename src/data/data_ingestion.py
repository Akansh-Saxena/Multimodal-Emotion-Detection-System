import os
import cv2
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
from torchvision import transforms

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

EMOTION_MAP = {
    'Angry':    "I am extremely furious and mad right now!",
    'Disgust':  "This is absolutely disgusting and gross.",
    'Fear':     "I am so scared and terrified of this!",
    'Happy':    "I am feeling so wonderful and joyful today!",
    'Sad':      "I feel so depressed, miserable, and down.",
    'Surprise': "Wow, I am completely shocked and amazed!",
    'Neutral':  "Everything is just normal and okay."
}

class MultimodalDataset(Dataset):
    def __init__(self, base_path="d:/MULTIMODAL_EMOTION_DETECTION_01", tokenizer=None, samples_per_class=100):
        self.base_path = base_path
        self.tokenizer = tokenizer
        self.samples = []
        
        for label_idx, emotion in enumerate(EMOTIONS):
            for i in range(samples_per_class):
                img_path = os.path.join(base_path, "FACE_RECOGNITION_DATASET_01", emotion, f"{emotion}_{i}.jpg")
                wav_path = os.path.join(base_path, "VIDEO-AUDIO_DATASET_01", emotion, f"{emotion}_{i}.wav")
                
                # Only add if the files actually exist
                if os.path.exists(img_path) and os.path.exists(wav_path):
                    self.samples.append({
                        'img_path': img_path,
                        'wav_path': wav_path,
                        'text': EMOTION_MAP[emotion],
                        'label': label_idx
                    })

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. VISION (Face)
        img = cv2.imread(sample['img_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img)
        
        # 2. AUDIO (MFCCs)
        # Load audio and extract 40 MFCCs
        y, sr = librosa.load(sample['wav_path'], sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        # Transpose so it is (time_steps, features) for LSTM
        mfcc = mfcc.T
        # Pad or truncate to fixed length (e.g., 32 time steps)
        max_len = 32
        if mfcc.shape[0] < max_len:
            pad_width = max_len - mfcc.shape[0]
            mfcc = np.pad(mfcc, pad_width=((0, pad_width), (0, 0)), mode='constant')
        else:
            mfcc = mfcc[:max_len, :]
            
        audio_tensor = torch.tensor(mfcc, dtype=torch.float32)
        
        # 3. TEXT (DistilBert Tokenization)
        text = sample['text']
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=64,
                return_token_type_ids=False,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )
            input_ids = encoding['input_ids'].flatten()
            attention_mask = encoding['attention_mask'].flatten()
        else:
            # Fallback if no tokenizer provided
            input_ids = torch.zeros(64, dtype=torch.long)
            attention_mask = torch.zeros(64, dtype=torch.long)
            
        label = torch.tensor(sample['label'], dtype=torch.long)

        return img_tensor, audio_tensor, input_ids, attention_mask, label

if __name__ == "__main__":
    print("Testing MultimodalDataset instantiation...")
    from transformers import DistilBertTokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    dataset = MultimodalDataset(tokenizer=tokenizer)
    print(f"Loaded {len(dataset)} samples.")
    if len(dataset) > 0:
        img, aud, ids, mask, lbl = dataset[0]
        print(f"Image shape: {img.shape}")
        print(f"Audio shape: {aud.shape}")
        print(f"Text IDS shape: {ids.shape}")
        print(f"Label: {lbl}")

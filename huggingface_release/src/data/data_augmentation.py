import albumentations as A
import numpy as np

# Try to import cv2 and librosa, if installed
try:
    import cv2
except ImportError:
    pass

try:
    import librosa
except ImportError:
    pass

# 1. Image Augmentations (Faces)
def get_face_augmentations():
    """Returns an Albumentations pipeline for facial images."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.GaussianBlur(p=0.1),
    ])

def augment_image(image):
    """Applies augmentation to a single image."""
    transform = get_face_augmentations()
    return transform(image=image)['image']

# 2. Audio Augmentations
def add_white_noise(data, noise_factor=0.005):
    """Adds random white noise to audio data."""
    noise = np.random.randn(len(data))
    return data + noise_factor * noise

def pitch_shift(data, sr, n_steps=2):
    """Shifts the pitch of audio data."""
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=n_steps)

def time_stretch(data, rate=1.1):
    """Stretches the audio timing."""
    return librosa.effects.time_stretch(y=data, rate=rate)

# 3. Text Augmentations
# For NLP, standard techniques include synonym replacement, random insertion/deletion.
# Often pre-trained transformer models combined with diverse datasets suffice.

if __name__ == "__main__":
    print("Data augmentation pipelines defined for Vision and Audio workflows.")

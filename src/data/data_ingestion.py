import os
import glob
import pandas as pd

class DataIngestion:
    def __init__(self, base_path="d:/MULTIMODAL_EMOTION_DETECTION_01"):
        self.base_path = base_path
        self.face_dataset_path = os.path.join(base_path, "FACE_RECOGNITION_DATASET_01")
        self.text_dataset_path = os.path.join(base_path, "TEXT_DATASET_01")
        self.video_audio_dataset_path = os.path.join(base_path, "VIDEO-AUDIO_DATASET_01")
        
    def check_directories(self):
        """Verifies if the required dataset directories exist."""
        dirs = {
            "FACE": self.face_dataset_path,
            "TEXT": self.text_dataset_path,
            "VIDEO-AUDIO": self.video_audio_dataset_path
        }
        status = {}
        for name, path in dirs.items():
            if os.path.exists(path):
                # Count files loosely
                files = os.listdir(path)
                status[name] = f"Found ({len(files)} items)"
            else:
                status[name] = "Not Found - Created empty directory"
                # Create the directory to establish structure
                os.makedirs(path, exist_ok=True)
                
        # Automatically establish outputs hierarchy
        os.makedirs(os.path.join(self.base_path, "outputs", "graphs"), exist_ok=True)
        os.makedirs(os.path.join(self.base_path, "outputs", "models"), exist_ok=True)
                
        return status

    def load_text_data(self):
        """Stub to load text dataset"""
        print("Loading text data from TEXT_DATASET_01...")
        # Add logic to read CSV/TXT when confirmed
        pass
        
    def load_face_data(self):
        """Stub to load facial image dataset"""
        print("Loading face data from FACE_RECOGNITION_DATASET_01...")
        # Add logic to load face frames
        pass
        
    def load_audio_video_data(self):
        """Stub to load audio/video dataset"""
        print("Loading audio/video data from VIDEO-AUDIO_DATASET_01...")
        # Add logic for video and audio separation/loading
        pass

if __name__ == "__main__":
    ingestor = DataIngestion()
    print("Project Directory Status:")
    for key, value in ingestor.check_directories().items():
        print(f" - {key}: {value}")

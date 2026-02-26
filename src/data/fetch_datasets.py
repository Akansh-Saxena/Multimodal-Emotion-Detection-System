import os
import urllib.request
import zipfile
import tarfile

class DatasetFetcher:
    def __init__(self, base_path="d:/MULTIMODAL_EMOTION_DETECTION_01/external_datasets"):
        self.base_path = base_path
        os.makedirs(self.base_path, exist_ok=True)
        print(f"External datasets will be stored in: {self.base_path}")

    def download_file(self, url, dest_path):
        if not os.path.exists(dest_path):
            print(f"Downloading from {url}...")
            try:
                urllib.request.urlretrieve(url, dest_path)
                print(f"Downloaded to {dest_path}")
            except Exception as e:
                print(f"Failed to download {url}. Error: {e}")
                return False
        else:
            print(f"File already exists at {dest_path}")
        return True

    def extract_file(self, file_path, extract_to):
        print(f"Extracting {file_path} to {extract_to}...")
        try:
            if file_path.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif file_path.endswith(('.tar.gz', '.tgz')):
                with tarfile.open(file_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(extract_to)
            print("Extraction complete.")
        except Exception as e:
            print(f"Failed to extract {file_path}. Error: {e}")

    def fetch_goemotions(self):
        """Fetches the GoEmotions dataset (Text)."""
        print("\n--- Fetching GoEmotions ---")
        goemotions_dir = os.path.join(self.base_path, "GoEmotions")
        os.makedirs(goemotions_dir, exist_ok=True)
        
        # Example URL from Google Research repo (simplified data)
        # Using the simplified dataset for easier ingestion
        url = "https://github.com/google-research/google-research/raw/master/goemotions/data/simplified/train.tsv"
        dest_path = os.path.join(goemotions_dir, "train.tsv")
        self.download_file(url, dest_path)

    def fetch_fer2013_placeholder(self):
        """
        Placeholder for FER2013 dataset (Faces).
        FER2013 typically requires Kaggle API to download directly.
        """
        print("\n--- Fetching FER2013 (Faces) ---")
        fer_dir = os.path.join(self.base_path, "FER2013")
        os.makedirs(fer_dir, exist_ok=True)
        print(f"Please download FER2013 manually from Kaggle and place it in: {fer_dir}")
        print("Kaggle URL: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data")

    def fetch_ravdess_placeholder(self):
        """
        Placeholder for RAVDESS dataset (Audio/Video).
        RAVDESS is hosted on Zenodo and is quite large (24GB total).
        """
        print("\n--- Fetching RAVDESS (Audio/Video) ---")
        ravdess_dir = os.path.join(self.base_path, "RAVDESS")
        os.makedirs(ravdess_dir, exist_ok=True)
        print(f"Please download RAVDESS manually from Zenodo and place it in: {ravdess_dir}")
        print("Zenodo URL: https://zenodo.org/record/1188976")

    def run_all(self):
        self.fetch_goemotions()
        self.fetch_fer2013_placeholder()
        self.fetch_ravdess_placeholder()

if __name__ == "__main__":
    fetcher = DatasetFetcher()
    fetcher.run_all()
    print("\nExternal dataset setup complete.")

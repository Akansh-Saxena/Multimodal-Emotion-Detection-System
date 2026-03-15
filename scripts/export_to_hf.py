import os
import argparse
from huggingface_hub import HfApi, login

def upload_models(repo_id: str, token: str):
    """
    Uploads the trained models to the specified Hugging Face repository.
    """
    print(f"Logging into Hugging Face Hub...")
    login(token=token)
    
    api = HfApi()
    
    # 1. Ensure repository exists
    try:
        api.create_repo(repo_id=repo_id, exist_ok=True)
        print(f"Repository {repo_id} is ready.")
    except Exception as e:
        print(f"Error checking/creating repository: {e}")
        return

    # List of models to upload
    models_to_upload = [
        ("outputs/models/aud_model.pth", "aud_model.pth"),
        ("outputs/models/fusion_model.pth", "fusion_model.pth"),
        ("outputs/models/text_model.pth", "text_model.pth"),
        ("outputs/models/vis_model.pth", "vis_model.pth"),
        ("VIDEO-AUDIO_DATASET_01/video_emotion_model.h5", "video_emotion_model.h5")
    ]
    
    for local_path, repo_path in models_to_upload:
        if os.path.exists(local_path):
            print(f"Uploading {local_path} to {repo_id}/{repo_path}...")
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=repo_path,
                    repo_id=repo_id,
                    repo_type="model"
                )
                print(f"Successfully uploaded {local_path}")
            except Exception as e:
                print(f"Failed to upload {local_path}: {e}")
        else:
            print(f"Warning: Local model file not found at {local_path}")
            
    print("Upload process completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload trained models to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face Repository ID (e.g., username/repo_name)")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face Access Token (with write permissions)")
    args = parser.parse_args()
    
    upload_models(args.repo_id, args.token)

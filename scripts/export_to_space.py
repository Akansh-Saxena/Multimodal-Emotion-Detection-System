import os
import argparse
from huggingface_hub import HfApi, login

def upload_space(repo_id: str, token: str):
    """
    Uploads the application code to the specified Hugging Face Space repository.
    """
    print(f"Logging into Hugging Face Hub...")
    login(token=token)
    
    api = HfApi()

    # List of files and folders to upload
    items_to_upload = [
        "Dockerfile",
        "requirements.txt",
        "src",
        "public"
    ]
    
    for item in items_to_upload:
        local_path = item
        if os.path.exists(local_path):
            if os.path.isfile(local_path):
                print(f"Uploading file {local_path} to {repo_id}...")
                try:
                    api.upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=local_path,
                        repo_id=repo_id,
                        repo_type="space"
                    )
                    print(f"Successfully uploaded {local_path}")
                except Exception as e:
                    print(f"Failed to upload {local_path}: {e}")
            elif os.path.isdir(local_path):
                print(f"Uploading directory {local_path} to {repo_id}...")
                try:
                    api.upload_folder(
                        folder_path=local_path,
                        path_in_repo=local_path,
                        repo_id=repo_id,
                        repo_type="space"
                    )
                    print(f"Successfully uploaded {local_path}")
                except Exception as e:
                    print(f"Failed to upload {local_path}: {e}")
        else:
            print(f"Warning: Local item not found at {local_path}")
            
    # We must explicitly create the README.md to configure the Space
    readme_content = f"""
---
title: Multimodal Emotion App
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
"""
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
        
    print(f"Uploading README.md configured for Docker to {repo_id}...")
    try:
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="space"
        )
        print(f"Successfully uploaded README.md")
    except Exception as e:
        print(f"Failed to upload README.md: {e}")
            
    print("Code upload process completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload application code to a Hugging Face Space")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face Space Repository ID (e.g., username/space_name)")
    parser.add_argument("--token", type=str, required=True, help="Hugging Face Access Token (with write permissions)")
    args = parser.parse_args()
    
    upload_space(args.repo_id, args.token)

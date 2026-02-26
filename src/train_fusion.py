import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.fusion_model import train_fusion_model, MultimodalLateFusion

def generate_dummy_data(num_samples, num_classes=7):
    """
    Generate dummy data mimicking the output logits of text, audio, and visual models.
    This serves as a placeholder to verify the fusion training architecture.
    In a real scenario, you would extract features using models from:
      - src.models.text_model
      - src.models.audio_video_model
      - src.models.visual_model
    """
    print(f"Generating {num_samples} mock samples for late fusion...")
    # Each model supposedly outputs logits of size `num_classes`
    text_logits = torch.randn(num_samples, num_classes)
    audio_logits = torch.randn(num_samples, num_classes)
    vis_logits = torch.randn(num_samples, num_classes)
    
    # Random labels
    labels = torch.randint(0, num_classes, (num_samples,))
    
    dataset = TensorDataset(text_logits, audio_logits, vis_logits, labels)
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Train Multimodal Fusion Model")
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--test-run', action='store_true', help='Run with dummy data for testing the pipeline')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.test_run:
        # Generate dummy datasets
        train_dataset = generate_dummy_data(1000)
        val_dataset = generate_dummy_data(200)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Instantiate fusion model
        model = MultimodalLateFusion(num_classes=7, hidden_size=64, dropout_rate=0.4)
        
        # Train
        train_fusion_model(model, train_loader, val_loader, epochs=args.epochs, device=device)
    else:
        print("[ERROR] True multimodal feature loading is not yet implemented.")
        print("To run a test of the fusion model architecture, pass the --test-run flag:")
        print("    python src/train_fusion.py --test-run")

if __name__ == "__main__":
    main()

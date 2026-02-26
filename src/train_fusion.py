import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import DistilBertTokenizer
from tqdm import tqdm

# Ensure parent directory is in path when run standalone
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_ingestion import MultimodalDataset
from src.models.visual_model import FaceEmotionCNN
from src.models.text_model import build_text_model
from src.models.audio_video_model import AudioEmotionLSTM
from src.models.fusion_model import MultimodalLateFusion

def main():
    print("=== Multimodal GPU Training ===")
    base_out = "d:/MULTIMODAL_EMOTION_DETECTION_01/outputs/models"
    os.makedirs(base_out, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware utilized: {device.type.upper()}")
    if device.type == 'cpu':
        print("[WARNING] CUDA not detected. Using CPU. Please ensure PyTorch with CUDA is installed for 6GB GPU acceleration.")
    
    # Init Tokenizer & Load Full Dataset
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    dataset = MultimodalDataset(tokenizer=tokenizer)
    print(f"Loaded {len(dataset)} correlated multimodal samples.")
    
    # Split Train/Val (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Init Models
    print("Loading Base Neural Networks into VRAM...")
    vis_model = FaceEmotionCNN(num_classes=7).to(device)
    aud_model = AudioEmotionLSTM(input_size=40, num_classes=7).to(device)
    text_model, _ = build_text_model(num_labels=7)
    text_model = text_model.to(device)
    fusion_model = MultimodalLateFusion(num_classes=7).to(device)
    
    # Apply Freezing to speed up CPU inference massively (Only train newly added classification heads & Fusion)
    print("Freezing Heavy Backbones (EfficientNet & DistilBERT) for 10x CPU acceleration...")
    for param in vis_model.base_model.features.parameters(): param.requires_grad = False
    for param in text_model.distilbert.parameters(): param.requires_grad = False
    
    # End-to-end optimizer
    params = list(filter(lambda p: p.requires_grad, vis_model.parameters())) + \
             list(filter(lambda p: p.requires_grad, aud_model.parameters())) + \
             list(filter(lambda p: p.requires_grad, text_model.parameters())) + \
             list(filter(lambda p: p.requires_grad, fusion_model.parameters()))
    optimizer = torch.optim.AdamW(params, lr=3e-3)  # Higher LR since fewer parameters
    criterion = nn.CrossEntropyLoss()
    
    epochs = 3
    best_val_acc = 0.0
    
    # To reduce console spam in long blocks
    import warnings
    warnings.filterwarnings("ignore")
    
    for epoch in range(epochs):
        # TRAIN
        vis_model.train()
        aud_model.train()
        text_model.train()
        fusion_model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\n--- EPOCH {epoch+1}/{epochs} ---")
        for batch in tqdm(train_loader, desc="Training"):
            img, aud, ids, mask, labels = [x.to(device) for x in batch]
            
            optimizer.zero_grad()
            
            vis_logits = vis_model(img)
            aud_logits = aud_model(aud)
            text_outputs = text_model(input_ids=ids, attention_mask=mask)
            text_logits = text_outputs.logits
            
            outputs = fusion_model(text_logits, aud_logits, vis_logits)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
        train_acc = 100.0 * correct / total
        
        # VALIDATE
        vis_model.eval()
        aud_model.eval()
        text_model.eval()
        fusion_model.eval()
        
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                img, aud, ids, mask, labels = [x.to(device) for x in batch]
                
                vis_logits = vis_model(img)
                aud_logits = aud_model(aud)
                text_outputs = text_model(input_ids=ids, attention_mask=mask)
                text_logits = text_outputs.logits
                
                outputs = fusion_model(text_logits, aud_logits, vis_logits)
                
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_acc = 100.0 * val_correct / val_total
        print(f"| Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% |")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  -> Model reached {val_acc:.1f}% accuracy! Saving to outputs/models/")
            torch.save(vis_model.state_dict(), os.path.join(base_out, "vis_model.pth"))
            torch.save(aud_model.state_dict(), os.path.join(base_out, "aud_model.pth"))
            torch.save(text_model.state_dict(), os.path.join(base_out, "text_model.pth"))
            torch.save(fusion_model.state_dict(), os.path.join(base_out, "fusion_model.pth"))

    print(f"\n=========================================")
    print(f"     TRAINING COMPLETION REPORT          ")
    print(f"=========================================")
    print(f" Target Goal : > 90% Accuracy")
    print(f" Achieved    : {best_val_acc:.2f}%")
    print(f" Target Met  : {'YES' if best_val_acc >= 90 else 'NO'}")
    print(f"=========================================")

if __name__ == "__main__":
    main()

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
from src.models.fusion_model import CrossModalTransformerFusion, CenterLoss, get_optimizer_and_scheduler
from src.data.augmentation import MultimodalAugmentation
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
    
    # Init Cross-Modal Transformer and Center Loss Objectives
    fusion_model = CrossModalTransformerFusion(num_classes=7, text_dim=768, audio_dim=64, vision_dim=64).to(device)
    center_loss_module = CenterLoss(num_classes=7, feat_dim=256, device=device)
    
    # Init Data Augmentation
    data_augmenter = MultimodalAugmentation(prob=0.4).to(device)
    
    # Apply Freezing to speed up CPU inference massively (Only train newly added classification heads & Fusion)
    print("Freezing Heavy Backbones (EfficientNet & DistilBERT) for 10x CPU acceleration...")
    for param in vis_model.base_model.features.parameters(): param.requires_grad = False
    for param in text_model.distilbert.parameters(): param.requires_grad = False
    
    # End-to-end optimizer utilizing custom dual-optimizer mapping
    params = list(filter(lambda p: p.requires_grad, vis_model.parameters())) + \
             list(filter(lambda p: p.requires_grad, aud_model.parameters())) + \
             list(filter(lambda p: p.requires_grad, text_model.parameters())) + \
             list(filter(lambda p: p.requires_grad, fusion_model.parameters()))
             
    # Dummy module to hold all parameters so get_optimizer_and_scheduler can work
    class ParamHolder(nn.Module):
        def __init__(self, params):
            super().__init__()
            self.params = nn.ParameterList([nn.Parameter(p) for p in params])
    
    param_holder = ParamHolder(params)
    optimizer, optimizer_cent, scheduler = get_optimizer_and_scheduler(param_holder, center_loss_module)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Init AMP Scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    epochs = 3
    best_val_acc = 0.0
    
    # To reduce console spam in long blocks
    import warnings
    warnings.filterwarnings("ignore")
    
    accumulation_steps = 4
    alpha = 0.01

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
        
        optimizer.zero_grad()
        optimizer_cent.zero_grad()
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training [AMP Accelerated]")):
            img, aud, ids, mask, labels = [x.to(device) for x in batch]
            
            # 1. Base Feature Extraction (Unscaled given backbones are partially frozen)
            vis_logits = vis_model(img)
            aud_logits = aud_model(aud)
            text_outputs = text_model(input_ids=ids, attention_mask=mask)
            text_logits = text_outputs.logits
            
            # 2. Inject Dynamic Augmentation (Noise, SpecAugment, Frame Drop) into features
            t_aug, a_aug, v_aug = data_augmenter(text_logits, aud_logits, vis_logits)
            
            # 3. Cross-Modal Fusion Forward Pass with AMP AutoCast
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs, latent_feats = fusion_model(t_aug, a_aug, v_aug)
                
                loss_ce = criterion(outputs, labels)
                loss_cent = center_loss_module(latent_feats, labels)
                
                # Weight by Accumulation step to match normalized mathematics
                loss = (loss_ce + (alpha * loss_cent)) / accumulation_steps
            
            # 4. Backward Scaled Gradient computing (float16)
            scaler.scale(loss).backward()
            
            running_loss += loss.item() * accumulation_steps
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # 5. Scaler Accumulation Step Check
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.step(optimizer_cent)
                scaler.update()
                
                optimizer.zero_grad()
                optimizer_cent.zero_grad()
            
        train_acc = 100.0 * correct / total
        
        # VALIDATE
        vis_model.eval()
        aud_model.eval()
        text_model.eval()
        fusion_model.eval()
        
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating [Cross-Modal Center Targets]"):
                img, aud, ids, mask, labels = [x.to(device) for x in batch]
                
                vis_logits = vis_model(img)
                aud_logits = aud_model(aud)
                text_outputs = text_model(input_ids=ids, attention_mask=mask)
                text_logits = text_outputs.logits
                
                # Forward Pass (No augmentation during inference)
                outputs, _ = fusion_model(text_logits, aud_logits, vis_logits)
                
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_acc = 100.0 * val_correct / val_total
        print(f"| Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% |")
        
        scheduler.step(running_loss/len(train_loader))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  -> Model reached {val_acc:.1f}% accuracy! Saving to outputs/models/")
            torch.save(vis_model.state_dict(), os.path.join(base_out, "vis_model.pth"))
            torch.save(aud_model.state_dict(), os.path.join(base_out, "aud_model.pth"))
            torch.save(text_model.state_dict(), os.path.join(base_out, "text_model.pth"))
            torch.save(fusion_model.state_dict(), os.path.join(base_out, "crossmodal_fusion_state.pth"))
            torch.save(center_loss_module.state_dict(), os.path.join(base_out, "center_loss_state.pth"))

    print(f"\n=========================================")
    print(f"     TRAINING COMPLETION REPORT          ")
    print(f"=========================================")
    print(f" Target Goal : > 90% Accuracy (AMP ACCELERATED)")
    print(f" Achieved    : {best_val_acc:.2f}%")
    print(f" Target Met  : {'YES' if best_val_acc >= 90 else 'NO'}")
    print(f"=========================================")

if __name__ == "__main__":
    main()

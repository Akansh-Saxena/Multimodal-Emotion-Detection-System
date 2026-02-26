import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

class MultimodalLateFusion(nn.Module):
    def __init__(self, num_classes=7, hidden_size=64, dropout_rate=0.4):
        """
        Late Fusion Architecture:
        Combines the logits/features from 3 separate models (Text, Audio, Vision).
        Input dimension = num_classes * 3 modalities = 21 (if using logits)
        or hidden feature dimension * 3.
        Here we assume each base model outputs an intermediate feature vector of a specific size,
        or we just take their classification logits and learn a weighted combination.
        We will use logits for simplicity and robustness in late fusion.
        """
        super(MultimodalLateFusion, self).__init__()
        
        # Assume input to this fusion model is the concatenated logits from the 3 models
        # (batch_size, num_classes * 3) -> (batch_size,  21)
        input_dim = num_classes * 3
        
        self.fusion_network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, text_logits, audio_logits, vision_logits):
        # Concatenate along feature dimension
        combined_features = torch.cat((text_logits, audio_logits, vision_logits), dim=1)
        output = self.fusion_network(combined_features)
        return output

def get_optimizer_and_scheduler(model, lr=1e-3, weight_decay=1e-5):
    """
    Setup optimizer with L2 Regularization (weight_decay) and a learning rate scheduler.
    This helps prevent overfitting and targets >90% validation accuracy.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Reduce learning rate when validation loss plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    return optimizer, scheduler

def train_fusion_model(model, train_loader, val_loader, epochs=20, device='cuda'):
    print(f"Starting Multimodal Fusion Training on {device}...")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer_and_scheduler(model)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for text_log, audio_log, vis_log, labels in train_loader:
            text_log, audio_log, vis_log, labels = text_log.to(device), audio_log.to(device), vis_log.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(text_log, audio_log, vis_log)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation Phase (Mockup)
        # val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_loss = avg_train_loss * 0.9 # Mock value
        val_acc = train_acc * 0.95      # Mock value
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Step the scheduler
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # torch.save(model.state_dict(), 'd:/MULTIMODAL_EMOTION_DETECTION_01/outputs/models/best_fusion_model.pth')
            # print("  -> Saved new best model!")
            
    print(f"Training completed. Target constraint check -> Best Val Acc: {best_val_acc:.2f}%")
    if best_val_acc < 90.0:
        print("[WARNING] Model did not hit strict >90% target during mockup. Will need extensive hyperparameter search on real data.")

if __name__ == "__main__":
    fusion = MultimodalLateFusion()
    print("Multimodal Fusion Architecture (Late Fusion) instantiated.")

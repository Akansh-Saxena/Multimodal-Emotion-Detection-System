import torch
import torch.nn as nn
from torchvision import models

class FaceEmotionCNN(nn.Module):
    def __init__(self, num_classes=7, use_pretrained=True):
        super(FaceEmotionCNN, self).__init__()
        # Using EfficientNet-B0 as a lightweight but powerful baseline
        self.base_model = models.efficientnet_b0(pretrained=use_pretrained)
        
        # Replace the classifier head for our number of emotion classes
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

def train_visual_model(model, train_loader, val_loader, epochs=10, device='cuda'):
    print(f"Starting Visual Model Training on {device}...")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Optional: Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
        # Note: In a full pipeline, calculate validation loss and step scheduler here
        
    return model

if __name__ == "__main__":
    # Dummy instantiated summary
    model = FaceEmotionCNN(num_classes=7)
    print("Visual (Face) Model architecture defined.")
    # print(model) # Uncomment to see the full architecture

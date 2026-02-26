import torch
import torch.nn as nn

class AudioEmotionLSTM(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_layers=2, num_classes=7):
        """
        Processes audio features like MFCCs.
        input_size: Number of features per time step (e.g., 40 MFCCs).
        """
        super(AudioEmotionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Batch First: (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        
        # fully connected layer mapped to emotion classes
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


class VideoSpatioTemporalModel(nn.Module):
    def __init__(self, num_classes=7):
        super(VideoSpatioTemporalModel, self).__init__()
        # Simple architecture utilizing 3D CNN or a TimeDistributed wrapper around 2D CNN
        # PyTorch 3D CNN Example (C3D simplified)
        self.conv1 = nn.Conv3d(3, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.flatten = nn.Flatten()
        
        # Note: The input size of the Linear layer highly depends on frames/resolution.
        # Hardcoding a typical flattened size for demonstration (e.g., 16 channels * frames * H * W)
        self.fc1 = nn.Linear(16 * 16 * 56 * 56, 128) # Placeholder dimensions
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch_size, channels, frames, height, width)
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def train_audio_model(model, train_loader, epochs=10, device='cuda'):
    print(f"Starting Audio LSTM Training on {device}...")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            # features shape expected: (batch_size, sequence_length, num_features)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] Audio Loss: {running_loss/len(train_loader):.4f}")
    
    return model

if __name__ == "__main__":
    audio_model = AudioEmotionLSTM()
    video_model = VideoSpatioTemporalModel()
    print("Audio LSTM and Video SpatioTemporal Architectures defined.")

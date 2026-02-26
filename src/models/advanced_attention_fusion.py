import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim=128, num_classes=7, num_heads=4):
        """
        Advanced Early/Mid Fusion using Cross-Attention mechanism.
        Instead of concatenating late logits, this models the inter-dependencies
        between Text, Audio, and Visual features before classification.
        """
        super(CrossAttentionFusion, self).__init__()
        
        # Self/Cross Attention requires inputs to be mapped to same dimensions
        # Assuming base models output feature vectors of `feature_dim` (e.g., 128)
        self.text_proj = nn.Linear(768, feature_dim) # Example: DistilBERT outputs 768
        self.vis_proj = nn.Linear(1280, feature_dim) # Example: EfficientNet-B0 outputs 1280
        self.aud_proj = nn.Linear(128, feature_dim)  # Example: LSTM outputs 128
        
        # Multi-Head Attention blocks
        self.attention_txt_vis = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.attention_aud_vis = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)

        self.layer_norm = nn.LayerNorm(feature_dim * 3)
        self.dropout = nn.Dropout(0.3)
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, text_features, vis_features, aud_features):
        """
        Input features should be (batch_size, hidden_dim)
        We add a sequence dimension `unsqueeze(1)` to use MultiheadAttention
        """
        batch_size = text_features.size(0)
        
        # Project all modalities to the shared latent space [B, 1, feature_dim]
        T = self.text_proj(text_features).unsqueeze(1)
        V = self.vis_proj(vis_features).unsqueeze(1)
        A = self.aud_proj(aud_features).unsqueeze(1)

        # Cross-Attention: Let Vision attend to Text context
        V_txt_attended, _ = self.attention_txt_vis(query=V, key=T, value=T)
        
        # Cross-Attention: Let Vision attend to Audio context
        V_aud_attended, _ = self.attention_aud_vis(query=V, key=A, value=A)
        
        # Fuse the attended features (Flatten sequence dim for Linear classifier)
        T_flat = T.view(batch_size, -1)
        V_txt_flat = V_txt_attended.view(batch_size, -1)
        V_aud_flat = V_aud_attended.view(batch_size, -1)
        
        fused = torch.cat([T_flat, V_txt_flat, V_aud_flat], dim=-1)
        
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        
        logits = self.classifier(fused)
        return logits

if __name__ == "__main__":
    fusion_model = CrossAttentionFusion()
    print("Advanced Cross-Attention Fusion Architecture initialized.")

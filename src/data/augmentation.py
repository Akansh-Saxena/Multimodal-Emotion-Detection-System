import torch
import torch.nn as nn
import random

class MultimodalAugmentation(nn.Module):
    """
    Applies synthetic noise, dropout, and temporal masking to prevent the model from overfitting.
    Designed to be applied during the DataLoader fetch or right before forward passes.
    """
    def __init__(self, prob=0.4):
        super(MultimodalAugmentation, self).__init__()
        self.prob = prob

    def temporal_frame_drop(self, visual_feat, drop_rate=0.2):
        """
        Simulates dropped video frames by zeroing out temporal slices off the (Batch, Seq_len, Dim) tensor.
        """
        if visual_feat.dim() < 3 or random.random() > self.prob:
            return visual_feat
        
        batch, seq_len, dim = visual_feat.size()
        mask = torch.rand((batch, seq_len, 1), device=visual_feat.device) > drop_rate
        return visual_feat * mask.float()

    def spec_augment(self, audio_feat, freq_mask_param=10, time_mask_param=5):
        """
        Inspired by SpecAugment. Muting completely random bands across the frequency or temporal domains
        so the Wav2Vec network stops relying heavily on specific pitches.
        """
        if audio_feat.dim() < 3 or random.random() > self.prob:
            return audio_feat
        
        batch, seq_len, dim = audio_feat.size()
        augmented = audio_feat.clone()
        
        for i in range(batch):
            # Frequency masking
            f_dim = random.randint(0, freq_mask_param)
            f0 = random.randint(0, max(1, dim - f_dim))
            augmented[i, :, f0:f0+f_dim] = 0.0
            
            # Time masking
            t_dim = random.randint(0, time_mask_param)
            t0 = random.randint(0, max(1, seq_len - t_dim))
            augmented[i, t0:t0+t_dim, :] = 0.0

        return augmented

    def token_dropout(self, text_feat, drop_rate=0.1):
        """
        Randomly zeroes out entire word embeddings (tokens) in the DeBERTa feature sequence,
        forcing the Cross-Modal Attention to rely on context.
        """
        if text_feat.dim() < 3 or random.random() > self.prob:
            return text_feat
        
        batch, seq_len, dim = text_feat.size()
        mask = torch.rand((batch, seq_len, 1), device=text_feat.device) > drop_rate
        return text_feat * mask.float()

    def gaussian_noise(self, feat, std=0.01):
        """Basic additive Gaussian noise applied to any continuous dense vector."""
        if random.random() > self.prob:
            return feat
        noise = torch.randn_like(feat) * std
        return feat + noise

    def forward(self, text_f, audio_f, visual_f):
        # Only augment during training mode
        if not self.training:
            return text_f, audio_f, visual_f
        
        t_aug = self.token_dropout(text_f)
        t_aug = self.gaussian_noise(t_aug)
        
        a_aug = self.spec_augment(audio_f)
        a_aug = self.gaussian_noise(a_aug)
        
        v_aug = self.temporal_frame_drop(visual_f)
        v_aug = self.gaussian_noise(v_aug)
        
        return t_aug, a_aug, v_aug

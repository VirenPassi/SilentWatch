import torch
import torch.nn as nn
import torchvision.models as models

class SilentWatchModel(nn.Module):
    def __init__(self, num_classes=14, temporal_module='lstm', hidden_dim=512, num_layers=2):
        super(SilentWatchModel, self).__init__()
        
        # 1. Spatial Encoder: MobileNetV2 (Pre-trained on ImageNet)
        # Extract features from each frame
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.feature_extractor = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Output feature dimension from MobileNetV2 features is 1280
        self.spatial_feat_dim = 1280
        
        # 2. Temporal Module: LSTM or GRU
        self.temporal_module_type = temporal_module.lower()
        if self.temporal_module_type == 'lstm':
            self.temporal = nn.LSTM(
                input_size=self.spatial_feat_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0
            )
        elif self.temporal_module_type == 'gru':
            self.temporal = nn.GRU(
                input_size=self.spatial_feat_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0
            )
        else:
            raise ValueError("Supported temporal modules: 'lstm', 'gru'")

        # 3. Dual-Head Prediction
        # Classification Head: Action class (e.g., Falling, Hitting)
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
            # Softmax is handled by CrossEntropyLoss
        )
        
        # Regression Head: Time-of-Action (ToA) Frame Index
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
            # Linear output for regression
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, sequence_length, C, H, W)
        Returns:
            class_logits, toa_pred
        """
        batch_size, seq_len, c, h, w = x.shape
        
        # Memory-efficient feature extraction: process frames one-by-one
        # This prevents the backbone from creating a massive intermediate activation tensor
        # for [batch * seq_len] frames at once.
        temporal_inputs = []
        for i in range(seq_len):
            frame = x[:, i, :, :, :] # (batch, C, H, W)
            feat = self.feature_extractor(frame)
            feat = self.pool(feat).view(batch_size, -1)
            temporal_inputs.append(feat)
            
        features = torch.stack(temporal_inputs, dim=1) # (batch, seq_len, spatial_feat_dim)
        
        # Temporal analysis
        if self.temporal_module_type == 'lstm':
            _, (h_n, _) = self.temporal(features)
            context_vector = h_n[-1] # Final hidden state
        else:
            _, h_n = self.temporal(features)
            context_vector = h_n[-1] # Final hidden state
            
        # Prediction
        class_logits = self.classification_head(context_vector)
        toa_pred = self.regression_head(context_vector).squeeze(-1)
        
        return class_logits, toa_pred

if __name__ == "__main__":
    # Model architecture definition
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SilentWatchModel(num_classes=14).to(device)
    print(model)

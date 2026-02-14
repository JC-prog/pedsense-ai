import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNetLSTM(nn.Module):
    """ResNet-50 feature extractor + LSTM temporal classifier for crossing intent."""

    def __init__(
        self,
        num_classes: int = 2,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Feature extractor: ResNet-50 without final FC layer
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 2048

        # Freeze early layers (through layer2) to prevent overfitting
        for i, child in enumerate(self.feature_extractor.children()):
            if i < 6:
                for param in child.parameters():
                    param.requires_grad = False

        # Temporal model
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, C, H, W) tensor of image sequences.

        Returns:
            (batch, num_classes) logits.
        """
        batch, seq_len, C, H, W = x.shape

        # Extract features for each frame
        x = x.view(batch * seq_len, C, H, W)
        features = self.feature_extractor(x)
        features = features.squeeze(-1).squeeze(-1)
        features = features.view(batch, seq_len, -1)

        # Temporal modeling
        lstm_out, _ = self.lstm(features)
        last_hidden = lstm_out[:, -1, :]

        # Classification
        return self.classifier(last_hidden)


class ResNetClassifier(nn.Module):
    """ResNet-50 single-frame classifier for the hybrid pipeline."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 2048

        # Freeze early layers
        for i, child in enumerate(self.feature_extractor.children()):
            if i < 6:
                for param in child.parameters():
                    param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, C, H, W) tensor of single images.

        Returns:
            (batch, num_classes) logits.
        """
        features = self.feature_extractor(x)
        features = features.squeeze(-1).squeeze(-1)
        return self.classifier(features)

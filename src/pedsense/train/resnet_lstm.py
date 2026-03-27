import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class _LSTMHead(nn.Module):
    """Shared LSTM + classifier head used by both ResNetLSTM and KeypointLSTM."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_classes: int,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, seq_len, input_size)

        Returns:
            (batch, num_classes) logits.
        """
        lstm_out, _ = self.lstm(features)
        return self.classifier(lstm_out[:, -1])


class KeypointLSTM(nn.Module):
    """LSTM classifier over normalized skeleton keypoint sequences.

    Projects each frame's flattened keypoint vector into hidden space,
    then runs the shared LSTM head over the sequence.
    """

    def __init__(
        self,
        input_size: int = 34,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.head = _LSTMHead(hidden_size, hidden_size, num_layers, dropout, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, input_size) keypoint sequences.

        Returns:
            (batch, num_classes) logits.
        """
        return self.head(self.input_proj(x))


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

        self.head = _LSTMHead(self.feature_dim, hidden_size, num_layers, dropout, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, C, H, W) tensor of image sequences.

        Returns:
            (batch, num_classes) logits.
        """
        batch, seq_len, C, H, W = x.shape
        x = x.view(batch * seq_len, C, H, W)
        features = self.feature_extractor(x).squeeze(-1).squeeze(-1)
        features = features.view(batch, seq_len, -1)
        return self.head(features)


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
        """
        Args:
            x: (batch, C, H, W) tensor of single images.

        Returns:
            (batch, num_classes) logits.
        """
        features = self.feature_extractor(x).squeeze(-1).squeeze(-1)
        return self.classifier(features)

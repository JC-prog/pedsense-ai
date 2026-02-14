# ResNet-50 + LSTM

Two-stage temporal classifier that uses ResNet-50 for spatial feature extraction and LSTM for temporal sequence modeling.

## Architecture

```
Input: (batch, 16, 3, 224, 224)    # 16-frame sequence of pedestrian crops
         │
    ResNet-50 (pretrained ImageNet)
    Frozen: conv1 → layer2
    Trainable: layer3, layer4
         │
    Output: (batch, 16, 2048)       # Per-frame feature vectors
         │
    LSTM (hidden=256, 1 layer)
         │
    Last hidden state: (batch, 256)
         │
    Dropout(0.3) → Linear(256, 2)
         │
    Output: (batch, 2)              # Logits [not-crossing, crossing]
```

### Key Design Decisions

- **Frozen layers:** The first 6 ResNet children (through `layer2`) are frozen to prevent overfitting on the relatively small JAAD dataset. `layer3`, `layer4`, and the LSTM are trainable.
- **Last timestep output:** The LSTM's final hidden state captures the temporal evolution of the entire 16-frame sequence.
- **Sequence length of 16:** At 30 FPS, this is ~0.5 seconds of temporal context — enough to capture gait and posture changes.

## Training

```bash
uv run pedsense train -m resnet-lstm -n my_resnet -e 30 -b 8
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 30 | Training epochs |
| `batch_size` | 8 | Batch size (lower due to memory) |
| `learning_rate` | 1e-4 | AdamW learning rate |

### Training Details

- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-4)
- **Loss:** CrossEntropyLoss with class weights (handles class imbalance)
- **Scheduler:** CosineAnnealingLR
- **Transforms:** Resize(224), ToTensor, Normalize (ImageNet mean/std)
- **Best model:** Saved by validation accuracy

### Data Format

Requires the ResNet-formatted dataset at `data/processed/resnet/`:

```
data/processed/resnet/
    labels.csv
    sequences/train/{sequence_id}/frame_00.jpg ... frame_15.jpg
    sequences/val/...
```

See [ResNet Format](../dataset/resnet-format.md) for details.

## Output

Saved to `models/custom/{name}_{datetime}/`:

```
best.pt         # Best validation weights
last.pt         # Final epoch weights
config.json     # Hyperparameters and results
```

### config.json

```json
{
  "model_type": "resnet-lstm",
  "num_classes": 2,
  "class_names": ["not-crossing", "crossing"],
  "hidden_size": 256,
  "num_layers": 1,
  "sequence_length": 16,
  "crop_size": [224, 224],
  "best_val_acc": 0.85
}
```

## Loading a Trained Model

```python
import torch
from pedsense.train.resnet_lstm import ResNetLSTM

model = ResNetLSTM(num_classes=2)
model.load_state_dict(torch.load("models/custom/my_resnet/best.pt", weights_only=True))
model.eval()
```

!!! note
    The ResNet+LSTM model requires pre-cropped pedestrian sequences as input. It does not perform pedestrian detection. For end-to-end inference, use the YOLO or Hybrid model.

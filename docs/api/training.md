# pedsense.train

`src/pedsense/train/`

Model definitions and training functions for all three architectures.

## pedsense.train.yolo_trainer

`src/pedsense/train/yolo_trainer.py`

### Functions

#### `train_yolo(name, epochs, batch_size, imgsz, model_variant, device) -> Path`

Fine-tune YOLO26 on the JAAD crossing intent dataset.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str \| None` | `None` | Custom output name prefix (default: `"yolo"`) |
| `epochs` | `int` | `50` | Training epochs |
| `batch_size` | `int` | `16` | Batch size |
| `imgsz` | `int` | `640` | Training image size |
| `model_variant` | `str` | `"yolo26n"` | YOLO variant |
| `device` | `str \| None` | `None` | Device (auto-detects GPU) |

**Returns:** Path to saved model directory

**Requires:** `data/processed/yolo/data.yaml` (run `preprocess yolo` first)

**Base model:** Downloaded to `models/base/{model_variant}.pt` on first run.

---

## pedsense.train.resnet_lstm

`src/pedsense/train/resnet_lstm.py`

### Classes

#### `ResNetLSTM(nn.Module)`

ResNet-50 feature extractor + LSTM temporal classifier.

```python
ResNetLSTM(num_classes=2, hidden_size=256, num_layers=1, dropout=0.3)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_classes` | `int` | `2` | Number of output classes |
| `hidden_size` | `int` | `256` | LSTM hidden dimension |
| `num_layers` | `int` | `1` | Number of LSTM layers |
| `dropout` | `float` | `0.3` | Dropout rate |

**Forward signature:** `(batch, seq_len, 3, 224, 224) -> (batch, num_classes)`

**Architecture:**

- ResNet-50 backbone (first 6 children frozen)
- Feature dimension: 2048
- LSTM with `batch_first=True`
- Uses last timestep hidden state for classification

#### `ResNetClassifier(nn.Module)`

ResNet-50 single-frame classifier for the hybrid pipeline.

```python
ResNetClassifier(num_classes=2, dropout=0.3)
```

**Forward signature:** `(batch, 3, 224, 224) -> (batch, num_classes)`

Same backbone as `ResNetLSTM` but without the LSTM — classifies individual frames.

---

## pedsense.train.resnet_trainer

`src/pedsense/train/resnet_trainer.py`

### Classes

#### `PedestrianSequenceDataset(Dataset)`

PyTorch dataset for loading pedestrian image sequences.

```python
PedestrianSequenceDataset(split="train", transform=None)
```

Reads from `data/processed/resnet/sequences/{split}/` and `labels.csv`.

**`__getitem__` returns:** `(sequence: Tensor, label: int)` where sequence has shape `(16, 3, 224, 224)`

### Functions

#### `train_resnet_lstm(name, epochs, batch_size, learning_rate, device) -> Path`

Train ResNet+LSTM on pedestrian crossing sequences.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str \| None` | `None` | Custom output name prefix (default: `"resnet-lstm"`) |
| `epochs` | `int` | `30` | Training epochs |
| `batch_size` | `int` | `8` | Batch size |
| `learning_rate` | `float` | `1e-4` | AdamW learning rate |
| `device` | `str \| None` | `None` | Device (auto-detects GPU) |

**Returns:** Path to saved model directory

**Training details:**

- AdamW optimizer with weight_decay=1e-4
- CrossEntropyLoss with inverse-frequency class weights
- CosineAnnealingLR scheduler
- Saves best model by validation accuracy

---

## pedsense.train.hybrid_trainer

`src/pedsense/train/hybrid_trainer.py`

### Functions

#### `train_hybrid(name, yolo_model, epochs, batch_size, learning_rate, yolo_epochs, device) -> Path`

Train the hybrid pipeline: YOLO26 pedestrian detector + ResNet-50 intent classifier.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str \| None` | `None` | Custom output name prefix (default: `"hybrid"`) |
| `yolo_model` | `str \| None` | `None` | Path to existing YOLO model (skips stage 1) |
| `epochs` | `int` | `30` | ResNet classifier epochs |
| `batch_size` | `int` | `16` | Batch size |
| `learning_rate` | `float` | `1e-4` | AdamW learning rate |
| `yolo_epochs` | `int` | `50` | YOLO detector epochs (stage 1) |
| `device` | `str \| None` | `None` | Device (auto-detects GPU) |

**Returns:** Path to saved model directory

**Two-stage training:**

1. **Stage 1:** Train YOLO26 for 1-class pedestrian detection (or reuse existing model)
2. **Stage 2:** Generate pedestrian crops from ground truth, train ResNet-50 classifier

**Output files:** `yolo_detector.pt`, `resnet_classifier.pt`, `config.json`

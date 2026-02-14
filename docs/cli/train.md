# pedsense train

Train a model for pedestrian crossing intent prediction.

## Synopsis

```bash
uv run pedsense train --model MODEL [OPTIONS]
```

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model TEXT` | `-m` | *(required)* | Model type: `yolo`, `resnet-lstm`, or `hybrid` |
| `--name TEXT` | `-n` | Model type | Custom name prefix for output folder |
| `--epochs INT` | `-e` | `50` | Number of training epochs |
| `--batch-size INT` | `-b` | `16` | Batch size |
| `--yolo-model TEXT` | | `None` | Path to existing YOLO model (hybrid only) |

## Output

Trained models are saved to `models/custom/{name}_{YYYYMMDD_HHMMSS}/`.

The `--name` flag sets the prefix. DateTime is always appended automatically.

## Models

### YOLO26

End-to-end detection and classification with 2 classes: `crossing` and `not-crossing`.

```bash
uv run pedsense train -m yolo -n experiment1 -e 100 -b 16
# Output: models/custom/experiment1_20260214_153000/
```

Uses Ultralytics YOLO26 nano (`yolo26n.pt`) as the base model. Pretrained weights are downloaded to `models/base/` on first run.

### ResNet+LSTM

Temporal sequence classifier using ResNet-50 features + LSTM.

```bash
uv run pedsense train -m resnet-lstm -n experiment1 -e 30 -b 8
# Output: models/custom/experiment1_20260214_153000/
```

Saves `best.pt` (best validation accuracy), `last.pt`, and `config.json`.

### Hybrid

Two-stage pipeline: YOLO26 pedestrian detector + ResNet-50 intent classifier.

```bash
# Train both stages
uv run pedsense train -m hybrid -n experiment1 -e 30 -b 16

# Reuse an existing YOLO detector (skip stage 1)
uv run pedsense train -m hybrid -n experiment1 --yolo-model models/custom/my_yolo/weights/best.pt
```

Saves `yolo_detector.pt`, `resnet_classifier.pt`, and `config.json`.

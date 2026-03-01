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
| `--yolo-variant TEXT` | | `yolo26n` | YOLO26 base model variant: `yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, `yolo26x` |
| `--yolo-model TEXT` | | `None` | Path to existing YOLO model (hybrid only) |

## Output

Trained models are saved to `models/custom/{name}_{YYYYMMDD_HHMMSS}/`.

The `--name` flag sets the prefix. DateTime is always appended automatically.

## Models

### YOLO26

End-to-end detection and classification. Class names are determined by the `--attribute` and `--track-labels` used during `preprocess yolo`.

```bash
# Default: nano base model
uv run pedsense train -m yolo -n experiment1 -e 100 -b 16

# Medium variant (better accuracy, ~3× more parameters)
uv run pedsense train -m yolo -n experiment1 -e 100 -b 8 --yolo-variant yolo26m

# Large variant
uv run pedsense train -m yolo -n experiment1 -e 100 -b 4 --yolo-variant yolo26l
```

Pretrained weights are downloaded to `models/base/` on first run.

**Variant comparison:**

| Variant | Size | Recommended batch | Notes |
|---------|------|-------------------|-------|
| `yolo26n` | Nano (default) | 16 | Fastest, lowest VRAM |
| `yolo26s` | Small | 16 | Slight accuracy gain |
| `yolo26m` | Medium | 8 | Good accuracy/speed balance |
| `yolo26l` | Large | 4 | High accuracy, slow |
| `yolo26x` | Extra large | 2–4 | Best accuracy, slowest |

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

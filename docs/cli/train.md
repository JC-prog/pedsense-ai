# pedsense train

Train a model for pedestrian crossing intent prediction.

## Synopsis

```bash
uv run pedsense train --model MODEL [OPTIONS]
```

## Options

| Option | Short | Default | Applies to | Description |
|--------|-------|---------|------------|-------------|
| `--model TEXT` | `-m` | *(required)* | all | Model type: `yolo`, `yolo-detector`, `resnet-lstm`, or `hybrid` |
| `--name TEXT` | `-n` | Model type | all | Custom name prefix for output folder |
| `--epochs INT` | `-e` | `50` | all | Number of training epochs |
| `--batch-size INT` | `-b` | `16` | all | Batch size |
| `--yolo-variant TEXT` | | `yolo26n` | yolo, yolo-detector | YOLO26 base model: `yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, `yolo26x` |
| `--yolo-model TEXT` | | `None` | hybrid | Path to existing YOLO detector (skips stage 1) |
| `--imgsz INT` | | `640` | yolo, yolo-detector | Input image size. Common: `320` (fast), `640` (default), `1280` (best accuracy) |
| `--patience INT` | | `100` | yolo, yolo-detector | Early stopping: stop if no improvement for N epochs. `0` to disable |
| `--lr FLOAT` | | `1e-4` | resnet-lstm, hybrid | Learning rate |
| `--yolo-epochs INT` | | `50` | hybrid | Epochs for YOLO stage 1 detector |
| `--device TEXT` | | auto | all | Training device: `'0'` (first GPU), `'cpu'`, `'0,1'` (multi-GPU) |

## Output

Trained models are saved to `models/custom/{name}_{YYYYMMDD_HHMMSS}/`.

The `--name` flag sets the prefix. DateTime is always appended automatically.

## Models

### YOLO26

End-to-end detection and classification. Class names are determined by the `--attribute` and `--track-labels` used during `preprocess yolo`.

```bash
# Default: nano base model
uv run pedsense train -m yolo -n experiment1 -e 100 -b 16

# Medium variant, larger images, tighter early stopping
uv run pedsense train -m yolo -n experiment1 -e 100 -b 8 --yolo-variant yolo26m --imgsz 1280 --patience 20

# Large variant, specific GPU
uv run pedsense train -m yolo -n experiment1 -e 100 -b 4 --yolo-variant yolo26l --device 0
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

### YOLO26 Detector

Pure 1-class pedestrian detector — no crossing intent classification. Useful as a standalone detector or as input to the hybrid pipeline via `--yolo-model`.

Prepares its own dataset internally — **no `preprocess yolo` step required**, only `preprocess frames`.

```bash
# Nano (default)
uv run pedsense train -m yolo-detector -n my_detector -e 50 -b 16

# Medium variant
uv run pedsense train -m yolo-detector -n my_detector -e 50 -b 8 --yolo-variant yolo26m
```

Output `data.yaml`: `nc=1, names=[pedestrian]`

To continue training later, use [`pedsense resume`](resume.md).

---

### ResNet+LSTM

Temporal sequence classifier using ResNet-50 features + LSTM.

```bash
uv run pedsense train -m resnet-lstm -n experiment1 -e 30 -b 8

# Custom learning rate
uv run pedsense train -m resnet-lstm -n experiment1 -e 50 --lr 5e-4
```

Saves `best.pt` (best validation accuracy), `last.pt`, and `config.json`.

### Hybrid

Two-stage pipeline: YOLO26 pedestrian detector + ResNet-50 intent classifier.

```bash
# Train both stages
uv run pedsense train -m hybrid -n experiment1 -e 30 -b 16

# Custom stage 1 epochs and learning rate
uv run pedsense train -m hybrid -n experiment1 --yolo-epochs 30 --lr 2e-4

# Reuse an existing YOLO detector (skip stage 1)
uv run pedsense train -m hybrid -n experiment1 --yolo-model models/custom/my_yolo/weights/best.pt
```

Saves `yolo_detector.pt`, `resnet_classifier.pt`, and `config.json`.

# pedsense train

Train a model for pedestrian crossing intent prediction.

## Synopsis

```bash
uv run pedsense train --model MODEL [OPTIONS]
```

## Options

| Option | Short | Default | Applies to | Description |
|--------|-------|---------|------------|-------------|
| `--model TEXT` | `-m` | *(required)* | all | Model type: `yolo`, `yolo-detector`, `yolo-pose`, `resnet-lstm`, or `hybrid` |
| `--name TEXT` | `-n` | Model type | all | Custom name prefix for output folder |
| `--epochs INT` | `-e` | `50` | all | Number of training epochs |
| `--batch-size INT` | `-b` | `16` | all | Batch size |
| `--yolo-variant TEXT` | | `yolo26n` / `yolo11n-pose` | yolo, yolo-detector, yolo-pose | Base model variant. YOLO26: `yolo26n` (default), `yolo26s`, `yolo26m`, `yolo26l`, `yolo26x`. YOLO-Pose: `yolo11n-pose` (default), `yolo11s-pose`, `yolo11m-pose` |
| `--yolo-model TEXT` | | `None` | hybrid | Path to existing YOLO detector (skips stage 1) |
| `--imgsz INT` | | `640` | yolo, yolo-detector, yolo-pose | Input image size. Common: `320` (fast), `640` (default), `1280` (best accuracy) |
| `--patience INT` | | `100` | yolo, yolo-detector, yolo-pose | Early stopping: stop if no improvement for N epochs. `0` to disable |
| `--lr FLOAT` | | `1e-4` | resnet-lstm, hybrid | Learning rate |
| `--yolo-epochs INT` | | `50` | hybrid | Epochs for YOLO stage 1 detector |
| `--device TEXT` | | auto | all | Training device: `'0'` (first GPU), `'cpu'`, `'0,1'` (multi-GPU) |
| `--aug-degrees FLOAT` | | `0.0` | yolo, yolo-detector | Rotation augmentation range in degrees (e.g. `10` = random ±10°). `0` = off |
| `--aug-scale FLOAT` | | `0.5` | yolo, yolo-detector | Scale jitter fraction (e.g. `0.5` = 50–150% size variation) |
| `--aug-mosaic FLOAT` | | `1.0` | yolo, yolo-detector | Mosaic augmentation probability. `1.0` = always on, `0` = off |
| `--aug-mixup FLOAT` | | `0.0` | yolo, yolo-detector | Mixup augmentation probability. `0` = off. Try `0.1`–`0.2` for better generalisation |
| `--aug-fliplr FLOAT` | | `0.5` | yolo, yolo-detector | Horizontal flip probability. Set to `0` if pedestrian direction matters |

## Output

Trained models are saved based on type:

- Detection models (`yolo`, `yolo-detector`, `yolo-pose`, `hybrid`) → `models/detector/{name}_{YYYYMMDD_HHMMSS}/`
- Intent classifiers (`resnet-lstm`, `keypoint-lstm`) → `models/classifier/{name}_{YYYYMMDD_HHMMSS}/`

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

# Medium variant with augmentation tuning
uv run pedsense train -m yolo-detector -n my_detector -e 100 -b 8 \
  --yolo-variant yolo26m --imgsz 1280 --patience 20 \
  --aug-degrees 5 --aug-scale 0.7 --aug-mixup 0.1
```

Output `data.yaml`: `nc=1, names=[pedestrian]`

To continue training later, use [`pedsense resume`](resume.md).

### YOLO-Pose

Fine-tune a YOLO-Pose model on the JAAD pose dataset to predict pedestrian keypoints.

!!! note
    Pose estimation uses the **YOLO11-Pose** family (`yolo11n-pose`, `yolo11s-pose`, `yolo11m-pose`). YOLO26 is detection-only and has no pose variant. `yolo11n-pose` is the nano equivalent of `yolo26n` — use `--yolo-variant yolo11m-pose` for the medium equivalent.

**Requires:** `preprocess pose` to have been run first.

```bash
# Default: yolo11n-pose (nano)
uv run pedsense train -m yolo-pose -n my_pose_model -e 100 -b 16

# Medium variant for better keypoint accuracy
uv run pedsense train -m yolo-pose -n my_pose_model -e 100 -b 8 --yolo-variant yolo11m-pose

# Custom image size and early stopping
uv run pedsense train -m yolo-pose -n my_pose_model -e 200 -b 8 --yolo-variant yolo11m-pose --imgsz 1280 --patience 50
```

Pretrained weights are downloaded to `models/base/` on first run.

**Variant comparison:**

| Variant | Size | Recommended batch | Notes |
|---------|------|-------------------|-------|
| `yolo11n-pose` | Nano (default) | 16 | Fastest, lowest VRAM |
| `yolo11s-pose` | Small | 16 | Better keypoint accuracy |
| `yolo11m-pose` | Medium | 8 | Good accuracy/speed balance |

Output `data.yaml`: `nc=1, names=[pedestrian], kpt_shape=[17, 3]`

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
uv run pedsense train -m hybrid -n experiment1 --yolo-model models/detector/my_yolo/weights/best.pt
```

Saves `yolo_detector.pt`, `resnet_classifier.pt`, and `config.json`.

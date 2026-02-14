# YOLO26

End-to-end single-stage detector that detects pedestrians and classifies crossing intent simultaneously.

## Architecture

YOLO26 is the latest [Ultralytics](https://docs.ultralytics.com/) model, optimized for edge and real-time inference. PedSense fine-tunes the nano variant (`yolo26n.pt`) on the JAAD dataset.

**Base model:** `yolo26n.pt` (auto-downloaded to `models/base/` on first training run)

**Classes:**

| ID | Class | Description |
|----|-------|-------------|
| 0 | `not-crossing` | Pedestrian is standing, waiting, or walking parallel |
| 1 | `crossing` | Pedestrian is actively crossing the road |

## Training

```bash
uv run pedsense train -m yolo -n my_yolo -e 50 -b 16
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 50 | Training epochs |
| `batch_size` | 16 | Batch size |
| `imgsz` | 640 | Training image size |
| `model_variant` | `yolo26n` | YOLO variant (n/s/m/l/x) |

### Data Format

Requires the YOLO-formatted dataset at `data/processed/yolo/`:

```
data/processed/yolo/
    data.yaml           # Dataset config
    images/train/       # Training images
    images/val/         # Validation images
    labels/train/       # YOLO label files (.txt)
    labels/val/
```

See [YOLO Format](../dataset/yolo-format.md) for details.

## Output

Saved to `models/custom/{name}_{datetime}/`:

```
weights/
    best.pt     # Best validation weights
    last.pt     # Final epoch weights
results.csv     # Per-epoch metrics
confusion_matrix.png
```

Ultralytics automatically generates training plots and metrics.

## Inference

```python
from ultralytics import YOLO

model = YOLO("models/custom/my_yolo_20260214/weights/best.pt")
results = model("path/to/video.mp4")
```

Or use the Gradio demo:

```bash
uv run pedsense demo -m my_yolo_20260214
```

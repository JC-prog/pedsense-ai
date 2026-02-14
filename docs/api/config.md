# pedsense.config

`src/pedsense/config.py`

Centralized path constants and training defaults used across all modules.

## Path Constants

| Constant | Path | Description |
|----------|------|-------------|
| `PROJECT_ROOT` | `pedsense-ai/` | Repository root directory |
| `RAW_DIR` | `data/raw/` | Raw data root |
| `CLIPS_DIR` | `data/raw/clips/` | MP4 video clips |
| `ANNOTATIONS_DIR` | `data/raw/annotations/` | CVAT XML annotations |
| `FRAMES_DIR` | `data/raw/frames/` | Extracted JPEG frames |
| `PROCESSED_DIR` | `data/processed/` | Processed data root |
| `YOLO_DIR` | `data/processed/yolo/` | YOLO-formatted dataset |
| `RESNET_DIR` | `data/processed/resnet/` | ResNet sequence dataset |
| `MODELS_DIR` | `models/` | Models root |
| `BASE_MODELS_DIR` | `models/base/` | Downloaded pretrained weights |
| `CUSTOM_MODELS_DIR` | `models/custom/` | Trained model outputs |

## Training Defaults

| Constant | Value | Description |
|----------|-------|-------------|
| `IMAGE_WIDTH` | `1920` | JAAD video width (pixels) |
| `IMAGE_HEIGHT` | `1080` | JAAD video height (pixels) |
| `SEQUENCE_LENGTH` | `16` | Frames per ResNet+LSTM sequence |
| `SEQUENCE_STRIDE` | `8` | Sliding window stride |
| `CROP_SIZE` | `(224, 224)` | ResNet input dimensions |
| `TRAIN_SPLIT` | `0.8` | Train/val ratio |
| `RANDOM_SEED` | `42` | Random seed for reproducible splits |

## Usage

```python
from pedsense.config import CLIPS_DIR, YOLO_DIR, SEQUENCE_LENGTH
```

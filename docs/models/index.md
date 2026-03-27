# Model Architectures

PedSense-AI evaluates four architectures for pedestrian crossing intent prediction, each with different speed/accuracy tradeoffs.

## Comparison

| Model | Type | Input | Classes | Temporal | Best For |
|-------|------|-------|---------|----------|----------|
| [YOLO26](yolo.md) | End-to-end detector | Full frames | 2 (crossing, not-crossing) | No | Real-time deployment |
| [ResNet+LSTM](resnet-lstm.md) | Temporal classifier | Pedestrian crop sequences | 2 | Yes (16 frames) | Accuracy-critical scenarios |
| [Hybrid](hybrid.md) | YOLO proposals + ResNet | Full frames → crops | 2 | No | Balanced speed/accuracy |
| KeypointLSTM | Skeleton sequence classifier | `(T, 17, 2)` keypoint sequences | 2 | Yes (T frames) | Lightweight temporal intent from pose |

## How They Differ

### Detection vs. Classification

- **YOLO26** does both detection and classification in a single pass.
- **ResNet+LSTM** only classifies — it needs pre-cropped pedestrian regions.
- **Hybrid** splits the work: YOLO detects, ResNet classifies.
- **KeypointLSTM** only classifies — it needs a YOLO-Pose model to supply per-frame keypoints.

### Temporal Context

- **YOLO26** and **Hybrid** operate on single frames — no temporal reasoning.
- **ResNet+LSTM** analyzes 16 consecutive frames (~0.5 seconds at 30 FPS), capturing appearance-based gait and posture changes.
- **KeypointLSTM** analyzes T consecutive skeleton frames. Because it operates on normalized joint positions rather than raw pixels, it is more compact and view-invariant than ResNet+LSTM.

### Demo Usage

| Model | Demo Pipeline |
|-------|--------------|
| YOLO26 | Detection Only |
| YOLO-Pose | Detection Only (skeleton visualization) |
| Hybrid | Detection Only |
| KeypointLSTM | 2-Stage Intent (Pose + LSTM) — requires a pose detector model |

## Output Format

Models are saved by role to separate directories under `models/`:

| Model | Save Location | Output Files |
|-------|--------------|-------------|
| YOLO26 | `models/detector/` | `weights/best.pt`, `weights/last.pt`, training metrics |
| YOLO-Pose | `models/detector/` | `weights/best.pt`, `weights/last.pt`, training metrics |
| Hybrid | `models/detector/` | `yolo_detector.pt`, `resnet_classifier.pt`, `config.json` |
| ResNet+LSTM | `models/classifier/` | `best.pt`, `last.pt`, `config.json` |
| KeypointLSTM | `models/classifier/` | `best.pt`, `last.pt`, `config.json` |

`config.json` for KeypointLSTM includes `sequence_length` so the demo can size the per-pedestrian frame buffer automatically.

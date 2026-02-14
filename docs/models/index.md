# Model Architectures

PedSense-AI evaluates three architectures for pedestrian crossing intent prediction, each with different speed/accuracy tradeoffs.

## Comparison

| Model | Type | Input | Classes | Temporal | Best For |
|-------|------|-------|---------|----------|----------|
| [YOLO26](yolo.md) | End-to-end detector | Full frames | 2 (crossing, not-crossing) | No | Real-time deployment |
| [ResNet+LSTM](resnet-lstm.md) | Temporal classifier | Pedestrian crop sequences | 2 | Yes (16 frames) | Accuracy-critical scenarios |
| [Hybrid](hybrid.md) | YOLO proposals + ResNet | Full frames → crops | 2 | No | Balanced speed/accuracy |

## How They Differ

### Detection vs. Classification

- **YOLO26** does both detection (finding pedestrians) and classification (crossing intent) in a single pass.
- **ResNet+LSTM** only does classification — it needs pre-cropped pedestrian regions.
- **Hybrid** splits the work: YOLO detects, ResNet classifies.

### Temporal Context

- **YOLO26** and **Hybrid** operate on single frames — no temporal reasoning.
- **ResNet+LSTM** analyzes 16 consecutive frames (~0.5 seconds at 30 FPS), capturing gait and posture changes that indicate crossing intent.

## Output Format

All trained models are saved to `models/custom/{name}_{datetime}/`:

| Model | Output Files |
|-------|-------------|
| YOLO26 | `weights/best.pt`, `weights/last.pt`, training metrics |
| ResNet+LSTM | `best.pt`, `last.pt`, `config.json` |
| Hybrid | `yolo_detector.pt`, `resnet_classifier.pt`, `config.json` |

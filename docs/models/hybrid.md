# Hybrid (YOLO + ResNet)

Two-stage pipeline where YOLO26 acts as the **Proposal Engine** (detecting pedestrians) and ResNet-50 acts as the **Decision Engine** (classifying crossing intent).

## Architecture

```
Input: Full frame (1920x1080)
         │
    YOLO26 Detector (1 class: pedestrian)
         │
    Detected bounding boxes
         │
    Crop each pedestrian → Resize to 224x224
         │
    ResNet-50 Classifier (2 classes)
         │
    Output: crossing / not-crossing per pedestrian
```

### Key Differences from Standalone Models

| Aspect | Standalone YOLO | Standalone ResNet+LSTM | Hybrid |
|--------|----------------|----------------------|--------|
| YOLO classes | 2 (crossing, not-crossing) | N/A | 1 (pedestrian) |
| ResNet input | N/A | 16-frame sequences | Single frame |
| Temporal modeling | No | Yes (LSTM) | No |
| Detection | Built-in | External | YOLO stage |

## Training

Training happens in two stages:

### Stage 1: YOLO Pedestrian Detector

A YOLO26 model is trained for single-class pedestrian detection (all track types: `pedestrian`, `ped`, `people`).

```bash
# Train both stages
uv run pedsense train -m hybrid -n my_hybrid -e 30

# Or skip stage 1 by providing an existing YOLO model
uv run pedsense train -m hybrid -n my_hybrid --yolo-model path/to/best.pt
```

### Stage 2: ResNet Intent Classifier

Ground-truth bounding boxes are used to crop pedestrian regions from training frames. A ResNet-50 classifier (without LSTM) is trained on these crops.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 30 | ResNet classifier training epochs |
| `batch_size` | 16 | Batch size |
| `yolo_epochs` | 50 | YOLO detector training epochs (stage 1) |

## Output

Saved to `models/custom/{name}_{datetime}/`:

```
yolo_detector.pt        # YOLO pedestrian detector weights
resnet_classifier.pt    # ResNet intent classifier weights
config.json             # Configuration and results
```

### config.json

```json
{
  "model_type": "hybrid",
  "num_classes": 2,
  "class_names": ["not-crossing", "crossing"],
  "yolo_detector": "yolo_detector.pt",
  "resnet_classifier": "resnet_classifier.pt",
  "best_val_acc": 0.82
}
```

## Inference

The Gradio demo supports hybrid models natively:

```bash
uv run pedsense demo -m my_hybrid_20260214
```

At inference time:

1. YOLO detects all pedestrians in the frame
2. Each detection is cropped and resized to 224x224
3. ResNet classifies each crop as `crossing` or `not-crossing`
4. Results are drawn on the frame with colored bounding boxes

## When to Use Hybrid

- When you want YOLO-speed detection with a more specialized classifier
- When the 2-class YOLO struggles to distinguish intent (detection and classification compete for model capacity)
- When you want to swap out the classifier without retraining the detector

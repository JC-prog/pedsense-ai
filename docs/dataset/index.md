# Dataset Pipeline

PedSense-AI uses the **JAAD (Joint Attention in Autonomous Driving)** dataset for training pedestrian crossing intent models.

## JAAD at a Glance

| Attribute | Value |
|-----------|-------|
| Videos | 346 clips (~5-15 seconds each) |
| Resolution | 1920 x 1080 |
| Total frames | ~207,600 (~600 per video) |
| Annotated pedestrians | 2,793 unique tracks |
| Bounding boxes | 390,000+ |
| Annotation format | CVAT XML |

## Data Flow

```
data/raw/clips/*.mp4          Raw video files
data/raw/annotations/*.xml    CVAT XML annotations
        │
    pedsense preprocess frames
        │
data/raw/frames/              Extracted JPEG frames
        │
    ┌───┴───┐
    │       │
 preprocess  preprocess
   yolo      resnet
    │       │
data/processed/yolo/     data/processed/resnet/
 (images + labels)        (sequences + labels.csv)
```

## Pipeline Sections

- [Raw JAAD Format](raw-format.md) — Structure of the raw dataset
- [Preprocessing](preprocessing.md) — Frame extraction and annotation parsing
- [YOLO Format](yolo-format.md) — YOLO dataset output
- [ResNet Format](resnet-format.md) — ResNet+LSTM sequence output

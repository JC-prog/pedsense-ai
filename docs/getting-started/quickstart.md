# Quickstart

This guide walks through the full pipeline: setup, preprocessing, training, and running the demo.

## 1. Setup

```bash
uv run pedsense setup
```

Ensure your JAAD data is in `data/raw/clips/` and `data/raw/annotations/`.

## 2. Extract Frames

Extract video frames to individual images:

```bash
# Extract all videos (~207K frames, ~40GB)
uv run pedsense preprocess frames

# Or extract a single video for testing
uv run pedsense preprocess frames --video video_0001
```

Frames are saved to `data/raw/frames/{video_id}/frame_000000.jpg`.

## 3. Prepare Datasets

Convert annotations to model-specific formats:

```bash
# Prepare YOLO dataset (2-class: crossing / not-crossing)
uv run pedsense preprocess yolo

# Prepare ResNet+LSTM sequences (16-frame sliding window)
uv run pedsense preprocess resnet

# Or do everything at once
uv run pedsense preprocess all
```

## 4. Train a Model

### YOLO26

```bash
uv run pedsense train --model yolo --name my_yolo --epochs 50 --batch-size 16
```

### ResNet+LSTM

```bash
uv run pedsense train --model resnet-lstm --name my_resnet --epochs 30 --batch-size 8
```

### Hybrid (YOLO + ResNet)

```bash
uv run pedsense train --model hybrid --name my_hybrid --epochs 30

# Or reuse an existing YOLO detector
uv run pedsense train --model hybrid --name my_hybrid --yolo-model models/custom/my_yolo_20260214_120000/weights/best.pt
```

Trained models are saved to `models/custom/{name}_{datetime}/`.

## 5. Run the Demo

```bash
# Launch with the most recent model
uv run pedsense demo

# Or specify a model
uv run pedsense demo --model my_yolo_20260214_120000
```

Open [http://localhost:7860](http://localhost:7860) to upload a video and see crossing intent predictions with annotated bounding boxes.

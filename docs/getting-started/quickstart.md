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
uv run pedsense train --model hybrid --name my_hybrid --yolo-model models/detector/my_yolo_20260214_120000/weights/best.pt
```

Detection models (yolo, hybrid) are saved to `models/detector/{name}_{datetime}/`. Intent classifiers (keypoint-lstm, resnet-lstm) are saved to `models/classifier/{name}_{datetime}/`.

### KeypointLSTM (Skeleton-Based)

The 2-stage intent pipeline — YOLO-Pose extracts keypoints, a lightweight LSTM classifies crossing intent from skeleton sequences.

```bash
# Step 1: Extract pose labels (generates YOLO-Pose training data)
uv run pedsense preprocess pose

# Step 2: Fine-tune a YOLO-Pose model
uv run pedsense train -m yolo-pose -n my_pose_model -e 10 -b 16 --yolo-variant yolo11m-pose

# Step 3: Build keypoint sequences (1s horizon, default output dir)
uv run pedsense preprocess keypoints --sequence-length 5 --prediction-horizon 1.0

# Step 4: Train KeypointLSTM
uv run pedsense train -m keypoint-lstm -n my_lstm -e 50 -b 16
```

**Training at multiple prediction horizons** — run `preprocess keypoints` with `--keypoints-dir` to keep each horizon separate, then train a model per horizon:

```bash
uv run pedsense preprocess keypoints --prediction-horizon 3.0 --keypoints-dir data/processed/keypoints_3s
uv run pedsense preprocess keypoints --prediction-horizon 5.0 --keypoints-dir data/processed/keypoints_5s

uv run pedsense train -m keypoint-lstm -n my_lstm_3s -e 50 --keypoints-dir data/processed/keypoints_3s
uv run pedsense train -m keypoint-lstm -n my_lstm_5s -e 50 --keypoints-dir data/processed/keypoints_5s
```

## 5. Run the Demo

```bash
# Launch with the most recent model
uv run pedsense demo

# Or specify a model
uv run pedsense demo --model my_yolo_20260214_120000
```

Open [http://localhost:7860](http://localhost:7860) to upload a video and see crossing intent predictions with annotated bounding boxes.

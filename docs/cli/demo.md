# pedsense demo

Launch the Gradio web interface for inference.

## Synopsis

```bash
uv run pedsense demo [OPTIONS]
```

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model TEXT` | `-m` | Latest model | Model directory name (pre-selects in dropdown) |
| `--port INT` | `-p` | `7860` | Gradio server port |

## Description

Launches a web interface with two inference pipelines:

**Detection Only** — classify pedestrian intent frame-by-frame using a single model:

1. Upload a video
2. Select a detection model from the dropdown
3. Adjust the confidence threshold
4. Run inference to get an annotated video

**2-Stage Intent (Pose + LSTM)** — detect pedestrians with YOLO-Pose, then classify crossing intent using a trained KeypointLSTM:

1. Upload a video
2. Switch the pipeline radio to **2-Stage Intent (Pose + LSTM)**
3. Select a pose detector from the first dropdown (filtered to YOLO-Pose models)
4. Select a KeypointLSTM intent model from the second dropdown
5. Run inference

## Visual Output

### Detection Only

For crossing intent models (YOLO, Hybrid):

- **Green** box — Not crossing
- **Red** box — Crossing

For YOLO-Pose (skeleton-only):

- **Cyan** box — pedestrian detection
- **Green** lines — COCO skeleton connections
- **Yellow** dots — 17 body keypoints

### 2-Stage Intent

- **Yellow** box — buffering (`buffering N/T` label shown until T frames are collected per track)
- **Green** box — Not crossing (once classified)
- **Red** box — Crossing (once classified)
- **Green** skeleton lines — drawn over each tracked pedestrian
- `ID:{track_id} {label} {confidence}` label on each bounding box

## Examples

```bash
# Use the most recently trained model
uv run pedsense demo

# Pre-select a detection model
uv run pedsense demo -m my_yolo_20260214_153000

# Custom port
uv run pedsense demo -p 8080
```

## Supported Model Types

| Model Type | Pipeline | Inference Approach | Output |
|------------|----------|-------------------|--------|
| YOLO | Detection Only | Frame-by-frame detection + classification | Colored boxes by intent |
| YOLO-Pose | Detection Only | Keypoint detection with skeleton overlay | Cyan boxes + skeleton |
| Hybrid | Detection Only | YOLO detects, ResNet classifies each crop | Colored boxes by intent |
| YOLO-Pose + KeypointLSTM | 2-Stage Intent | YOLO-Pose tracks pedestrians, LSTM classifies sequences | Colored boxes with track IDs |

!!! note
    ResNet+LSTM models are not supported standalone in the demo — they require a separate pedestrian detector. Use the Hybrid model type for ResNet-based inference.

## How 2-Stage Intent Works

The KeypointLSTM pipeline runs online inference using YOLO's byte-tracker to assign stable pedestrian IDs across frames:

1. Each frame is processed by the pose detector (`model.track()`)
2. For each tracked pedestrian, keypoints are normalized relative to the bounding box center and height — the same normalization used during preprocessing
3. Normalized keypoints are pushed into a per-pedestrian deque of length `T` (read from the model's `config.json`)
4. Once the deque is full, the sequence `(T, 34)` is passed to the KeypointLSTM for classification
5. The prediction is displayed on the bounding box and updated every frame once the buffer is full

## Model Discovery

The demo automatically discovers trained models in `models/detector/` and `models/classifier/`. A model is listed if it has:

- A `config.json` file (ResNet+LSTM, Hybrid, KeypointLSTM), **or**
- Any `.pt` weights in a `weights/` subdirectory (YOLO, YOLO-Pose)

For the **2-Stage Intent** pipeline, the pose detector dropdown is filtered to models detected as `yolo`, `yolo-pose`, or `hybrid`. The intent model dropdown shows only `keypoint-lstm` models.

For YOLO models, weights are loaded with fallback priority: `best.pt` > `last.pt` > any other `.pt` file.

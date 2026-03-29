# pedsense demo

Launch the Gradio web interface for inference.

## Synopsis

```bash
uv run pedsense demo [OPTIONS]
```

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model TEXT` | `-m` | Latest model | Model directory name to pre-select in the detector dropdown |
| `--port INT` | `-p` | `7860` | Gradio server port |

## Modes

The demo has two modes, selected via a radio button:

### Detector

Runs a single model on each frame. Use this for:

- Visualising a trained YOLO 2-class model (crossing / not-crossing classification per frame — action detection)
- Visualising a YOLO-Pose model (skeleton overlay, no intent classification)
- Hybrid model (YOLO detects, ResNet classifies each crop)

**Controls:**

| Control | Description |
|---------|-------------|
| Detection Model | Any model from `models/detector/` or `models/detector-pose/` |
| Confidence Threshold | Minimum detection confidence (default 0.50) |

### Predictor

Two-stage pipeline: a YOLO-Pose model extracts skeleton keypoints per tracked pedestrian, and a KeypointLSTM classifies crossing intent over a sequence of frames.

Optionally, a second YOLO model can run in parallel as an **action detector** — classifying the pedestrian's *current* state per frame independently of the LSTM.

| Model role | What it answers |
|---|---|
| LSTM intent model | *"Will this pedestrian cross in the next N seconds?"* |
| Action detector (optional) | *"Is this pedestrian crossing right now?"* |

Together they cover the full timeline: the intent model fires before crossing, the action detector fires during it.

**Controls:**

| Control | Description |
|---------|-------------|
| Pose Detector | YOLO-Pose model from `models/detector-pose/` (required) |
| Intent Model | KeypointLSTM from `models/classifier-lstm/` (required) |
| Action Detector | YOLO 2-class detector from `models/detector/` (optional) |
| Confidence Threshold | Detection confidence for the pose model (default 0.50) |
| Prediction Smoothing | Rolling average window for LSTM outputs, 1–30 frames (default 10). Higher = smoother label, slower to react. Set to 1 to see raw per-frame predictions |

## Refresh Models

Click **Refresh Models** at any time while the demo is running to re-scan all model directories and update the dropdowns — no restart needed. Use this after copying a newly trained model into its folder.

## Visual Output

### Detector mode

| Box color | Meaning |
|-----------|---------|
| Green | Not crossing (YOLO 2-class) |
| Red | Crossing (YOLO 2-class) |
| Cyan | Pedestrian detected (YOLO-Pose / 1-class detector) |

YOLO-Pose additionally draws:

- Green skeleton lines (COCO connections)
- Amber dots (17 body keypoints)

### Predictor mode

Each tracked pedestrian gets two annotations:

**Top label** (LSTM intent — colored box):

| Box color | Label | Meaning |
|-----------|-------|---------|
| Amber | `[N/T]` | Buffering — accumulating frames before first prediction |
| Green | `not crossing  NN%` | Model predicts no crossing intent |
| Red | `crossing  NN%` | Model predicts crossing intent |

The confidence shown is the smoothed crossing probability (rolling mean over the last `smooth_window` LSTM outputs).

**Bottom label** (action detector — darker tag below box, only shown if an action detector is selected):

| Tag color | Label | Meaning |
|-----------|-------|---------|
| Dark green | `action: not crossing` | Pedestrian is not currently crossing |
| Dark red | `action: crossing` | Pedestrian is currently crossing |

The action label is matched to each pose-tracked pedestrian by IoU (threshold 0.3). If no YOLO detection overlaps sufficiently, no action tag is shown.

!!! tip "Intent vs. action"
    The LSTM intent label and the action detector label are complementary. A pedestrian approaching a crossing will show **intent: crossing** (LSTM fires early) while **action: not crossing** (they haven't stepped off the kerb yet). Once they begin crossing, the action detector confirms it even if LSTM uncertainty rises.

!!! warning
    The Pose Detector must be a YOLO-Pose model (`train -m yolo-pose`). Selecting a standard YOLO detection model will raise a `"does not output keypoints"` error.

## Examples

```bash
# Launch on default port
uv run pedsense demo

# Pre-select a specific detector
uv run pedsense demo -m my_detector_20260329_120000

# Custom port
uv run pedsense demo -p 8080
```

## Model Discovery

The demo scans four directories automatically:

| Directory | Models listed in |
|-----------|-----------------|
| `models/detector/` | Detector dropdown (Detector mode) |
| `models/detector-pose/` | Detector dropdown + Pose Detector dropdown (Predictor mode) |
| `models/classifier-lstm/` | Intent Model dropdown (Predictor mode) |
| `models/classifier-stgcn/` | Intent Model dropdown (Predictor mode, future) |

A model is listed if it has a `weights/best.pt` (YOLO family) or a `config.json` + `best.pt` (LSTM/Hybrid).

## How the Predictor Pipeline Works

1. Each frame is passed to the pose detector via `model.track()` — YOLO's built-in tracker assigns stable pedestrian IDs across frames
2. For each pedestrian, keypoints are normalized relative to the detected bounding box center and height: `(kx - cx) / h`, `(ky - cy) / h`
3. Normalized keypoints (flattened to 34 values) are pushed into a per-pedestrian deque of length `T` (read from `config.json`)
4. Once full, the `(T, 34)` sequence is passed to the KeypointLSTM → crossing probability
5. The raw crossing probability is pushed into a second deque of length `smooth_window`; the rolling mean is used to determine the displayed label and color
6. If an action detector is loaded, it runs a separate forward pass on the same frame and results are IoU-matched to tracked boxes

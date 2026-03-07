# pedsense demo

Launch the Gradio web interface for inference.

## Synopsis

```bash
uv run pedsense demo [OPTIONS]
```

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model TEXT` | `-m` | Latest model | Model directory name or path |
| `--port INT` | `-p` | `7860` | Gradio server port |

## Description

Launches a web interface where you can:

1. Upload a video file
2. Select a trained model from the dropdown
3. Adjust the confidence threshold
4. Run inference to get an annotated video with bounding boxes

For crossing intent models (YOLO, Hybrid), bounding boxes are colored by prediction:

- **Green** — Not crossing
- **Red** — Crossing

For YOLO-Pose models, the overlay shows:

- **Cyan** bounding box — pedestrian detection
- **Green** lines — COCO skeleton connections
- **Yellow** dots — 17 body keypoints

## Examples

```bash
# Use the most recently trained model
uv run pedsense demo

# Specify a model
uv run pedsense demo -m experiment1_20260214_153000

# Custom port
uv run pedsense demo -p 8080
```

## Supported Model Types

| Model Type | Inference Approach | Output |
|------------|-------------------|--------|
| YOLO | Direct frame-by-frame detection + classification | Colored boxes by crossing intent |
| YOLO-Pose | Keypoint detection with skeleton overlay | Cyan boxes + green skeleton + yellow keypoints |
| Hybrid | YOLO detects pedestrians, ResNet classifies each crop | Colored boxes by crossing intent |

!!! note
    ResNet+LSTM models require a separate pedestrian detector and are not directly supported in the demo. Use a YOLO or Hybrid model instead.

## Model Discovery

The demo automatically discovers trained models in `models/custom/`. A model is listed if it has:

- A `config.json` file (ResNet+LSTM, Hybrid), **or**
- Any `.pt` weights in a `weights/` subdirectory (YOLO, YOLO-Pose)

For YOLO models, weights are loaded with fallback priority: `best.pt` > `last.pt` > any other `.pt` file. This means renamed weight files are still discovered correctly.

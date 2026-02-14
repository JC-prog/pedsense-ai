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

Bounding boxes are colored by crossing intent:

- **Green** — Not crossing
- **Red** — Crossing

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

| Model Type | Inference Approach |
|------------|-------------------|
| YOLO | Direct frame-by-frame detection + classification |
| Hybrid | YOLO detects pedestrians, ResNet classifies each crop |

!!! note
    ResNet+LSTM models require a separate pedestrian detector and are not directly supported in the demo. Use a YOLO or Hybrid model instead.

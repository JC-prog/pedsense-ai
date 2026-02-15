# pedsense.demo

`src/pedsense/demo.py`

Gradio web interface for running inference with trained models.

## Functions

### `launch_demo(model_path: str | None = None, port: int = 7860) -> None`

Launch the Gradio web interface.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str \| None` | `None` | Model directory name. If `None`, uses latest model. |
| `port` | `int` | `7860` | Gradio server port |

**Interface components:**

- Video upload input
- Model selector dropdown (populated from `models/custom/`)
- Confidence threshold slider (0.1 - 1.0)
- Annotated output video
- Detection statistics (JSON)

### `run_inference(video_path: str, model_name: str, confidence: float) -> tuple[str | None, dict]`

Run model inference on a video file.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `video_path` | `str` | Path to input video |
| `model_name` | `str` | Model directory name in `models/custom/` |
| `confidence` | `float` | Confidence threshold for detections |

**Returns:** Tuple of (output video path, statistics dict)

**Statistics dict:**

```python
{
    "total_detections": 150,
    "crossing": 42,
    "not_crossing": 108
}
```

## Model Type Detection

The demo auto-detects the model type from `config.json` in the model directory:

| `model_type` | Inference Method |
|-------------|-----------------|
| `"yolo"` | Direct YOLO frame-by-frame detection + classification |
| `"hybrid"` | YOLO detects pedestrians, ResNet classifies each crop |
| `"resnet-lstm"` | Not supported in demo (needs external detector) |

## Helper Functions

### `_find_yolo_weights(model_dir: Path) -> Path | None`

Finds the best available YOLO weights file in a model's `weights/` directory. Uses a fallback priority:

1. `best.pt` (preferred)
2. `last.pt`
3. Any other `.pt` file

Returns `None` if no weights are found.

### `_list_available_models() -> list[str]`

Lists all model directories in `models/custom/` that contain either `config.json` or any `.pt` weights in a `weights/` subdirectory. Sorted by name (most recent first).

### `_get_latest_model() -> str | None`

Returns the first model from `_list_available_models()`, or `None` if no models exist.

### `_detect_model_type(model_dir: Path) -> str`

Reads `config.json` to determine model type. Falls back to checking for YOLO weight files via `_find_yolo_weights()`.

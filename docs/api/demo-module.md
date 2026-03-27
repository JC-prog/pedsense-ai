# pedsense.demo

`src/pedsense/demo.py`

Gradio web interface for running inference with trained models.

## Functions

### `launch_demo(model_path: str | None = None, port: int = 7860) -> None`

Launch the Gradio web interface.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str \| None` | `None` | Model directory name to pre-select in the dropdown. If `None`, defaults to the first available detection model. |
| `port` | `int` | `7860` | Gradio server port |

**Interface components:**

- Video upload input
- Pipeline selector radio: `Detection Only` | `2-Stage Intent (Pose + LSTM)`
- Detection/Pose model dropdown (all models for Detection Only; filtered to detection models for 2-Stage)
- Intent model dropdown (KeypointLSTM models only; visible only in 2-Stage pipeline)
- Confidence threshold slider (0.1–1.0)
- Annotated output video
- Statistics (JSON)

---

### `run_inference(video_path, pipeline, detection_model_name, intent_model_name, confidence) -> tuple[str | None, dict]`

Route inference to the appropriate runner based on pipeline selection.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `video_path` | `str` | Path to input video |
| `pipeline` | `str` | `"Detection Only"` or `"2-Stage Intent (Pose + LSTM)"` |
| `detection_model_name` | `str` | Model directory name in `models/detector/` |
| `intent_model_name` | `str \| None` | KeypointLSTM model directory name (required for 2-Stage) |
| `confidence` | `float` | Confidence threshold for detections |

**Returns:** `(output_video_path, stats_dict)`. On error, returns `(None, {"error": "..."})`.

**Statistics dict — Detection Only (YOLO / Hybrid):**
```python
{"total_detections": 150, "crossing": 42, "not_crossing": 108}
```

**Statistics dict — Detection Only (YOLO-Pose):**
```python
{"total_pedestrians": 150, "model_type": "yolo-pose"}
```

**Statistics dict — 2-Stage Intent (Pose + LSTM):**
```python
{
    "model_type": "keypoint-lstm",
    "sequence_length": 16,
    "pedestrians_tracked": 4,
    "crossing": 18,
    "not_crossing": 76,
}
```

---

## Model Type Detection

The demo auto-detects model type by checking, in order:

1. `args.yaml` — if the base model file contains `"pose"` in its name → `"yolo-pose"`
2. `config.json` — reads `model_type` field → `"hybrid"`, `"resnet-lstm"`, or `"keypoint-lstm"`
3. Weights directory presence → `"yolo"` (fallback)

| `model_type` | Pipeline | Inference Method |
|-------------|----------|-----------------|
| `"yolo"` | Detection Only | Frame-by-frame YOLO detection + classification |
| `"yolo-pose"` | Detection Only | YOLO-Pose skeleton visualization |
| `"hybrid"` | Detection Only | YOLO detects, ResNet classifies each crop |
| `"resnet-lstm"` | — | Not supported standalone (returns error) |
| `"keypoint-lstm"` | 2-Stage Intent | Requires pose detector — use 2-Stage pipeline |

---

## Inference Runners

### `_run_yolo_inference(video_path, model_dir, confidence)`

Frame-by-frame YOLO detection. Draws colored bounding boxes (green = not-crossing, red = crossing).

### `_run_yolo_pose_inference(video_path, model_dir, confidence)`

YOLO-Pose skeleton visualization. Draws cyan bounding boxes, green COCO skeleton connections, and yellow keypoint dots. No intent classification.

### `_run_hybrid_inference(video_path, model_dir, confidence)`

Two-stage per-frame inference: YOLO detects pedestrian bounding boxes, ResNet-50 classifies each cropped region independently.

### `_run_keypoint_lstm_inference(video_path, pose_model_dir, lstm_model_dir, confidence)`

Online 2-stage inference using YOLO tracking + KeypointLSTM classification.

**Algorithm:**

1. Run `pose_model.track(frame, persist=True)` each frame — byte-tracker assigns stable integer `track_id` values across frames
2. On the first frame with detections, validate that the model outputs keypoints (`r.keypoints is not None`). If not, raises `ValueError` with a message directing the user to select a YOLO-Pose model
3. For each tracked pedestrian, normalize the 17 keypoints relative to the bounding box center and height (identical to preprocessing normalization)
4. A per-track buffer (`deque(maxlen=T)`) is created **only** when valid keypoints are first seen for that track — pedestrians with no keypoint output never enter the buffer
5. Once the deque reaches length `T`, stack it into `(T, 34)` and run the KeypointLSTM
6. Display prediction + confidence on the bounding box; prediction is updated every subsequent frame as new keypoints are pushed in

**Bounding box colors:**

- **Yellow** — buffering (`N/T` frames collected, valid keypoints received)
- **Yellow** — `"no keypoints"` label — pedestrian detected but pose model returned no keypoints for that track
- **Green** — not-crossing (classified)
- **Red** — crossing (classified)

---

## Helper Functions

### `_resolve_model_dir(name: str) -> Path | None`

Searches `models/detector/` then `models/classifier/` for a directory matching `name`. Returns the first match, or `None` if not found. Used by `run_inference` so model names from either directory work without hardcoding the base path.

### `_find_yolo_weights(model_dir: Path) -> Path | None`

Finds the best available YOLO weights file in a model's `weights/` directory. Fallback priority: `best.pt` > `last.pt` > any `.pt` file. Returns `None` if no weights are found.

### `_list_models_by_type() -> dict[str, list[str]]`

Returns model names grouped by type:

```python
{
    "detection": ["pose_model_20260301_120000", "yolo_20260228_093000"],
    "keypoint-lstm": ["keypoint-lstm_20260302_140000"],
    "all": [...],
}
```

`"detection"` includes `yolo`, `yolo-pose`, and `hybrid` models.

### `_load_keypoint_lstm_model(model_dir, device) -> tuple[KeypointLSTM, int, int]`

Loads a KeypointLSTM model from `best.pt` and reads architecture parameters from `config.json`.

Returns `(model, input_size, sequence_length)`.

### `_detect_model_type(model_dir: Path) -> str`

Detects model type from `args.yaml` (pose check), `config.json`, or weights directory. Returns one of `"yolo"`, `"yolo-pose"`, `"hybrid"`, `"resnet-lstm"`, or `"keypoint-lstm"`.

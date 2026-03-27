# pedsense preprocess

Extract frames and prepare datasets from raw JAAD data.

## Synopsis

```bash
uv run pedsense preprocess [STEP] [OPTIONS]
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `STEP` | `all` | Processing step: `frames`, `yolo`, `resnet`, `pose`, `keypoints`, or `all` |

## Options

| Option | Short | Default | Applies to | Description |
|--------|-------|---------|------------|-------------|
| `--video TEXT` | `-v` | — | all | Process a single video (e.g., `video_0001`). Default: all videos |
| `--fps FLOAT` | — | — | frames | Target FPS for frame extraction (e.g. `10`, `15`). Default: native FPS |
| `--attribute TEXT` | `-a` | `cross` | yolo, resnet | Behavioral attribute for pedestrian classification. Run `pedsense attributes` to see options |
| `--track-labels TEXT` | `-t` | — | yolo, resnet | Track label types to include. Repeat for multiple. Default: `pedestrian` only. Run `pedsense attributes` to see options |
| `--pose-variant TEXT` | — | `yolo11n-pose` | pose, keypoints | YOLO-Pose model: `yolo11n-pose`, `yolo11s-pose`, `yolo11m-pose` |
| `--conf FLOAT` | — | `0.25` | pose, keypoints | Detection confidence threshold |
| `--iou-threshold FLOAT` | — | `0.3` | keypoints | Minimum IoU to match a YOLO-Pose detection to a JAAD pedestrian track |
| `--sequence-length INT` | — | `16` | keypoints | Frames per keypoint sequence window |
| `--sequence-stride INT` | — | `8` | keypoints | Step between consecutive windows (in extracted frames) |
| `--prediction-horizon FLOAT` | — | `1.0` | keypoints | Seconds before `crossing_point` that windows must end by |

## Steps

### `frames` — Extract Video Frames

Extracts individual JPEG frames from MP4 video clips using OpenCV.

```bash
# All videos at native FPS (~207K frames)
uv run pedsense preprocess frames

# Single video
uv run pedsense preprocess frames -v video_0001

# Downsample to 10 FPS for smaller dataset
uv run pedsense preprocess frames --fps 10
```

**Output:**

- `data/raw/frames/{video_id}/frame_{N:06d}.jpg` — extracted frames
- `data/raw/frames/{video_id}/meta.json` — FPS metadata used by downstream steps

**`meta.json` schema:**
```json
{
  "native_fps": 29.97,
  "interval": 3,
  "extracted_fps": 9.99
}
```

!!! note
    Frame extraction is idempotent — videos with existing frame directories are skipped, including `meta.json` generation.

!!! warning
    When using `--fps`, original frame indices are preserved in filenames (e.g. `frame_000030.jpg` for frame 30 at native 30 FPS sampled to 1 FPS). This ensures annotation lookups remain valid for the ResNet+LSTM pipeline, but sequences referencing unsampled frames will be silently skipped during `preprocess resnet`.

### `yolo` — Convert to YOLO Format

Parses XML annotations and creates a YOLO-compatible dataset.

**Behavioral attribute classification (pedestrian tracks):**

```bash
# Default: classify pedestrians by crossing intent
uv run pedsense preprocess yolo

# Classify by gaze direction
uv run pedsense preprocess yolo --attribute look

# Include ped and people variants alongside pedestrian
uv run pedsense preprocess yolo -t pedestrian -t ped -t people
```

**Multi-class detection (pedestrians + environment objects):**

```bash
# Pedestrian crossing intent + traffic lights + crosswalks
uv run pedsense preprocess yolo -t pedestrian -t traffic_light -t crosswalk
# data.yaml: nc=4, names=[not-crossing, crossing, traffic_light, crosswalk]

# All track types with gaze classification
uv run pedsense preprocess yolo --attribute look -t pedestrian -t traffic_light -t crosswalk
# data.yaml: nc=4, names=[not-looking, looking, traffic_light, crosswalk]
```

**Class ID scheme:**

- Pedestrian-variant tracks (`pedestrian`, `ped`, `people`) → class IDs derived from `--attribute` values
- Non-pedestrian tracks (`traffic_light`, `crosswalk`) → appended class IDs after attribute classes

Output: `data/processed/yolo/` with `data.yaml`, `images/`, and `labels/` directories.

!!! important
    Requires frames to be extracted first. Run `preprocess frames` before `preprocess yolo`.

### `resnet` — Convert to ResNet+LSTM Format

Creates 16-frame pedestrian crop sequences for temporal classification.

```bash
# Default: classify by crossing intent (pedestrian tracks only)
uv run pedsense preprocess resnet

# Classify by gaze direction
uv run pedsense preprocess resnet --attribute look

# Include ped and people variants alongside pedestrian
uv run pedsense preprocess resnet -t pedestrian -t ped -t people
```

Output: `data/processed/resnet/` with `labels.csv` and `sequences/` directory.

!!! note
    Non-pedestrian track types (`traffic_light`, `crosswalk`) are always ignored for ResNet+LSTM — they have no behavioral attributes to classify on.

### `pose` — Extract Pose Keypoints

Runs a pretrained YOLO-Pose model on extracted frames to detect pedestrians and extract **17 COCO body keypoints** per detection. Saves YOLO pose-format labels for downstream training or analysis.

**Does not use JAAD annotations** — keypoints are inferred by the pose model directly from pixel data.

```bash
# Default (yolo11n-pose, conf=0.25)
uv run pedsense preprocess pose

# Larger model, higher confidence
uv run pedsense preprocess pose --pose-variant yolo11m-pose --conf 0.4

# Single video test
uv run pedsense preprocess pose --video video_0001
```

Output: `data/processed/pose/` with `data.yaml`, `images/`, and `labels/` directories.

**Label format** (per detection per line):
```
0 cx cy w h  kp1x kp1y 2  kp2x kp2y 2  ...  kp17x kp17y 2
```
All coordinates normalized 0–1. The trailing `2` is the YOLO visibility flag (visible).

**`data.yaml`:**
```yaml
nc: 1
names: [pedestrian]
kpt_shape: [17, 3]
```

!!! note
    YOLO-Pose models (`yolo11n-pose.pt`, etc.) are downloaded to `models/base/` on first run.

!!! important
    Requires frames to exist. Run `preprocess frames` before `preprocess pose`.

**Pose variant comparison:**

| Variant | Size | Speed | Accuracy |
|---------|------|-------|----------|
| `yolo11n-pose` | Nano (default) | Fastest | Good |
| `yolo11s-pose` | Small | Fast | Better |
| `yolo11m-pose` | Medium | Moderate | Best |

### `keypoints` — Build Keypoint Sequence Dataset

The full upstream pipeline for skeleton-based intent prediction. Runs YOLO-Pose on extracted frames, matches detections to JAAD pedestrian tracks via IoU, and builds normalized `(T, 17, 2)` keypoint sequences anchored to each pedestrian's `crossing_point`.

```bash
# Default (1s prediction horizon, T=16, yolo11n-pose)
uv run pedsense preprocess keypoints

# Recommended for 1fps extraction
uv run pedsense preprocess keypoints --sequence-length 5 --prediction-horizon 1.0

# Larger model, stricter matching, longer horizon
uv run pedsense preprocess keypoints --pose-variant yolo11s-pose --iou-threshold 0.4 --prediction-horizon 2.0

# Single video test
uv run pedsense preprocess keypoints --video video_0001
```

**Output:** `data/processed/keypoints/sequences/{train,val}/*.npy` and `labels.csv`

**Key design decisions:**

- Windows end **before** `crossing_point` by at least `--prediction-horizon` seconds — prevents the model from seeing the crossing in progress
- Keypoints normalized relative to JAAD bbox center and height — view- and scale-invariant
- Frames with `occlusion == "full"` are rejected; any window containing one is dropped
- FPS is read from `meta.json` (written by `preprocess frames`) so the horizon is correct for any extraction rate

!!! important
    Requires frames to be extracted first. Run `preprocess frames` before `preprocess keypoints`.

!!! tip
    When extracting at low FPS (e.g. `--fps 1`), reduce `--sequence-length` proportionally. At 1fps, `--sequence-length 5` gives 5 seconds of observation — sufficient for gait/posture context.

### `all` — Full Pipeline

Runs `frames`, `yolo`, and `resnet` sequentially. Does **not** include `pose` or `keypoints` (run those separately after).

```bash
# Default (cross, pedestrian only)
uv run pedsense preprocess all

# Full pipeline including ped/people variants
uv run pedsense preprocess all -t pedestrian -t ped -t people

# Multi-class YOLO + ResNet with action attribute
uv run pedsense preprocess all --attribute action -t pedestrian -t traffic_light
```

## Examples

```bash
# List all available attributes and track labels
uv run pedsense attributes

# Standard preprocessing (crossing intent, pedestrian only)
uv run pedsense preprocess all

# Lightweight dataset at 10 FPS
uv run pedsense preprocess frames --fps 10

# Multi-class YOLO: detect pedestrian intent + traffic lights + crosswalks
uv run pedsense preprocess yolo -t pedestrian -t traffic_light -t crosswalk

# Broader pedestrian coverage (include all variants)
uv run pedsense preprocess yolo -t pedestrian -t ped -t people
uv run pedsense preprocess resnet -t pedestrian -t ped -t people

# Error: unknown track label
uv run pedsense preprocess yolo -t vehicle
# → Unknown track labels: ['vehicle']. Run 'pedsense attributes' to see options.
```

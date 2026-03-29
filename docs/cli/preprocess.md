# pedsense preprocess

Extract frames and prepare datasets from raw JAAD data.

## Synopsis

```bash
uv run pedsense preprocess [STEP] [OPTIONS]
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `STEP` | `all` | Processing step: `frames`, `yolo`, `resnet`, `pose`, `keypoints`, `dataset`, or `all` |

## Options

### Shared

| Option | Short | Default | Applies to | Description |
|--------|-------|---------|------------|-------------|
| `--video TEXT` | `-v` | — | all | Process a single video (e.g., `video_0001`). Default: all videos |
| `--fps FLOAT` | — | — | frames, dataset `--with-frames` | Target FPS for frame extraction. Default: native FPS |
| `--conf FLOAT` | — | `0.25` | pose, keypoints, dataset | Detection confidence threshold |
| `--iou-threshold FLOAT` | — | `0.3` | keypoints, dataset | Minimum IoU to match a YOLO-Pose detection to a JAAD track |

### `frames`, `yolo`, `resnet`, `pose`, `keypoints` (legacy steps)

| Option | Short | Default | Applies to | Description |
|--------|-------|---------|------------|-------------|
| `--attribute TEXT` | `-a` | `cross` | yolo, resnet | Behavioral attribute for pedestrian classification |
| `--track-labels TEXT` | `-t` | — | yolo, resnet | Track label types to include. Repeat for multiple |
| `--pose-variant TEXT` | — | `yolo11n-pose` | pose, keypoints | Variant name or path to `.pt` file |
| `--sequence-length INT` | — | `16` | keypoints | Frames per sequence window |
| `--sequence-stride INT` | — | `8` | keypoints | Step between consecutive windows |
| `--prediction-horizon FLOAT` | — | `1.0` | keypoints | Seconds before `crossing_point` that windows must end by |
| `--keypoints-dir TEXT` | — | `data/processed/keypoints/` | keypoints | Root output directory |
| `--csv / --no-csv` | — | `--no-csv` | keypoints | Save as CSV rows instead of `.npy` files |

### `dataset` step

| Option | Default | Description |
|--------|---------|-------------|
| `--name TEXT` | *(required)* | Output folder name. Creates `data/processed/{name}/` |
| `--mode TEXT` | *(required)* | `pedestrian`, `keypoint`, or `crossing_keypoint` |
| `--split TEXT` | `70/15/15` | Train/test/val ratios as integers summing to 100 |
| `--pose-model TEXT` | `yolo11n-pose` | YOLO-Pose variant name or path to `.pt` file. Used by `keypoint` and `crossing_keypoint` modes |
| `--horizon INT` | `30` | Frames before `crossing_point` labeled as crossing. Used by `crossing_keypoint` mode |
| `--with-frames` | `False` | Extract frames from clips before building the dataset |

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

**Multiple prediction horizons** — use `--keypoints-dir` to keep each horizon in its own directory so datasets don't overwrite each other:

```bash
# 1-second horizon (default path)
uv run pedsense preprocess keypoints --prediction-horizon 1.0

# 3-second horizon
uv run pedsense preprocess keypoints \
    --prediction-horizon 3.0 \
    --keypoints-dir data/processed/keypoints_3s

# 5-second horizon
uv run pedsense preprocess keypoints \
    --prediction-horizon 5.0 \
    --keypoints-dir data/processed/keypoints_5s
```

**Using a fine-tuned YOLO-Pose model** — pass a path to your trained weights instead of a variant name. This is recommended when you have a model fine-tuned on JAAD, as it produces more accurate keypoint matches:

```bash
uv run pedsense preprocess keypoints \
    --pose-variant models/detector/my_pose_model_20260307_234738/weights/best.pt \
    --prediction-horizon 3.0 \
    --keypoints-dir data/processed/keypoints_3s \
    --sequence-length 5
```

**CSV output (adapter pattern)** — save as CSV first, convert to npy when ready to train. Useful for inspecting the data or sharing it without binary files:

```bash
# Save as CSV
uv run pedsense preprocess keypoints \
    --prediction-horizon 3.0 \
    --keypoints-dir data/processed/keypoints_3s \
    --csv

# Convert CSV → npy for training (no trainer code changes needed)
uv run pedsense convert-sequences data/processed/keypoints_3s
```

**Output (npy mode):** `<keypoints-dir>/sequences/{train,val}/*.npy` and `labels.csv`

**Output (csv mode):** `<keypoints-dir>/sequences_train.csv`, `sequences_val.csv`, and `labels.csv`

**Key design decisions:**

- Windows end **before** `crossing_point` by at least `--prediction-horizon` seconds — prevents the model from seeing the crossing in progress
- Keypoints normalized relative to JAAD bbox center and height — view- and scale-invariant
- Frames with `occlusion == "full"` are rejected; any window containing one is dropped
- FPS is read from `meta.json` (written by `preprocess frames`) so the horizon is correct for any extraction rate

!!! important
    Requires frames to be extracted first. Run `preprocess frames` before `preprocess keypoints`.

!!! tip
    When extracting at low FPS (e.g. `--fps 1`), reduce `--sequence-length` proportionally. At 1fps, `--sequence-length 5` gives 5 seconds of observation — sufficient for gait/posture context.

### `dataset` — Named, Inspectable Dataset

Creates a self-contained dataset folder under `data/processed/{name}/`. Unlike the legacy steps, each run produces a named directory with frame images copied alongside their annotations — making it easy to open a sequence CSV and immediately find the corresponding frame images.

**Three annotation modes:**

#### `pedestrian` — YOLO bounding-box labels

Extracts pedestrian bounding boxes from JAAD annotations and writes YOLO detection format labels (`.txt` files). Ready to use directly with any YOLO trainer.

```bash
uv run pedsense preprocess dataset \
    --name pedestrian_crossing \
    --mode pedestrian \
    --split 70/15/15
```

Output:
```
data/processed/pedestrian_crossing/
    frames/{split}/{video_id}/{video_id}_frame_{n:06d}.jpg
    labels/{split}/{video_id}/{video_id}_frame_{n:06d}.txt   ← YOLO: class cx cy w h
    data.yaml
```

#### `keypoint` — Per-frame skeleton CSVs (no label)

Runs YOLO-Pose on extracted frames, IoU-matches detections to JAAD pedestrian tracks, and writes normalized 17-joint keypoints. One CSV per video per split. No crossing label — useful as a raw skeleton dataset.

```bash
uv run pedsense preprocess dataset \
    --name keypoints_raw \
    --mode keypoint \
    --pose-model models/detector/my_pose_model_20260307_234738/weights/best.pt \
    --split 70/15/15
```

`annotations/{split}/{video_id}.csv` columns: `track_id, frame, k0, k1, ..., k33`

#### `crossing_keypoint` — Per-frame skeletons with crossing label

Same as `keypoint` but adds a `label` column. A frame is labeled `1` (crossing) if it falls within `--horizon` frames before the pedestrian's `crossing_point` annotation; `0` otherwise.

```bash
uv run pedsense preprocess dataset \
    --name crossing_kp_90f \
    --mode crossing_keypoint \
    --pose-model models/detector/my_pose_model_20260307_234738/weights/best.pt \
    --horizon 90 \
    --split 70/15/15
```

`annotations/{split}/{video_id}.csv` columns: `track_id, frame, label, k0, k1, ..., k33`

**Output (keypoint and crossing_keypoint):**
```
data/processed/{name}/
    frames/
        train/video_0001/video_0001_frame_000030.jpg
        test/video_0002/...
        val/video_0003/...
    annotations/
        train/video_0001.csv
        test/video_0002.csv
        val/video_0003.csv
    labels.csv    ← flat per-frame index across all videos and splits
```

**`labels.csv` columns (crossing_keypoint):**
```
video_id, track_id, frame, label, split, annotation_file, frame_file
```

**Also extract frames first:**

```bash
uv run pedsense preprocess dataset \
    --name my_exp \
    --mode crossing_keypoint \
    --with-frames \
    --fps 10 \
    --pose-model yolo11m-pose \
    --horizon 30 \
    --split 70/15/15
```

!!! note
    The `--horizon` is in **frames**, not seconds. At native 30 FPS, `--horizon 90` = 3 seconds before crossing. At 10 FPS, `--horizon 30` = 3 seconds before crossing.

!!! tip
    Use a different `--name` for each configuration to keep datasets separate and comparable:
    ```bash
    # 3-second horizon at native FPS
    uv run pedsense preprocess dataset --name crossing_kp_3s --mode crossing_keypoint --horizon 90
    # 5-second horizon
    uv run pedsense preprocess dataset --name crossing_kp_5s --mode crossing_keypoint --horizon 150
    ```

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

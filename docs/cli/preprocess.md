# pedsense preprocess

Extract frames and prepare datasets from raw JAAD data.

## Synopsis

```bash
uv run pedsense preprocess [STEP] [OPTIONS]
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `STEP` | `all` | Processing step: `frames`, `yolo`, `resnet`, or `all` |

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--video TEXT` | `-v` | — | Process a single video (e.g., `video_0001`). Default: all videos |
| `--fps FLOAT` | — | — | Target FPS for frame extraction (e.g. `10`, `15`). Default: native FPS |
| `--attribute TEXT` | `-a` | `cross` | Behavioral attribute for pedestrian classification. Run `pedsense attributes` to see options |
| `--track-labels TEXT` | `-t` | — | Track label types to include. Repeat for multiple. Default: `pedestrian` only. Run `pedsense attributes` to see options |

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

Output: `data/raw/frames/{video_id}/frame_{N:06d}.jpg`

!!! note
    Frame extraction is idempotent — videos with existing frame directories are skipped.

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

### `all` — Full Pipeline

Runs all three steps sequentially.

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

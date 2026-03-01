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
| `--attribute TEXT` | `-a` | `cross` | Annotation attribute to classify on. Run `pedsense attributes` to see options |

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

```bash
# Default: classify by crossing intent
uv run pedsense preprocess yolo

# Classify by gaze direction
uv run pedsense preprocess yolo --attribute look

# Classify by body movement
uv run pedsense preprocess yolo --attribute action
```

Output: `data/processed/yolo/` with `data.yaml`, `images/`, and `labels/` directories.

The `data.yaml` `names` field reflects the chosen attribute's class values. Run `pedsense attributes` to see all options.

!!! important
    Requires frames to be extracted first. Run `preprocess frames` before `preprocess yolo`.

### `resnet` — Convert to ResNet+LSTM Format

Creates 16-frame pedestrian crop sequences for temporal classification.

```bash
# Default: classify by crossing intent
uv run pedsense preprocess resnet

# Classify by gaze direction
uv run pedsense preprocess resnet --attribute look
```

Output: `data/processed/resnet/` with `labels.csv` and `sequences/` directory.

Labels in `labels.csv` reflect the chosen attribute's class values via majority vote over each 16-frame window.

### `all` — Full Pipeline

Runs all three steps sequentially with the same attribute applied to both YOLO and ResNet conversion.

```bash
# Default (cross)
uv run pedsense preprocess all

# Full pipeline for action classification
uv run pedsense preprocess all --attribute action
```

## Examples

```bash
# List available attributes before preprocessing
uv run pedsense attributes

# Standard preprocessing (crossing intent)
uv run pedsense preprocess all

# Lightweight dataset at 10 FPS for teammates
uv run pedsense preprocess frames --fps 10

# Train on looking vs not-looking instead of crossing
uv run pedsense preprocess yolo --attribute look
uv run pedsense preprocess resnet --attribute look
```

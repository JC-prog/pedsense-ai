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

| Option | Short | Description |
|--------|-------|-------------|
| `--video TEXT` | `-v` | Process a single video (e.g., `video_0001`). Default: all videos |

## Steps

### `frames` — Extract Video Frames

Extracts individual JPEG frames from MP4 video clips using OpenCV.

```bash
# All videos (~207K frames)
uv run pedsense preprocess frames

# Single video
uv run pedsense preprocess frames -v video_0001
```

Output: `data/raw/frames/{video_id}/frame_{N:06d}.jpg`

!!! note
    Frame extraction is idempotent — videos with existing frame directories are skipped.

### `yolo` — Convert to YOLO Format

Parses XML annotations and creates a YOLO-compatible dataset.

```bash
uv run pedsense preprocess yolo
```

Output: `data/processed/yolo/` with `data.yaml`, `images/`, and `labels/` directories.

!!! important
    Requires frames to be extracted first. Run `preprocess frames` before `preprocess yolo`.

### `resnet` — Convert to ResNet+LSTM Format

Creates 16-frame pedestrian crop sequences for temporal classification.

```bash
uv run pedsense preprocess resnet
```

Output: `data/processed/resnet/` with `labels.csv` and `sequences/` directory.

### `all` — Full Pipeline

Runs all three steps sequentially.

```bash
uv run pedsense preprocess all
```

# Preprocessing

The preprocessing pipeline extracts frames from raw videos and parses XML annotations into structured Python objects.

## Frame Extraction

```bash
uv run pedsense preprocess frames
```

### How It Works

1. Iterates over all `.mp4` files in `data/raw/clips/`
2. Opens each video with OpenCV's `VideoCapture`
3. Extracts every frame as a JPEG (quality 95)
4. Saves to `data/raw/frames/{video_id}/frame_{N:06d}.jpg`

### Output

```
data/raw/frames/
    video_0001/
        frame_000000.jpg
        frame_000001.jpg
        ...
        frame_000599.jpg
    video_0002/
        ...
```

### Disk Space

- ~207,600 frames total (346 videos x ~600 frames)
- ~200 KB per JPEG at quality 95
- **Total: ~40 GB**

!!! tip
    Use `--video video_0001` to extract a single video for testing before processing the full dataset.

### Idempotency

Frame extraction is idempotent. If a video's frame directory already exists and is non-empty, it is skipped. To re-extract, delete the existing frame directory first.

## Annotation Parsing

Annotations are parsed automatically during `preprocess yolo` and `preprocess resnet`. The parser (`pedsense.processing.annotations`) converts CVAT XML into Python dataclasses.

### Dataclasses

```python
@dataclass
class BoundingBox:
    frame: int
    xtl: float
    ytl: float
    xbr: float
    ybr: float
    occluded: bool
    outside: bool
    track_id: str
    action: str       # "standing" or "walking"
    cross: str        # "crossing" or "not-crossing"
    look: str         # "looking" or "not-looking"
    occlusion: str    # "none", "part", or "full"

@dataclass
class Track:
    label: str              # "pedestrian", "ped", or "people"
    boxes: list[BoundingBox]

@dataclass
class VideoAnnotation:
    video_id: str
    num_frames: int
    width: int
    height: int
    time_of_day: str    # "daytime", "nighttime"
    weather: str        # "cloudy", "clear", etc.
    location: str       # "plaza", "intersection", etc.
    tracks: list[Track]
```

### Filtering

- Boxes with `outside="1"` are discarded (pedestrian not visible)
- Empty tracks (no visible boxes) are excluded
- Only `label="pedestrian"` tracks are used for intent classification

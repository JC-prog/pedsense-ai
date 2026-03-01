# pedsense.processing

`src/pedsense/processing/`

Modules for parsing JAAD annotations, extracting video frames, and converting data to model-specific formats.

## pedsense.processing.annotations

`src/pedsense/processing/annotations.py`

### Constants

#### `ATTRIBUTE_LABELS`

```python
ATTRIBUTE_LABELS: dict[str, list[str]] = {
    "cross":     ["not-crossing", "crossing"],
    "action":    ["standing", "walking"],
    "look":      ["not-looking", "looking"],
    "occlusion": ["none", "part", "full"],
}
```

Maps each supported JAAD behavioral attribute to its ordered list of class values. The list order determines class IDs (index 0, 1, …) for both YOLO labels and ResNet+LSTM training targets.

Use `pedsense attributes` from the CLI to display this at runtime.

### Dataclasses

#### `BoundingBox`

```python
@dataclass
class BoundingBox:
    frame: int          # Frame number (0-indexed)
    xtl: float          # Top-left x (pixels)
    ytl: float          # Top-left y (pixels)
    xbr: float          # Bottom-right x (pixels)
    ybr: float          # Bottom-right y (pixels)
    occluded: bool      # Whether occluded
    outside: bool       # Whether outside the frame
    track_id: str       # Unique pedestrian ID
    action: str         # "standing" or "walking"
    cross: str          # "crossing" or "not-crossing"
    look: str           # "looking" or "not-looking"
    occlusion: str      # "none", "part", or "full"
```

#### `Track`

```python
@dataclass
class Track:
    label: str                      # "pedestrian", "ped", or "people"
    boxes: list[BoundingBox]        # Per-frame bounding boxes
```

#### `VideoAnnotation`

```python
@dataclass
class VideoAnnotation:
    video_id: str       # e.g., "video_0001"
    num_frames: int     # Total frame count
    width: int          # Video width (pixels)
    height: int         # Video height (pixels)
    time_of_day: str    # "daytime", "nighttime"
    weather: str        # "cloudy", "clear", etc.
    location: str       # "plaza", "intersection", etc.
    tracks: list[Track] # All annotated tracks
```

### Functions

#### `parse_annotation(xml_path: Path) -> VideoAnnotation`

Parse a single CVAT XML annotation file into a `VideoAnnotation` object.

- Filters out boxes with `outside="1"`
- Extracts all attributes from `<box>` elements
- Discards empty tracks

#### `load_all_annotations() -> dict[str, VideoAnnotation]`

Parse all XML files in `data/raw/annotations/`. Returns a dictionary mapping video IDs to their annotations.

Displays a Rich progress bar during parsing.

---

## pedsense.processing.frames

`src/pedsense/processing/frames.py`

### Functions

#### `extract_frames(video_id: str | None = None, fps: float | None = None) -> None`

Extract frames from MP4 videos using OpenCV.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_id` | `str \| None` | `None` | Process a single video. If `None`, processes all. |
| `fps` | `float \| None` | `None` | Target frames per second. If `None`, extracts every frame at native FPS. |

**Output:** `data/raw/frames/{video_id}/frame_{N:06d}.jpg`

**Behavior:**

- Saves frames as JPEG quality 95
- Skips videos with existing non-empty frame directories
- Shows Rich progress bar
- Raises `FileNotFoundError` if video file or clips directory is missing
- When `fps` is set, saves only every `round(native_fps / fps)`-th frame while **preserving original frame indices** in filenames — ensures annotation-based lookups remain valid

---

## pedsense.processing.yolo_format

`src/pedsense/processing/yolo_format.py`

### Functions

#### `convert_to_yolo(train_ratio: float = 0.8, attribute: str = "cross") -> Path`

Convert JAAD annotations to YOLO format.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_ratio` | `float` | `0.8` | Fraction of videos for training |
| `attribute` | `str` | `"cross"` | Annotation attribute to classify on. Must be a key in `ATTRIBUTE_LABELS`. |

**Returns:** Path to `data.yaml`

**Behavior:**

- Splits at video level (not frame level) to prevent data leakage
- Class names and IDs derived from `ATTRIBUTE_LABELS[attribute]`
- Copies images from `data/raw/frames/` to `data/processed/yolo/images/`
- Writes normalized YOLO labels to `.txt` files
- Generates `data.yaml` with `nc` and `names` matching the chosen attribute

---

## pedsense.processing.resnet_format

`src/pedsense/processing/resnet_format.py`

### Functions

#### `convert_to_resnet(sequence_length: int = 16, stride: int = 8, crop_size: tuple[int, int] = (224, 224), train_ratio: float = 0.8, attribute: str = "cross") -> Path`

Create cropped pedestrian sequences for ResNet+LSTM training.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sequence_length` | `int` | `16` | Frames per sequence |
| `stride` | `int` | `8` | Sliding window stride |
| `crop_size` | `tuple[int, int]` | `(224, 224)` | Output crop dimensions |
| `train_ratio` | `float` | `0.8` | Fraction of videos for training |
| `attribute` | `str` | `"cross"` | Annotation attribute to classify on. Must be a key in `ATTRIBUTE_LABELS`. |

**Returns:** Path to `labels.csv`

**Behavior:**

- Sliding window with configurable stride (default 50% overlap)
- Crops and resizes pedestrian regions to `crop_size`
- Labels each sequence by majority vote of `attribute` value across the window
- Skips occluded frames
- Same video-level split as YOLO converter

# YOLO Format

The YOLO dataset converter creates an Ultralytics-compatible dataset from JAAD annotations.

## Command

```bash
uv run pedsense preprocess yolo
```

## Output Structure

```
data/processed/yolo/
    data.yaml
    images/
        train/
            video_0001_frame_000000.jpg
            video_0001_frame_000042.jpg
            ...
        val/
            ...
    labels/
        train/
            video_0001_frame_000000.txt
            video_0001_frame_000042.txt
            ...
        val/
            ...
```

## data.yaml

```yaml
path: /absolute/path/to/data/processed/yolo
train: images/train
val: images/val
nc: 2
names:
  - not-crossing
  - crossing
```

## Label Format

Each `.txt` file contains one line per annotated pedestrian in the corresponding image:

```
class_id x_center y_center width height
```

All values are **normalized to [0, 1]** relative to image dimensions (1920 x 1080).

### Example

```
0 0.260417 0.730556 0.035417 0.109259
1 0.480208 0.620370 0.042708 0.133333
```

### Coordinate Conversion

From CVAT `(xtl, ytl, xbr, ybr)` pixel coordinates:

```
x_center = ((xtl + xbr) / 2) / image_width
y_center = ((ytl + ybr) / 2) / image_height
width    = (xbr - xtl) / image_width
height   = (ybr - ytl) / image_height
```

## Classes

| ID | Name | Description |
|----|------|-------------|
| 0 | `not-crossing` | Pedestrian with `cross="not-crossing"` |
| 1 | `crossing` | Pedestrian with `cross="crossing"` |

Only `label="pedestrian"` tracks are included (they have crossing intent annotations).

## Train/Val Split

- Split at the **video level** (not frame level) to prevent data leakage
- 80% of videos → train, 20% → val
- Fixed random seed (42) for reproducibility
- Consecutive frames from the same video never appear in both splits

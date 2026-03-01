# pedsense attributes

List all supported annotation attributes and their class values.

## Synopsis

```bash
uv run pedsense attributes
```

## Description

Prints each JAAD behavioral attribute alongside its ordered class values. The order of class values is significant — it determines the class ID mapping used when generating YOLO labels and ResNet+LSTM training data.

## Output

```
cross:     ['not-crossing', 'crossing']
action:    ['standing', 'walking']
look:      ['not-looking', 'looking']
occlusion: ['none', 'part', 'full']
```

## Attributes

| Attribute | Classes | Description |
|-----------|---------|-------------|
| `cross` | `not-crossing`, `crossing` | Whether the pedestrian is crossing the road (default) |
| `action` | `standing`, `walking` | Body movement state |
| `look` | `not-looking`, `looking` | Whether the pedestrian is looking towards the vehicle |
| `occlusion` | `none`, `part`, `full` | Degree of occlusion |

## Usage with preprocess

Pass the attribute name to `pedsense preprocess` with `--attribute`:

```bash
# Default — classify by crossing intent
uv run pedsense preprocess yolo

# Classify by gaze direction
uv run pedsense preprocess yolo --attribute look
uv run pedsense preprocess resnet --attribute look

# Classify by body movement
uv run pedsense preprocess yolo --attribute action
uv run pedsense preprocess resnet --attribute action
```

!!! note
    The selected attribute affects which class IDs are written into YOLO `.txt` files and which label is assigned to each ResNet sequence. Train models using the same attribute that was used during preprocessing.

# pedsense attributes

List annotation attributes, class values, and track label types.

## Synopsis

```bash
uv run pedsense attributes
```

## Description

Prints two sections:

1. **Behavioral attributes** — per-box properties used for classification (pass to `--attribute`)
2. **Track label types** — object categories from the XML annotations (pass to `--track-labels`)

## Output

```
Behavioral attributes (use with --attribute):
  cross:     ['not-crossing', 'crossing']
  action:    ['standing', 'walking']
  look:      ['not-looking', 'looking']
  occlusion: ['none', 'part', 'full']

Track label types (use with --track-labels):
  pedestrian  (pedestrian variant)
  ped         (pedestrian variant)
  people      (pedestrian variant)
  traffic_light
  crosswalk
```

## Behavioral Attributes

Behavioral attributes are per-frame properties of a pedestrian bounding box. Their class value order determines YOLO class IDs.

| Attribute | Classes | Description |
|-----------|---------|-------------|
| `cross` | `not-crossing`, `crossing` | Whether the pedestrian is crossing the road (default) |
| `action` | `standing`, `walking` | Body movement state |
| `look` | `not-looking`, `looking` | Whether the pedestrian is looking towards the vehicle |
| `occlusion` | `none`, `part`, `full` | Degree of occlusion |

## Track Label Types

Track labels are the object category assigned to each annotation track in the JAAD XML.

| Track Label | Type | Description |
|-------------|------|-------------|
| `pedestrian` | Pedestrian variant | Primary pedestrian label; supports all behavioral attributes |
| `ped` | Pedestrian variant | Alternate pedestrian label; same behavioral attributes |
| `people` | Pedestrian variant | Group pedestrian label; same behavioral attributes |
| `traffic_light` | Environment object | Traffic light; no behavioral attributes |
| `crosswalk` | Environment object | Crosswalk marking; no behavioral attributes |

**Pedestrian variants** are classified by the chosen `--attribute` (e.g. `cross`, `action`).

**Environment objects** (`traffic_light`, `crosswalk`) are added as additional YOLO detection classes appended after the behavioral attribute classes. They cannot be used for ResNet+LSTM training.

## Usage

```bash
# Classify pedestrians by crossing intent (default)
uv run pedsense preprocess yolo

# Classify by gaze direction
uv run pedsense preprocess yolo --attribute look

# Include ped and people variants alongside pedestrian
uv run pedsense preprocess yolo -t pedestrian -t ped -t people

# Multi-class: pedestrian intent + traffic lights + crosswalks
uv run pedsense preprocess yolo -t pedestrian -t traffic_light -t crosswalk
# → data.yaml: nc=4, names=[not-crossing, crossing, traffic_light, crosswalk]
```

!!! note
    The selected `--attribute` affects class IDs for pedestrian-type tracks. Train your model with the same attribute that was used during preprocessing.

!!! note
    For ResNet+LSTM, only pedestrian-variant tracks are ever included regardless of `--track-labels`. Non-pedestrian objects have no behavioral attributes to classify.

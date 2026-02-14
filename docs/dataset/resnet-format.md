# ResNet Format

The ResNet converter creates 16-frame pedestrian crop sequences for the ResNet+LSTM temporal classifier.

## Command

```bash
uv run pedsense preprocess resnet
```

## Output Structure

```
data/processed/resnet/
    labels.csv
    sequences/
        train/
            video_0001_0_1_3b_seq_000/
                frame_00.jpg    # 224x224 cropped pedestrian
                frame_01.jpg
                ...
                frame_15.jpg
            video_0001_0_1_3b_seq_001/
                ...
        val/
            ...
```

## labels.csv

| Column | Description |
|--------|-------------|
| `sequence_id` | Directory name (e.g., `video_0001_0_1_3b_seq_000`) |
| `label` | `crossing` or `not-crossing` |
| `video_id` | Source video (e.g., `video_0001`) |
| `track_id` | Pedestrian track ID |
| `start_frame` | First frame number in the sequence |
| `end_frame` | Last frame number in the sequence |

### Example

```csv
sequence_id,label,video_id,track_id,start_frame,end_frame
video_0001_0_1_3b_seq_000,not-crossing,video_0001,0_1_3b,0,15
video_0001_0_1_3b_seq_001,not-crossing,video_0001,0_1_3b,8,23
video_0001_0_1_3b_seq_002,crossing,video_0001,0_1_3b,16,31
```

## Sequence Generation

### Sliding Window

- **Window size:** 16 frames (configurable)
- **Stride:** 8 frames (50% overlap)
- **Overlap** roughly doubles the dataset size

### Crop and Resize

For each frame in a window:

1. Read the full frame from `data/raw/frames/`
2. Crop the pedestrian bounding box region
3. Resize the crop to **224 x 224** (ResNet input size)
4. Save as JPEG (quality 95)

### Labeling

Each sequence is labeled by **majority vote** of the `cross` attribute across all 16 frames. If a sequence spans a transition (e.g., standing → crossing), the majority label is used.

### Filtering

- Only `label="pedestrian"` tracks (with crossing intent annotations)
- Occluded frames (`occluded="1"`) are excluded
- Tracks with fewer than 16 non-occluded frames are skipped

## Train/Val Split

- Same **video-level** split as the YOLO dataset (same random seed)
- Ensures consistent splits across both pipelines
- 80% train / 20% val

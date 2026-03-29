# pedsense convert-sequences

Convert CSV keypoint sequences to npy files for training.

## Synopsis

```bash
uv run pedsense convert-sequences KEYPOINTS_DIR
```

## Arguments

| Argument | Description |
|----------|-------------|
| `KEYPOINTS_DIR` | Path to the keypoint dataset directory containing `sequences_train.csv`, `sequences_val.csv`, and `labels.csv` (e.g. `data/processed/keypoints_3s`) |

## Description

`convert-sequences` is the second step of the CSV adapter workflow introduced by `preprocess keypoints --csv`. It reads the flat CSV sequence files and writes one `.npy` file per sequence, then rewrites `labels.csv` with updated file paths — producing a directory that is identical in structure to the default npy output and fully compatible with the trainer.

This adapter pattern lets you save disk I/O during extraction (one CSV per split vs. thousands of small npy files), inspect or share the data as plain text, and delay committing to a final npy layout until you are ready to train.

## Workflow

```bash
# Step 1: Extract keypoint sequences and save as CSV
uv run pedsense preprocess keypoints \
    --prediction-horizon 3.0 \
    --keypoints-dir data/processed/keypoints_3s \
    --csv

# Step 2: Convert CSV → npy (no trainer code changes required)
uv run pedsense convert-sequences data/processed/keypoints_3s

# Step 3: Train normally
uv run pedsense train -m keypoint-lstm -n my_lstm_3s -e 50 \
    --keypoints-dir data/processed/keypoints_3s
```

## Output

After conversion the directory contains both the original CSV files and the trainer-ready npy layout:

```
data/processed/keypoints_3s/
    sequences_train.csv        ← original CSV (kept, not deleted)
    sequences_val.csv
    sequences/
        train/
            video_0001_42_000030.npy   ← (T, 17, 2) float32
            ...
        val/
            ...
    labels.csv                 ← rewritten with npy file paths
```

## CSV Format

The CSV files written by `preprocess keypoints --csv` have the following structure:

| Columns | Description |
|---------|-------------|
| `video_id` | Source video identifier |
| `track_id` | Pedestrian track ID |
| `start_frame` | First frame index in the window |
| `end_frame` | Last frame index in the window |
| `label` | `0` = not-crossing, `1` = crossing |
| `k0` … `k(T×34-1)` | Flattened keypoint values — `T` frames × 17 keypoints × 2 coordinates (x, y) |

Keypoints are normalized relative to the JAAD bounding box center and height: `(kx - cx) / h`, `(ky - cy) / h`.

## Notes

- The original CSV files are preserved after conversion — you can re-run `convert-sequences` safely.
- If `sequences_val.csv` does not exist (e.g. single-video extraction), only `sequences_train.csv` is processed.
- `labels.csv` is fully rewritten; the `file` column changes from `sequences_train.csv` to `sequences/train/{name}.npy`.

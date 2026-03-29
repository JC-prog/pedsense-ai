"""
Unified named dataset builder for PedSense-AI.

Creates a self-contained, inspectable dataset folder under data/processed/{name}/.
Supports three annotation modes and a configurable train/test/val split.

Usage (via CLI):
    uv run pedsense preprocess dataset --name my_exp --mode pedestrian --split 70/15/15
    uv run pedsense preprocess dataset --name my_kp --mode crossing_keypoint \\
        --pose-model models/detector/my_pose_model_YYYYMMDD_HHMMSS/weights/best.pt \\
        --horizon 90 --split 70/15/15
"""
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from rich.progress import track as rtrack

from pedsense.config import (
    BASE_MODELS_DIR,
    FRAMES_DIR,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    PROCESSED_DIR,
    RANDOM_SEED,
)
from pedsense.processing.annotations import PEDESTRIAN_LABELS, load_all_annotations
from pedsense.processing.keypoint_pipeline import (
    _get_extracted_fps,   # noqa: F401 (unused here but available)
    _iou,
    _normalize_keypoints,
    _run_pose_on_video,
)


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------

def _split_videos_three_way(
    video_ids: list[str],
    ratio: str = "70/15/15",
    seed: int = RANDOM_SEED,
) -> dict[str, set]:
    """Split a list of video IDs into train / test / val sets.

    Args:
        video_ids: All video IDs to split.
        ratio: Three integers separated by '/' that sum to 100 (e.g. '70/15/15').
        seed: Random seed for reproducibility.

    Returns:
        {'train': set, 'test': set, 'val': set}
    """
    parts = [int(x) for x in ratio.split("/")]
    if len(parts) != 3 or sum(parts) != 100:
        raise ValueError(
            f"--split must be three integers summing to 100 (e.g. '70/15/15'), got '{ratio}'"
        )
    vids = sorted(video_ids)
    random.seed(seed)
    random.shuffle(vids)
    n = len(vids)
    n_train = round(n * parts[0] / 100)
    n_test = round(n * parts[1] / 100)
    return {
        "train": set(vids[:n_train]),
        "test": set(vids[n_train : n_train + n_test]),
        "val": set(vids[n_train + n_test :]),
    }


def _video_split(vid_id: str, split_map: dict[str, set]) -> str:
    for split, ids in split_map.items():
        if vid_id in ids:
            return split
    return "val"  # safe fallback


# ---------------------------------------------------------------------------
# Frame copy helper
# ---------------------------------------------------------------------------

def _copy_frame(vid_id: str, frame_idx: int, dst_dir: Path) -> Path | None:
    """Copy one frame from data/raw/frames/ to dst_dir.

    Returns the destination path, or None if the source does not exist.
    """
    src = FRAMES_DIR / vid_id / f"frame_{frame_idx:06d}.jpg"
    if not src.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{vid_id}_frame_{frame_idx:06d}.jpg"
    if not dst.exists():
        shutil.copy2(src, dst)
    return dst


# ---------------------------------------------------------------------------
# Pose model loader
# ---------------------------------------------------------------------------

def _load_pose_model(pose_model: str):
    """Load a YOLO-Pose model from a variant name or a direct .pt path."""
    from ultralytics import YOLO

    p = Path(pose_model)
    if p.is_absolute() or p.exists():
        return YOLO(str(p))
    BASE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return YOLO(str(BASE_MODELS_DIR / f"{pose_model}.pt"))


# ---------------------------------------------------------------------------
# pedestrian mode — YOLO bounding-box format
# ---------------------------------------------------------------------------

def _build_pedestrian_dataset(
    out: Path,
    split_map: dict[str, set],
    annotations: dict,
    video_ids: list[str],
) -> Path:
    """Write YOLO detection labels for pedestrian crossing intent.

    Output::

        {out}/
            frames/{split}/{video_id}/{video_id}_frame_{n:06d}.jpg
            labels/{split}/{video_id}/{video_id}_frame_{n:06d}.txt
            data.yaml
    """
    # frame_labels[vid_id][frame] = [(class_id, xtl, ytl, xbr, ybr)]
    # class 0 = not-crossing, class 1 = crossing
    frame_labels: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))

    for vid_id in video_ids:
        ann = annotations[vid_id]
        for trk in ann.tracks:
            if trk.label not in PEDESTRIAN_LABELS:
                continue
            for box in trk.boxes:
                if box.cross not in ("not-crossing", "crossing"):
                    continue
                cls_id = 0 if box.cross == "not-crossing" else 1
                frame_labels[vid_id][box.frame].append(
                    (cls_id, box.xtl, box.ytl, box.xbr, box.ybr)
                )

    for vid_id in rtrack(video_ids, description="[pedestrian] writing labels..."):
        if vid_id not in frame_labels:
            continue
        ann = annotations[vid_id]
        split = _video_split(vid_id, split_map)
        frame_dst = out / "frames" / split / vid_id
        label_dst = out / "labels" / split / vid_id
        label_dst.mkdir(parents=True, exist_ok=True)

        for frame_num, boxes in sorted(frame_labels[vid_id].items()):
            _copy_frame(vid_id, frame_num, frame_dst)

            txt_path = label_dst / f"{vid_id}_frame_{frame_num:06d}.txt"
            with open(txt_path, "w") as f:
                for cls_id, xtl, ytl, xbr, ybr in boxes:
                    x_center = max(0.0, min(1.0, ((xtl + xbr) / 2) / ann.width))
                    y_center = max(0.0, min(1.0, ((ytl + ybr) / 2) / ann.height))
                    width    = max(0.0, min(1.0, (xbr - xtl) / ann.width))
                    height   = max(0.0, min(1.0, (ybr - ytl) / ann.height))
                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    data_yaml = out / "data.yaml"
    splits_present = [s for s in ("train", "test", "val") if any(split_map[s] for s in split_map)]
    config: dict = {"path": str(out.resolve()), "nc": 2, "names": ["not-crossing", "crossing"]}
    for s in ("train", "test", "val"):
        config[s] = f"frames/{s}"
    with open(data_yaml, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return data_yaml


# ---------------------------------------------------------------------------
# keypoint and crossing_keypoint modes — per-frame skeleton CSVs
# ---------------------------------------------------------------------------

def _build_keypoint_annotation(
    out: Path,
    split_map: dict[str, set],
    annotations: dict,
    video_ids: list[str],
    pose_model_name: str,
    conf: float,
    iou_threshold: float,
    crossing_keypoint: bool,
    horizon: int,
) -> Path:
    """Write per-frame keypoint CSVs, one file per video per split.

    keypoint mode columns:
        track_id, frame, k0, k1, ..., k33

    crossing_keypoint mode columns:
        track_id, frame, label, k0, k1, ..., k33

    Also copies used frame images and writes a flat labels.csv index.

    Returns path to labels.csv.
    """
    model = _load_pose_model(pose_model_name)

    kpt_cols = [f"k{i}" for i in range(34)]
    if crossing_keypoint:
        header = ["track_id", "frame", "label"] + kpt_cols
    else:
        header = ["track_id", "frame"] + kpt_cols

    for split in ("train", "test", "val"):
        (out / "annotations" / split).mkdir(parents=True, exist_ok=True)

    records: list[dict] = []  # rows for labels.csv

    for vid_id in rtrack(video_ids, description="Building keypoint annotations..."):
        ann = annotations[vid_id]
        vid_frame_dir = FRAMES_DIR / vid_id
        if not vid_frame_dir.exists():
            continue

        split = _video_split(vid_id, split_map)
        frame_dst = out / "frames" / split / vid_id

        # Run YOLO-Pose on all extracted frames for this video
        pose_dets = _run_pose_on_video(vid_frame_dir, model, conf)

        csv_path = out / "annotations" / split / f"{vid_id}.csv"
        vid_rows: list[list] = []

        for trk_idx, trk in enumerate(ann.tracks):
            if trk.label not in PEDESTRIAN_LABELS or not trk.boxes:
                continue

            boxes_by_frame = {box.frame: box for box in trk.boxes}
            track_id = trk.boxes[0].track_id or str(trk_idx)

            # Crossing anchor for this track (crossing_keypoint mode only)
            crossing_point: int | None = None
            if crossing_keypoint:
                for box in trk.boxes:
                    if box.cross == "crossing":
                        crossing_point = box.frame
                        break

            # Only process frames that exist in pose_dets (i.e. extracted frames)
            for frame_idx in sorted(f for f in boxes_by_frame if f in pose_dets):
                jaad_box = boxes_by_frame[frame_idx]

                if getattr(jaad_box, "occlusion", "none") == "full":
                    continue  # skip fully occluded frames

                jaad_bbox = (jaad_box.xtl, jaad_box.ytl, jaad_box.xbr, jaad_box.ybr)

                # Find best IoU match in YOLO-Pose detections for this frame
                best_iou = 0.0
                best_kpts: np.ndarray | None = None
                for det_bbox, kpts in pose_dets.get(frame_idx, []):
                    score = _iou(jaad_bbox, det_bbox)
                    if score > best_iou:
                        best_iou = score
                        best_kpts = kpts

                if best_kpts is None or best_iou < iou_threshold:
                    continue  # no confident match

                norm_kpts = _normalize_keypoints(best_kpts, jaad_bbox)  # (17, 2)
                flat = norm_kpts.reshape(-1).tolist()                    # 34 values

                if crossing_keypoint:
                    if crossing_point is not None:
                        label = 1 if (crossing_point - horizon) <= frame_idx < crossing_point else 0
                    else:
                        label = 0
                    row = [track_id, frame_idx, label] + flat
                else:
                    row = [track_id, frame_idx] + flat

                vid_rows.append(row)

                # Copy the frame image
                dst_frame = _copy_frame(vid_id, frame_idx, frame_dst)
                frame_file = (
                    f"frames/{split}/{vid_id}/{vid_id}_frame_{frame_idx:06d}.jpg"
                    if dst_frame is not None
                    else ""
                )

                # labels.csv record
                rec: dict = {
                    "video_id": vid_id,
                    "track_id": track_id,
                    "frame": frame_idx,
                    "split": split,
                    "annotation_file": f"annotations/{split}/{vid_id}.csv",
                    "frame_file": frame_file,
                }
                if crossing_keypoint:
                    rec["label"] = row[2]  # the label value
                records.append(rec)

        if vid_rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(vid_rows)

    # Write flat labels.csv index
    labels_csv = out / "labels.csv"
    if crossing_keypoint:
        fieldnames = ["video_id", "track_id", "frame", "label", "split", "annotation_file", "frame_file"]
    else:
        fieldnames = ["video_id", "track_id", "frame", "split", "annotation_file", "frame_file"]

    with open(labels_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return labels_csv


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_dataset(
    name: str,
    mode: str,
    split_ratio: str = "70/15/15",
    video_id: str | None = None,
    pose_model: str = "yolo11n-pose",
    horizon: int = 30,
    horizon_seconds: float | None = None,
    conf: float = 0.25,
    iou_threshold: float = 0.3,
) -> Path:
    """Build a named, self-contained dataset under data/processed/{name}/.

    Args:
        name: Output folder name. Created at data/processed/{name}/.
        mode: One of 'pedestrian', 'keypoint', 'crossing_keypoint'.
        split_ratio: Train/test/val split as 'T/Te/V' integers summing to 100.
            Default '70/15/15'.
        video_id: Process a single video only. Default: all videos.
        pose_model: YOLO-Pose variant name (e.g. 'yolo11n-pose') or path to a
            trained .pt file. Used for keypoint and crossing_keypoint modes.
        horizon: For crossing_keypoint — number of **frames** before crossing_point
            that are labeled as crossing (label=1). Default 30. Ignored if
            horizon_seconds is set.
        horizon_seconds: Convenience alternative to horizon. If set, automatically
            converts to frames using the extracted FPS from meta.json (written by
            ``preprocess frames``). E.g. ``horizon_seconds=1.0`` at 10fps → horizon=10.
        conf: YOLO-Pose detection confidence threshold.
        iou_threshold: Minimum IoU to match a YOLO-Pose detection to a JAAD track.

    Returns:
        Path to the dataset root directory (data/processed/{name}/).

    Output layout by mode::

        pedestrian:
            {name}/frames/{split}/{video_id}/*.jpg
            {name}/labels/{split}/{video_id}/*.txt   (YOLO format)
            {name}/data.yaml

        keypoint:
            {name}/frames/{split}/{video_id}/*.jpg
            {name}/annotations/{split}/{video_id}.csv
            {name}/labels.csv

        crossing_keypoint:
            {name}/frames/{split}/{video_id}/*.jpg
            {name}/annotations/{split}/{video_id}.csv   (includes 'label' column)
            {name}/labels.csv
    """
    if mode not in ("pedestrian", "keypoint", "crossing_keypoint"):
        raise ValueError(f"Unknown mode '{mode}'. Must be pedestrian, keypoint, or crossing_keypoint.")

    out = PROCESSED_DIR / name
    out.mkdir(parents=True, exist_ok=True)

    annotations = load_all_annotations()

    # Resolve horizon_seconds → horizon frames using extracted FPS from meta.json.
    # Uses the first available video's meta.json as a representative FPS.
    if horizon_seconds is not None and mode == "crossing_keypoint":
        fps = 30.0  # fallback
        for vid in sorted(annotations.keys()):
            try:
                fps = _get_extracted_fps(vid)
                if fps > 0:
                    break
            except Exception:
                pass
        horizon = max(1, round(horizon_seconds * fps))

    if video_id:
        if video_id not in annotations:
            raise FileNotFoundError(f"Video '{video_id}' not found in annotations.")
        video_ids = [video_id]
    else:
        video_ids = sorted(annotations.keys())

    split_map = _split_videos_three_way(video_ids, split_ratio)

    if mode == "pedestrian":
        _build_pedestrian_dataset(out, split_map, annotations, video_ids)
    elif mode == "keypoint":
        _build_keypoint_annotation(
            out, split_map, annotations, video_ids,
            pose_model_name=pose_model,
            conf=conf,
            iou_threshold=iou_threshold,
            crossing_keypoint=False,
            horizon=0,
        )
    else:  # crossing_keypoint
        _build_keypoint_annotation(
            out, split_map, annotations, video_ids,
            pose_model_name=pose_model,
            conf=conf,
            iou_threshold=iou_threshold,
            crossing_keypoint=True,
            horizon=horizon,
        )

    _write_readme(
        out=out,
        name=name,
        mode=mode,
        pose_model=pose_model,
        horizon=horizon,
        horizon_seconds=horizon_seconds,
        split_ratio=split_ratio,
        conf=conf,
        iou_threshold=iou_threshold,
    )

    return out


# ---------------------------------------------------------------------------
# README generator
# ---------------------------------------------------------------------------

_KEYPOINT_LABELS = [
    "Nose", "Left eye", "Right eye", "Left ear", "Right ear",
    "Left shoulder", "Right shoulder", "Left elbow", "Right elbow",
    "Left wrist", "Right wrist", "Left hip", "Right hip",
    "Left knee", "Right knee", "Left ankle", "Right ankle",
]


def _write_readme(
    out: Path,
    name: str,
    mode: str,
    pose_model: str,
    horizon: int,
    horizon_seconds: float | None,
    split_ratio: str,
    conf: float,
    iou_threshold: float,
) -> None:
    """Write a README.md dataset summary to the dataset root directory."""
    from datetime import date

    lines: list[str] = [f"# Dataset: {name}", ""]

    # --- Parameters ---
    lines += ["## Parameters", ""]
    lines += [f"| Key | Value |", "|-----|-------|"]
    lines += [f"| Mode | `{mode}` |"]
    lines += [f"| Pose model | `{pose_model}` |"]
    lines += [f"| Split | `{split_ratio}` (train/test/val, video-level) |"]
    lines += [f"| Confidence threshold | `{conf}` |"]
    lines += [f"| IoU threshold | `{iou_threshold}` |"]
    if mode == "crossing_keypoint":
        hs = f"{horizon_seconds}s → " if horizon_seconds is not None else ""
        lines += [f"| Horizon | `{hs}{horizon} frames` before crossing anchor |"]
        lines += [f"| Label | `1` = within horizon before crossing, `0` = otherwise |"]
    lines += [f"| Generated | {date.today()} |"]
    lines += [""]

    # --- Split summary (from labels.csv if available) ---
    labels_csv = out / "labels.csv"
    if labels_csv.exists():
        import csv as _csv
        with open(labels_csv) as f:
            rows = list(_csv.DictReader(f))

        lines += ["## Split Summary", ""]
        has_label = "label" in (rows[0] if rows else {})

        if has_label:
            lines += ["| Split | Videos | Tracks | Frames | Crossing (1) | Not-Crossing (0) |"]
            lines += ["|-------|--------|--------|--------|-------------|-----------------|"]
        else:
            lines += ["| Split | Videos | Tracks | Frames |"]
            lines += ["|-------|--------|--------|--------|"]

        totals = [0, 0, 0, 0, 0]
        for split in ("train", "test", "val"):
            sr = [r for r in rows if r["split"] == split]
            vids = len(set(r["video_id"] for r in sr))
            tracks = len(set((r["video_id"], r["track_id"]) for r in sr))
            frames = len(sr)
            if has_label:
                c = sum(1 for r in sr if r.get("label") == "1")
                nc = frames - c
                lines += [f"| {split.capitalize()} | {vids} | {tracks} | {frames:,} | {c:,} | {nc:,} |"]
                totals[0] += vids; totals[1] += tracks; totals[2] += frames; totals[3] += c; totals[4] += nc
            else:
                lines += [f"| {split.capitalize()} | {vids} | {tracks} | {frames:,} |"]
                totals[0] += vids; totals[1] += tracks; totals[2] += frames

        if has_label:
            lines += [f"| **Total** | **{totals[0]}** | **{totals[1]}** | **{totals[2]:,}** | **{totals[3]:,}** | **{totals[4]:,}** |"]
        else:
            lines += [f"| **Total** | **{totals[0]}** | **{totals[1]}** | **{totals[2]:,}** |"]
        lines += [""]

    # --- File structure ---
    lines += ["## File Structure", ""]
    lines += ["```"]
    lines += [f"{name}/"]
    if mode == "pedestrian":
        lines += [
            "    frames/{split}/{video_id}/{video_id}_frame_{n:06d}.jpg",
            "    labels/{split}/{video_id}/{video_id}_frame_{n:06d}.txt   # YOLO: class cx cy w h",
            "    data.yaml",
        ]
    else:
        lines += [
            "    frames/{split}/{video_id}/{video_id}_frame_{n:06d}.jpg",
            "    annotations/{split}/{video_id}.csv",
            "    labels.csv",
        ]
    lines += ["    README.md", "```", ""]

    # --- Column reference (keypoint modes only) ---
    if mode in ("keypoint", "crossing_keypoint"):
        lines += ["## Column Reference", ""]
        lines += ["`annotations/{split}/{video_id}.csv`", ""]
        lines += ["| Column | Type | Description |"]
        lines += ["|--------|------|-------------|"]
        lines += ["| `track_id` | string | Pedestrian track identifier from JAAD |"]
        lines += ["| `frame` | int | Original video frame index |"]
        if mode == "crossing_keypoint":
            lines += ["| `label` | int | `1` = within horizon before crossing point, `0` = otherwise |"]
        for i, joint in enumerate(_KEYPOINT_LABELS):
            lines += [f"| `k{i*2}` | float | {joint} — x |"]
            lines += [f"| `k{i*2+1}` | float | {joint} — y |"]
        lines += [""]
        lines += ["**Normalization:** `(pixel_coord - bbox_center) / bbox_height` — scale- and position-invariant.", ""]

    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")

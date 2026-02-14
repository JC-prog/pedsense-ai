import csv
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
from rich.progress import track

from pedsense.config import (
    FRAMES_DIR,
    RESNET_DIR,
    SEQUENCE_LENGTH,
    SEQUENCE_STRIDE,
    CROP_SIZE,
    TRAIN_SPLIT,
    RANDOM_SEED,
)
from pedsense.processing.annotations import load_all_annotations


def convert_to_resnet(
    sequence_length: int = SEQUENCE_LENGTH,
    stride: int = SEQUENCE_STRIDE,
    crop_size: tuple[int, int] = CROP_SIZE,
    train_ratio: float = TRAIN_SPLIT,
) -> Path:
    """Create cropped pedestrian sequences for ResNet+LSTM training.

    Returns path to labels.csv.
    """
    # Create output dirs
    for split in ("train", "val"):
        (RESNET_DIR / "sequences" / split).mkdir(parents=True, exist_ok=True)

    annotations = load_all_annotations()

    # Same video-level split as YOLO
    video_ids = sorted(annotations.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(video_ids)
    split_idx = int(len(video_ids) * train_ratio)
    train_videos = set(video_ids[:split_idx])

    labels_csv = RESNET_DIR / "labels.csv"
    rows: list[dict[str, str]] = []

    for vid_id in track(sorted(annotations.keys()), description="Converting to ResNet..."):
        ann = annotations[vid_id]
        split = "train" if vid_id in train_videos else "val"
        frames_dir = FRAMES_DIR / vid_id

        if not frames_dir.exists():
            continue

        for trk in ann.tracks:
            if trk.label != "pedestrian":
                continue

            # Sort boxes by frame, skip occluded
            boxes = sorted(
                [b for b in trk.boxes if not b.occluded],
                key=lambda b: b.frame,
            )

            if len(boxes) < sequence_length:
                continue

            # Sliding window
            seq_idx = 0
            for start in range(0, len(boxes) - sequence_length + 1, stride):
                window = boxes[start : start + sequence_length]

                # Determine label by majority vote
                cross_counts = Counter(b.cross for b in window)
                label = cross_counts.most_common(1)[0][0]
                if label not in ("crossing", "not-crossing"):
                    continue

                # Create sequence directory
                track_id_safe = trk.boxes[0].track_id.replace("/", "_")
                seq_name = f"{vid_id}_{track_id_safe}_seq_{seq_idx:03d}"
                seq_dir = RESNET_DIR / "sequences" / split / seq_name

                if seq_dir.exists() and len(list(seq_dir.iterdir())) == sequence_length:
                    # Already processed, just record
                    rows.append({
                        "sequence_id": seq_name,
                        "label": label,
                        "video_id": vid_id,
                        "track_id": trk.boxes[0].track_id,
                        "start_frame": str(window[0].frame),
                        "end_frame": str(window[-1].frame),
                    })
                    seq_idx += 1
                    continue

                seq_dir.mkdir(parents=True, exist_ok=True)

                # Crop and save each frame
                valid = True
                for i, box in enumerate(window):
                    src_img = frames_dir / f"frame_{box.frame:06d}.jpg"
                    if not src_img.exists():
                        valid = False
                        break

                    img = cv2.imread(str(src_img))
                    if img is None:
                        valid = False
                        break

                    # Crop bounding box (clamp to image bounds)
                    h, w = img.shape[:2]
                    x1 = max(0, int(box.xtl))
                    y1 = max(0, int(box.ytl))
                    x2 = min(w, int(box.xbr))
                    y2 = min(h, int(box.ybr))

                    if x2 <= x1 or y2 <= y1:
                        valid = False
                        break

                    crop = img[y1:y2, x1:x2]
                    crop = cv2.resize(crop, crop_size)
                    cv2.imwrite(
                        str(seq_dir / f"frame_{i:02d}.jpg"),
                        crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95],
                    )

                if valid:
                    rows.append({
                        "sequence_id": seq_name,
                        "label": label,
                        "video_id": vid_id,
                        "track_id": trk.boxes[0].track_id,
                        "start_frame": str(window[0].frame),
                        "end_frame": str(window[-1].frame),
                    })
                    seq_idx += 1
                else:
                    # Clean up partial sequence
                    shutil.rmtree(seq_dir, ignore_errors=True)

    # Write labels.csv
    with open(labels_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sequence_id", "label", "video_id", "track_id", "start_frame", "end_frame"]
        )
        writer.writeheader()
        writer.writerows(rows)

    return labels_csv

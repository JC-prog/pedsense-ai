import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml
from rich.progress import track

from pedsense.config import (
    FRAMES_DIR,
    YOLO_DIR,
    TRAIN_SPLIT,
    RANDOM_SEED,
)
from pedsense.processing.annotations import load_all_annotations, ATTRIBUTE_LABELS, PEDESTRIAN_LABELS


def convert_to_yolo(
    train_ratio: float = TRAIN_SPLIT,
    attribute: str = "cross",
    track_labels: list[str] | None = None,
) -> Path:
    """Convert all annotations to YOLO format.

    Split at the video level to prevent data leakage.
    Returns path to data.yaml.

    Args:
        attribute: Annotation attribute to classify on for pedestrian-type tracks.
                   Must be a key in ATTRIBUTE_LABELS. Run 'pedsense attributes' to list options.
        track_labels: Track label types to include. Pedestrian variants (pedestrian, ped, people)
                      are classified by `attribute`; all others become additional detection classes
                      appended after the attribute classes. Default None = ["pedestrian"] only.
    """
    # Resolve which track labels to process
    if track_labels is None:
        track_labels = ["pedestrian"]

    attr_classes = ATTRIBUTE_LABELS[attribute]
    ped_set = set(PEDESTRIAN_LABELS) & set(track_labels)
    extra_labels = [t for t in track_labels if t not in PEDESTRIAN_LABELS]
    all_names = attr_classes + extra_labels

    # Create output dirs
    for split in ("train", "val"):
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    annotations = load_all_annotations()

    # Split videos into train/val
    video_ids = sorted(annotations.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(video_ids)
    split_idx = int(len(video_ids) * train_ratio)
    train_videos = set(video_ids[:split_idx])
    val_videos = set(video_ids[split_idx:])

    # Build per-frame label data: {video_id: {frame: [(class_id, xtl, ytl, xbr, ybr)]}}
    frame_labels: dict[str, dict[int, list[tuple[int, float, float, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for vid_id, ann in annotations.items():
        for trk in ann.tracks:
            if trk.label in ped_set:
                # Pedestrian variant: classify by behavioral attribute
                for box in trk.boxes:
                    label_value = getattr(box, attribute)
                    if label_value not in attr_classes:
                        continue
                    cls_id = attr_classes.index(label_value)
                    frame_labels[vid_id][box.frame].append(
                        (cls_id, box.xtl, box.ytl, box.xbr, box.ybr)
                    )
            elif trk.label in extra_labels:
                # Non-pedestrian object: class ID = position in extra_labels list
                cls_id = len(attr_classes) + extra_labels.index(trk.label)
                for box in trk.boxes:
                    frame_labels[vid_id][box.frame].append(
                        (cls_id, box.xtl, box.ytl, box.xbr, box.ybr)
                    )

    # Process each video
    for vid_id in track(sorted(frame_labels.keys()), description="Converting to YOLO..."):
        ann = annotations[vid_id]
        split = "train" if vid_id in train_videos else "val"
        frames_dir = FRAMES_DIR / vid_id

        if not frames_dir.exists():
            continue

        for frame_num, boxes in sorted(frame_labels[vid_id].items()):
            src_img = frames_dir / f"frame_{frame_num:06d}.jpg"
            if not src_img.exists():
                continue

            img_name = f"{vid_id}_frame_{frame_num:06d}.jpg"
            dst_img = YOLO_DIR / "images" / split / img_name
            label_name = f"{vid_id}_frame_{frame_num:06d}.txt"
            dst_label = YOLO_DIR / "labels" / split / label_name

            # Copy image
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            # Write YOLO label
            with open(dst_label, "w") as f:
                for cls_id, xtl, ytl, xbr, ybr in boxes:
                    x_center = ((xtl + xbr) / 2) / ann.width
                    y_center = ((ytl + ybr) / 2) / ann.height
                    width = (xbr - xtl) / ann.width
                    height = (ybr - ytl) / ann.height
                    # Clamp to [0, 1]
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    width = max(0.0, min(1.0, width))
                    height = max(0.0, min(1.0, height))
                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # Write data.yaml
    data_yaml = YOLO_DIR / "data.yaml"
    config = {
        "path": str(YOLO_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(all_names),
        "names": all_names,
    }
    with open(data_yaml, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return data_yaml

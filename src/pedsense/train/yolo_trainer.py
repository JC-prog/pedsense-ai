import random
import shutil
from datetime import datetime
from pathlib import Path

import torch
import yaml
from rich.progress import track
from ultralytics import YOLO

from pedsense.config import (
    BASE_MODELS_DIR,
    DETECTOR_MODELS_DIR,
    DETECTOR_POSE_MODELS_DIR,
    FRAMES_DIR,
    POSE_DIR,
    PROCESSED_DIR,
    RANDOM_SEED,
    TRAIN_SPLIT,
    YOLO_DIR,
)
from pedsense.processing.annotations import PEDESTRIAN_LABELS, load_all_annotations


def _prepare_detector_data() -> Path:
    """Build a 1-class (pedestrian) YOLO dataset for standalone detector training.

    Filters to PEDESTRIAN_LABELS only (pedestrian, ped, people).
    Returns path to data.yaml.
    """
    detector_dir = PROCESSED_DIR / "yolo_detector"
    for split in ("train", "val"):
        (detector_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (detector_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    annotations = load_all_annotations()
    video_ids = sorted(annotations.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(video_ids)
    split_idx = int(len(video_ids) * TRAIN_SPLIT)
    train_videos = set(video_ids[:split_idx])

    ped_set = set(PEDESTRIAN_LABELS)

    for vid_id in track(sorted(annotations.keys()), description="Preparing detector data..."):
        ann = annotations[vid_id]
        split = "train" if vid_id in train_videos else "val"
        frames_dir = FRAMES_DIR / vid_id

        if not frames_dir.exists():
            continue

        frame_boxes: dict[int, list[tuple[float, float, float, float]]] = {}
        for trk in ann.tracks:
            if trk.label not in ped_set:
                continue
            for box in trk.boxes:
                frame_boxes.setdefault(box.frame, []).append(
                    (box.xtl, box.ytl, box.xbr, box.ybr)
                )

        for frame_num, boxes in frame_boxes.items():
            src_img = frames_dir / f"frame_{frame_num:06d}.jpg"
            if not src_img.exists():
                continue

            img_name = f"{vid_id}_frame_{frame_num:06d}.jpg"
            dst_img = detector_dir / "images" / split / img_name
            label_name = f"{vid_id}_frame_{frame_num:06d}.txt"
            dst_label = detector_dir / "labels" / split / label_name

            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)

            with open(dst_label, "w") as f:
                for xtl, ytl, xbr, ybr in boxes:
                    x_center = ((xtl + xbr) / 2) / ann.width
                    y_center = ((ytl + ybr) / 2) / ann.height
                    width = (xbr - xtl) / ann.width
                    height = (ybr - ytl) / ann.height
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    width = max(0.0, min(1.0, width))
                    height = max(0.0, min(1.0, height))
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    data_yaml = detector_dir / "data.yaml"
    config = {
        "path": str(detector_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["pedestrian"],
    }
    with open(data_yaml, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return data_yaml


def train_yolo(
    name: str | None = None,
    epochs: int = 50,
    batch_size: int = 16,
    imgsz: int = 640,
    model_variant: str = "yolo26n",
    patience: int = 100,
    device: str | None = None,
    degrees: float = 0.0,
    scale: float = 0.5,
    mosaic: float = 1.0,
    mixup: float = 0.0,
    fliplr: float = 0.5,
) -> Path:
    """Fine-tune YOLO26 on the JAAD crossing intent dataset.

    Returns path to the saved model directory.
    """
    data_yaml = YOLO_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"YOLO dataset not found at {data_yaml}. Run 'pedsense preprocess yolo' first."
        )

    DETECTOR_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Build output name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = name if name else "yolo"
    output_name = f"{prefix}_{timestamp}"

    # Select device
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    # Load pretrained model (download to models/base/ if not cached)
    BASE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(BASE_MODELS_DIR / f"{model_variant}.pt"))

    # Train
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        patience=patience,
        device=device,
        project=str(DETECTOR_MODELS_DIR),
        name=output_name,
        degrees=degrees,
        scale=scale,
        mosaic=mosaic,
        mixup=mixup,
        fliplr=fliplr,
    )

    return DETECTOR_MODELS_DIR / output_name


def train_yolo_detector(
    name: str | None = None,
    epochs: int = 50,
    batch_size: int = 16,
    imgsz: int = 640,
    model_variant: str = "yolo26n",
    patience: int = 100,
    device: str | None = None,
    degrees: float = 0.0,
    scale: float = 0.5,
    mosaic: float = 1.0,
    mixup: float = 0.0,
    fliplr: float = 0.5,
) -> Path:
    """Train a 1-class YOLO26 pedestrian detector on JAAD data.

    Prepares a single-class (pedestrian) dataset internally — no preprocessing step required.
    Returns path to the saved model directory.
    """
    DETECTOR_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = name if name else "yolo-detector"
    output_name = f"{prefix}_{timestamp}"

    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    data_yaml = _prepare_detector_data()

    BASE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(BASE_MODELS_DIR / f"{model_variant}.pt"))

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        patience=patience,
        device=device,
        project=str(DETECTOR_MODELS_DIR),
        name=output_name,
        degrees=degrees,
        scale=scale,
        mosaic=mosaic,
        mixup=mixup,
        fliplr=fliplr,
    )

    return DETECTOR_MODELS_DIR / output_name


def train_yolo_pose(
    name: str | None = None,
    epochs: int = 100,
    batch_size: int = 8,
    imgsz: int = 640,
    model_variant: str = "yolo11n-pose",
    patience: int = 100,
    device: str | None = None,
) -> Path:
    """Fine-tune a YOLO-Pose model on the JAAD pose dataset.

    Requires pose dataset to exist (run 'pedsense preprocess pose' first).
    Returns path to the saved model directory.
    """
    data_yaml = POSE_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Pose dataset not found at {data_yaml}. Run 'pedsense preprocess pose' first."
        )

    DETECTOR_POSE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = name if name else "yolo-pose"
    output_name = f"{prefix}_{timestamp}"

    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    BASE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(BASE_MODELS_DIR / f"{model_variant}.pt"))

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        patience=patience,
        device=device,
        project=str(DETECTOR_POSE_MODELS_DIR),
        name=output_name,
    )

    return DETECTOR_POSE_MODELS_DIR / output_name


def train_yolo_resume(
    model_dir: Path,
    additional_epochs: int,
    device: str | None = None,
) -> Path:
    """Continue training a YOLO model for additional epochs from last.pt.

    Reads data path and hyperparameters from the model's args.yaml.
    Returns path to the new model directory.
    """
    last_pt = model_dir / "weights" / "last.pt"
    args_file = model_dir / "args.yaml"

    with open(args_file) as f:
        args = yaml.safe_load(f)

    data_path = args["data"]
    batch_size = args.get("batch", 16)
    imgsz = args.get("imgsz", 640)

    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{model_dir.name}_resumed_{timestamp}"

    # Save resumed model alongside the original (same parent dir, works for any sub-folder)
    project_dir = model_dir.parent

    model = YOLO(str(last_pt))
    model.train(
        data=data_path,
        epochs=additional_epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project=str(project_dir),
        name=output_name,
    )

    return project_dir / output_name

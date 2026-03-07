import random
import shutil
from pathlib import Path

import yaml
from rich.progress import track
from ultralytics import YOLO

from pedsense.config import BASE_MODELS_DIR, FRAMES_DIR, POSE_DIR, RANDOM_SEED, TRAIN_SPLIT


def extract_pose_labels(
    model_variant: str = "yolo11n-pose",
    video_id: str | None = None,
    conf: float = 0.25,
) -> Path:
    """Run YOLO-Pose on extracted frames and save YOLO pose-format labels.

    Uses a pretrained YOLO-Pose model to detect pedestrians and extract 17 COCO
    keypoints per detection. Saves results in YOLO pose label format to
    data/processed/pose/ with an 80/20 video-level train/val split.

    Requires frames to exist (run 'preprocess frames' first).
    Returns path to data.yaml.
    """
    for split in ("train", "val"):
        (POSE_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (POSE_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Determine which videos to process
    if video_id:
        vid_dirs = [FRAMES_DIR / video_id]
        if not vid_dirs[0].exists():
            raise FileNotFoundError(f"Frame directory not found: {vid_dirs[0]}")
    else:
        vid_dirs = sorted(d for d in FRAMES_DIR.iterdir() if d.is_dir())

    if not vid_dirs:
        raise FileNotFoundError(
            f"No frame directories found in {FRAMES_DIR}. Run 'preprocess frames' first."
        )

    # Consistent 80/20 train/val split (video-level, same seed as other steps)
    all_vids = sorted(d.name for d in FRAMES_DIR.iterdir() if d.is_dir())
    random.seed(RANDOM_SEED)
    random.shuffle(all_vids)
    split_idx = int(len(all_vids) * TRAIN_SPLIT)
    train_set = set(all_vids[:split_idx])

    # Load pretrained YOLO-Pose model (downloads to models/base/ if not cached)
    BASE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(BASE_MODELS_DIR / f"{model_variant}.pt"))

    for vid_dir in track(vid_dirs, description="Extracting pose labels..."):
        vid_id = vid_dir.name
        split = "train" if vid_id in train_set else "val"
        frame_paths = sorted(vid_dir.glob("frame_*.jpg"))

        for img_path in frame_paths:
            results = model(str(img_path), conf=conf, verbose=False)
            result = results[0]

            if result.keypoints is None or len(result.boxes) == 0:
                continue

            img_name = f"{vid_id}_{img_path.name}"
            dst_img = POSE_DIR / "images" / split / img_name
            dst_label = POSE_DIR / "labels" / split / img_name.replace(".jpg", ".txt")

            if not dst_img.exists():
                shutil.copy2(img_path, dst_img)

            h, w = result.orig_shape
            with open(dst_label, "w") as f:
                for box, kpts in zip(result.boxes.xywhn.cpu(), result.keypoints.data.cpu()):
                    cx, cy, bw, bh = box.tolist()
                    kp_parts = []
                    for kp in kpts:
                        # kp: [x_pixel, y_pixel, confidence] — normalize to 0-1
                        kx = float(kp[0]) / w
                        ky = float(kp[1]) / h
                        kp_parts.append(f"{kx:.6f} {ky:.6f} 2")
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {' '.join(kp_parts)}\n")

    # Write data.yaml
    data_yaml = POSE_DIR / "data.yaml"
    config = {
        "path": str(POSE_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["pedestrian"],
        "kpt_shape": [17, 3],
    }
    with open(data_yaml, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return data_yaml

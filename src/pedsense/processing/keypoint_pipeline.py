"""
Keypoint pipeline: YOLO-Pose detection → JAAD track alignment → sequence dataset.

Produces per-pedestrian keypoint sequences of shape (T, 17, 2) anchored to
crossing_point, with bounding-box-relative normalization. Output goes to
data/processed/keypoints/.

Pipeline:
    1. Load JAAD annotations (bboxes + crossing labels per pedestrian per frame).
    2. Run YOLO-Pose on extracted frames per video (once per video, kept in memory).
    3. For each frame, IoU-match YOLO-Pose detections to JAAD pedestrian bboxes.
    4. Build sliding-window sequences ending at or before crossing_point.
    5. Normalize each keypoint relative to the JAAD bbox center and height.
    6. Save (T, 17, 2) numpy arrays + labels.csv index.
"""
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
from rich.progress import track as rtrack
from ultralytics import YOLO

from pedsense.config import (
    BASE_MODELS_DIR,
    CLIPS_DIR,
    FRAMES_DIR,
    KEYPOINTS_DIR,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    SEQUENCE_STRIDE,
    TRAIN_SPLIT,
)
from pedsense.processing.annotations import PEDESTRIAN_LABELS, load_all_annotations


def _iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Intersection over Union for two (x1, y1, x2, y2) boxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _normalize_keypoints(
    kpts: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> np.ndarray:
    """Normalize (17, 2) pixel keypoints relative to JAAD bbox center and height.

    Each joint (kx, ky) becomes ((kx - cx) / h, (ky - cy) / h) where cx, cy
    is the bbox center and h is the bbox height. This makes the representation
    view-independent (scale- and position-invariant).
    """
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    h = bbox[3] - bbox[1]
    if h <= 0.0:
        return np.zeros_like(kpts)
    out = kpts.copy()
    out[:, 0] = (kpts[:, 0] - cx) / h
    out[:, 1] = (kpts[:, 1] - cy) / h
    return out


def _run_pose_on_video(
    vid_dir: Path,
    model: YOLO,
    conf: float,
) -> dict[int, list[tuple[tuple[float, float, float, float], np.ndarray]]]:
    """Run YOLO-Pose on all frames in a video directory.

    Returns {frame_idx: [(bbox_xyxy, kpts_17x2_pixels), ...]} where frame_idx
    matches the original video frame number preserved in the filename.
    """
    detections: dict[int, list] = {}

    for img_path in sorted(vid_dir.glob("frame_*.jpg")):
        frame_idx = int(img_path.stem.removeprefix("frame_"))

        results = model(str(img_path), conf=conf, verbose=False)
        result = results[0]

        if result.keypoints is None or len(result.boxes) == 0:
            detections[frame_idx] = []
            continue

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4) pixel
        kpts_xy = result.keypoints.xy.cpu().numpy()   # (N, 17, 2) pixel

        detections[frame_idx] = [
            (
                (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                kpts.astype(np.float32),
            )
            for box, kpts in zip(boxes_xyxy, kpts_xy)
        ]

    return detections


def _get_extracted_fps(vid_id: str, fallback: float = 30.0) -> float:
    """Return the effective FPS of extracted frames for a video.

    Reads extracted_fps from data/raw/frames/{vid_id}/meta.json (written by
    extract_frames). Falls back to reading the source clip for frames extracted
    before meta.json was introduced, then to `fallback` if neither is available.
    """
    meta_path = FRAMES_DIR / vid_id / "meta.json"
    if meta_path.exists():
        try:
            fps = float(json.loads(meta_path.read_text())["extracted_fps"])
            if fps > 0:
                return fps
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            pass

    clip = CLIPS_DIR / f"{vid_id}.mp4"
    if clip.exists():
        cap = cv2.VideoCapture(str(clip))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps > 0:
            return fps

    return fallback


def _track_crossing_info(
    boxes_by_frame: dict[int, object],
) -> tuple[int | None, int]:
    """Return (first_crossing_frame, last_annotated_frame) for a pedestrian track.

    first_crossing_frame is None if the pedestrian never crosses.
    """
    first_crossing: int | None = None
    last_frame = 0

    for frame_idx in sorted(boxes_by_frame):
        box = boxes_by_frame[frame_idx]
        last_frame = frame_idx
        if first_crossing is None and getattr(box, "cross", "") == "crossing":
            first_crossing = frame_idx

    return first_crossing, last_frame


def build_keypoint_dataset(
    model_variant: str = "yolo11n-pose",
    conf: float = 0.25,
    iou_threshold: float = 0.3,
    sequence_length: int = SEQUENCE_LENGTH,
    sequence_stride: int = SEQUENCE_STRIDE,
    prediction_horizon: float | None = 1.0,
    video_id: str | None = None,
) -> Path:
    """Build the keypoint sequence dataset from JAAD annotations and extracted frames.

    Args:
        model_variant: Pretrained YOLO-Pose model (e.g. yolo11n-pose, yolo11s-pose).
        conf: YOLO-Pose detection confidence threshold.
        iou_threshold: Minimum IoU to accept a YOLO detection as matching a JAAD track.
        sequence_length: Number of frames T per sequence window.
        sequence_stride: Step size between consecutive windows (in annotated frames).
        prediction_horizon: Seconds before crossing_point that observation windows must
            end by. Defaults to 1.0 second, giving the model a consistent prediction
            gap on JAAD clips. Pass None to revert to 1 frame before crossing_point.
        video_id: Process a single video only; default processes all.

    Returns:
        Path to labels.csv.

    Output layout::

        data/processed/keypoints/
            sequences/
                train/{video_id}_{track_id}_{start_frame:06d}.npy  # (T, 17, 2)
                val/{...}.npy
            labels.csv  # video_id, track_id, start_frame, end_frame, label, split, file
    """
    for split in ("train", "val"):
        (KEYPOINTS_DIR / "sequences" / split).mkdir(parents=True, exist_ok=True)

    # Load all JAAD annotations
    annotations = load_all_annotations()

    # Reproducible video-level train/val split — matches other preprocessing steps
    all_vids = sorted(annotations.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(all_vids)
    split_idx = int(len(all_vids) * TRAIN_SPLIT)
    train_set = set(all_vids[:split_idx])

    if video_id:
        if video_id not in annotations:
            raise FileNotFoundError(f"Video '{video_id}' not found in annotations.")
        process_vids = [video_id]
    else:
        process_vids = all_vids

    # Load pretrained YOLO-Pose model once
    BASE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(BASE_MODELS_DIR / f"{model_variant}.pt"))

    records: list[dict] = []

    for vid_id in rtrack(process_vids, description="Building keypoint sequences..."):
        ann = annotations[vid_id]
        vid_dir = FRAMES_DIR / vid_id

        if not vid_dir.exists():
            continue  # frames not yet extracted for this video

        split = "train" if vid_id in train_set else "val"
        seq_dir = KEYPOINTS_DIR / "sequences" / split

        # Run YOLO-Pose on all frames for this video (once, reused across all tracks)
        pose_dets = _run_pose_on_video(vid_dir, model, conf)

        # Horizon in frames — computed once per video from extracted FPS (matches frames on disk)
        horizon_frames: int | None = None
        if prediction_horizon is not None:
            fps = _get_extracted_fps(vid_id)
            horizon_frames = max(1, round(prediction_horizon * fps))

        for trk_idx, trk in enumerate(ann.tracks):
            if trk.label not in PEDESTRIAN_LABELS or not trk.boxes:
                continue

            # frame → BoundingBox lookup for this pedestrian track
            boxes_by_frame = {box.frame: box for box in trk.boxes}

            # Only include frames that have both a JAAD annotation AND an extracted
            # frame on disk (i.e. a YOLO-Pose detection entry). This handles
            # downsampled extractions (e.g. 1fps) where pose_dets only contains
            # every Nth frame while JAAD annotates every native frame.
            annotated_frames = sorted(f for f in boxes_by_frame if f in pose_dets)

            if len(annotated_frames) < sequence_length:
                continue  # track too short for even one window

            first_crossing, last_frame = _track_crossing_info(boxes_by_frame)
            label = 1 if first_crossing is not None else 0

            # Anchor = latest frame a window may end on.
            # With no horizon: 1 frame before crossing (original behaviour).
            # With horizon: crossing_point - horizon_frames, enforcing a fixed
            # prediction gap so all crossing samples have the same difficulty.
            if label == 1:
                gap = horizon_frames if horizon_frames is not None else 1
                anchor = first_crossing - gap
            else:
                anchor = last_frame

            # Unique, stable track identifier for filenames
            track_id = trk.boxes[0].track_id or str(trk_idx)

            # Sliding window over annotated frames (not raw frame indices,
            # to handle any gaps in JAAD annotations)
            i = 0
            while i + sequence_length - 1 < len(annotated_frames):
                window_frames = annotated_frames[i : i + sequence_length]

                # Window must end at or before the crossing anchor
                if window_frames[-1] > anchor:
                    break

                seq = _build_sequence(
                    window_frames, boxes_by_frame, pose_dets, iou_threshold, sequence_length
                )

                if seq is not None:
                    seq_name = f"{vid_id}_{track_id}_{window_frames[0]:06d}.npy"
                    seq_path = seq_dir / seq_name
                    np.save(seq_path, seq)

                    records.append(
                        {
                            "video_id": vid_id,
                            "track_id": track_id,
                            "start_frame": window_frames[0],
                            "end_frame": window_frames[-1],
                            "label": label,
                            "split": split,
                            "file": str(seq_path.relative_to(KEYPOINTS_DIR)),
                        }
                    )

                i += sequence_stride

    labels_csv = KEYPOINTS_DIR / "labels.csv"
    _write_labels_csv(labels_csv, records)
    return labels_csv


def _build_sequence(
    window_frames: list[int],
    boxes_by_frame: dict[int, object],
    pose_dets: dict[int, list],
    iou_threshold: float,
    sequence_length: int,
) -> np.ndarray | None:
    """Build one (T, 17, 2) normalized keypoint array for a window of frames.

    Returns None if any frame in the window is invalid (no matching YOLO-Pose
    detection above iou_threshold, or full occlusion in JAAD annotations).
    """
    seq = np.zeros((sequence_length, 17, 2), dtype=np.float32)

    for i, frame_idx in enumerate(window_frames):
        jaad_box = boxes_by_frame.get(frame_idx)
        if jaad_box is None:
            return None

        # Drop fully occluded frames — keypoints are likely unreliable
        if getattr(jaad_box, "occlusion", "none") == "full":
            return None

        jaad_bbox = (jaad_box.xtl, jaad_box.ytl, jaad_box.xbr, jaad_box.ybr)

        # Find the best-matching YOLO-Pose detection for this frame by IoU
        best_iou = 0.0
        best_kpts: np.ndarray | None = None

        for det_bbox, kpts in pose_dets.get(frame_idx, []):
            score = _iou(jaad_bbox, det_bbox)
            if score > best_iou:
                best_iou = score
                best_kpts = kpts

        if best_kpts is None or best_iou < iou_threshold:
            return None  # no confident match for this frame

        seq[i] = _normalize_keypoints(best_kpts, jaad_bbox)

    return seq


def _write_labels_csv(path: Path, records: list[dict]) -> None:
    fieldnames = ["video_id", "track_id", "start_frame", "end_frame", "label", "split", "file"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

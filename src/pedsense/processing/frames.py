from pathlib import Path

import cv2
from rich.progress import track as rtrack

from pedsense.config import CLIPS_DIR, FRAMES_DIR


def extract_frames(video_id: str | None = None) -> None:
    """Extract frames from MP4 videos to data/raw/frames/{video_id}/frame_{N:06d}.jpg.

    If video_id is provided, only extract for that video.
    Skips videos whose frame directory already exists and is non-empty.
    """
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    if video_id:
        videos = [CLIPS_DIR / f"{video_id}.mp4"]
        if not videos[0].exists():
            raise FileNotFoundError(f"Video not found: {videos[0]}")
    else:
        videos = sorted(CLIPS_DIR.glob("*.mp4"))

    if not videos:
        raise FileNotFoundError(f"No MP4 files found in {CLIPS_DIR}")

    for video_path in rtrack(videos, description="Extracting frames..."):
        vid_id = video_path.stem
        out_dir = FRAMES_DIR / vid_id

        # Skip if already extracted
        if out_dir.exists() and any(out_dir.iterdir()):
            continue

        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                out_path = out_dir / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                frame_idx += 1
        finally:
            cap.release()

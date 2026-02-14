from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO

from pedsense.config import BASE_MODELS_DIR, CUSTOM_MODELS_DIR, YOLO_DIR


def train_yolo(
    name: str | None = None,
    epochs: int = 50,
    batch_size: int = 16,
    imgsz: int = 640,
    model_variant: str = "yolo26n",
    device: str | None = None,
) -> Path:
    """Fine-tune YOLO26 on the JAAD crossing intent dataset.

    Returns path to the saved model directory.
    """
    data_yaml = YOLO_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"YOLO dataset not found at {data_yaml}. Run 'pedsense preprocess yolo' first."
        )

    CUSTOM_MODELS_DIR.mkdir(parents=True, exist_ok=True)

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
        device=device,
        project=str(CUSTOM_MODELS_DIR),
        name=output_name,
    )

    return CUSTOM_MODELS_DIR / output_name

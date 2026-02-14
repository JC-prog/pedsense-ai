import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image
from rich.console import Console
from rich.progress import track

from pedsense.config import (
    BASE_MODELS_DIR,
    CUSTOM_MODELS_DIR,
    FRAMES_DIR,
    YOLO_DIR,
    CROP_SIZE,
    TRAIN_SPLIT,
    RANDOM_SEED,
)
from pedsense.processing.annotations import load_all_annotations
from pedsense.train.resnet_lstm import ResNetClassifier

console = Console()

LABEL_MAP = {"not-crossing": 0, "crossing": 1}


class HybridCropDataset(Dataset):
    """Dataset of single-frame pedestrian crops with crossing intent labels."""

    def __init__(self, crops: list[tuple[Path, int]], transform=None):
        self.crops = crops
        self.transform = transform

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.crops[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def _prepare_hybrid_yolo_data() -> Path:
    """Create a 1-class (pedestrian-only) YOLO dataset from existing annotations.

    Returns path to data.yaml for the 1-class dataset.
    """
    import random
    import yaml

    hybrid_yolo_dir = YOLO_DIR.parent / "yolo_hybrid"

    for split in ("train", "val"):
        (hybrid_yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (hybrid_yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    annotations = load_all_annotations()
    video_ids = sorted(annotations.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(video_ids)
    split_idx = int(len(video_ids) * TRAIN_SPLIT)
    train_videos = set(video_ids[:split_idx])

    for vid_id in track(sorted(annotations.keys()), description="Preparing hybrid YOLO data..."):
        ann = annotations[vid_id]
        split = "train" if vid_id in train_videos else "val"
        frames_dir = FRAMES_DIR / vid_id

        if not frames_dir.exists():
            continue

        # Collect all pedestrian boxes per frame (all labels, 1 class)
        frame_boxes: dict[int, list[tuple[float, float, float, float]]] = {}
        for trk in ann.tracks:
            for box in trk.boxes:
                frame_boxes.setdefault(box.frame, []).append(
                    (box.xtl, box.ytl, box.xbr, box.ybr)
                )

        for frame_num, boxes in frame_boxes.items():
            src_img = frames_dir / f"frame_{frame_num:06d}.jpg"
            if not src_img.exists():
                continue

            img_name = f"{vid_id}_frame_{frame_num:06d}.jpg"
            dst_img = hybrid_yolo_dir / "images" / split / img_name
            label_name = f"{vid_id}_frame_{frame_num:06d}.txt"
            dst_label = hybrid_yolo_dir / "labels" / split / label_name

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

    data_yaml = hybrid_yolo_dir / "data.yaml"
    config = {
        "path": str(hybrid_yolo_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["pedestrian"],
    }
    with open(data_yaml, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return data_yaml


def _generate_crops(split_videos: set[str], split: str, output_dir: Path) -> list[tuple[Path, int]]:
    """Use ground-truth bounding boxes to generate labeled crops for ResNet training."""
    annotations = load_all_annotations()
    crops: list[tuple[Path, int]] = []
    crops_dir = output_dir / "crops" / split
    crops_dir.mkdir(parents=True, exist_ok=True)

    for vid_id in track(sorted(split_videos), description=f"Generating {split} crops..."):
        ann = annotations.get(vid_id)
        if ann is None:
            continue
        frames_dir = FRAMES_DIR / vid_id
        if not frames_dir.exists():
            continue

        for trk in ann.tracks:
            if trk.label != "pedestrian":
                continue
            for box in trk.boxes:
                if box.cross not in LABEL_MAP or box.occluded:
                    continue

                crop_name = f"{vid_id}_{box.track_id}_{box.frame:06d}.jpg"
                crop_path = crops_dir / crop_name

                if not crop_path.exists():
                    src_img = frames_dir / f"frame_{box.frame:06d}.jpg"
                    if not src_img.exists():
                        continue
                    img = cv2.imread(str(src_img))
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    x1, y1 = max(0, int(box.xtl)), max(0, int(box.ytl))
                    x2, y2 = min(w, int(box.xbr)), min(h, int(box.ybr))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = cv2.resize(img[y1:y2, x1:x2], CROP_SIZE)
                    cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

                crops.append((crop_path, LABEL_MAP[box.cross]))

    return crops


def train_hybrid(
    name: str | None = None,
    yolo_model: str | None = None,
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    yolo_epochs: int = 50,
    device: str | None = None,
) -> Path:
    """Train hybrid pipeline: YOLO26 pedestrian detector + ResNet intent classifier.

    Returns path to saved model directory.
    """
    import random

    CUSTOM_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = name if name else "hybrid"
    output_name = f"{prefix}_{timestamp}"
    output_dir = CUSTOM_MODELS_DIR / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # --- Stage 1: YOLO pedestrian detector ---
    if yolo_model:
        console.print(f"[cyan]Using existing YOLO model: {yolo_model}[/cyan]")
        yolo_best = Path(yolo_model)
    else:
        console.print("[yellow]Stage 1: Training YOLO26 pedestrian detector...[/yellow]")
        data_yaml = _prepare_hybrid_yolo_data()
        BASE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        yolo = YOLO(str(BASE_MODELS_DIR / "yolo26n.pt"))
        yolo.train(
            data=str(data_yaml),
            epochs=yolo_epochs,
            batch=batch_size,
            imgsz=640,
            device="0" if device == "cuda" else device,
            project=str(output_dir),
            name="yolo_detector",
        )
        yolo_best = output_dir / "yolo_detector" / "weights" / "best.pt"

    # Copy YOLO model to output
    shutil.copy2(yolo_best, output_dir / "yolo_detector.pt")

    # --- Stage 2: ResNet intent classifier ---
    console.print("[yellow]Stage 2: Training ResNet intent classifier...[/yellow]")

    # Prepare train/val split
    annotations = load_all_annotations()
    video_ids = sorted(annotations.keys())
    random.seed(RANDOM_SEED)
    random.shuffle(video_ids)
    split_idx = int(len(video_ids) * TRAIN_SPLIT)
    train_videos = set(video_ids[:split_idx])
    val_videos = set(video_ids[split_idx:])

    # Generate crops using ground truth boxes
    train_crops = _generate_crops(train_videos, "train", output_dir)
    val_crops = _generate_crops(val_videos, "val", output_dir)

    if not train_crops:
        raise RuntimeError("No training crops generated. Ensure frames are extracted.")

    transform = transforms.Compose([
        transforms.Resize(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = HybridCropDataset(train_crops, transform=transform)
    val_dataset = HybridCropDataset(val_crops, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Model
    model = ResNetClassifier(num_classes=2).to(dev)

    # Compute class weights
    counts = [0, 0]
    for _, label in train_crops:
        counts[label] += 1
    total = sum(counts)
    weights = torch.tensor(
        [total / (2 * c) if c > 0 else 1.0 for c in counts], dtype=torch.float32
    ).to(dev)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in track(train_loader, description=f"Epoch {epoch+1}/{epochs} [train]"):
            images = images.to(dev)
            labels = labels.to(dev)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(dev)
                labels = labels.to(dev)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / train_total if train_total > 0 else 0
        val_acc = val_correct / val_total if val_total > 0 else 0

        console.print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss / train_total:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss / val_total:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "resnet_classifier.pt")
            console.print(f"  [green]New best classifier saved (val_acc={val_acc:.4f})[/green]")

    # Save config
    config = {
        "model_type": "hybrid",
        "num_classes": 2,
        "class_names": ["not-crossing", "crossing"],
        "yolo_detector": "yolo_detector.pt",
        "resnet_classifier": "resnet_classifier.pt",
        "crop_size": list(CROP_SIZE),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "best_val_acc": best_val_acc,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Clean up intermediate crops directory
    crops_dir = output_dir / "crops"
    if crops_dir.exists():
        shutil.rmtree(crops_dir)

    return output_dir

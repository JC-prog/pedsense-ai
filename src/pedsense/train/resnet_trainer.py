import csv
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from rich.console import Console
from rich.progress import track

from pedsense.config import CUSTOM_MODELS_DIR, RESNET_DIR, SEQUENCE_LENGTH, CROP_SIZE
from pedsense.train.resnet_lstm import ResNetLSTM

console = Console()

LABEL_MAP = {"not-crossing": 0, "crossing": 1}


class PedestrianSequenceDataset(Dataset):
    """Dataset that loads pedestrian image sequences for ResNet+LSTM training."""

    def __init__(self, split: str = "train", transform=None):
        self.split = split
        self.transform = transform
        self.sequences: list[tuple[Path, int]] = []

        labels_csv = RESNET_DIR / "labels.csv"
        if not labels_csv.exists():
            raise FileNotFoundError(
                f"Labels file not found: {labels_csv}. Run 'pedsense preprocess resnet' first."
            )

        with open(labels_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                seq_dir = RESNET_DIR / "sequences" / split / row["sequence_id"]
                if seq_dir.exists() and row["label"] in LABEL_MAP:
                    self.sequences.append((seq_dir, LABEL_MAP[row["label"]]))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        seq_dir, label = self.sequences[idx]
        frames = []

        for i in range(SEQUENCE_LENGTH):
            img_path = seq_dir / f"frame_{i:02d}.jpg"
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            frames.append(img)

        # Stack into (seq_len, C, H, W)
        sequence = torch.stack(frames)
        return sequence, label


def _compute_class_weights(dataset: PedestrianSequenceDataset) -> torch.Tensor:
    """Compute inverse-frequency class weights for balanced training."""
    counts = [0, 0]
    for _, label in dataset.sequences:
        counts[label] += 1
    total = sum(counts)
    weights = [total / (len(counts) * c) if c > 0 else 1.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def train_resnet_lstm(
    name: str | None = None,
    epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    device: str | None = None,
) -> Path:
    """Train ResNet+LSTM on pedestrian crossing sequences.

    Returns path to saved model directory.
    """
    CUSTOM_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Build output name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = name if name else "resnet-lstm"
    output_name = f"{prefix}_{timestamp}"
    output_dir = CUSTOM_MODELS_DIR / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    # Transforms
    transform = transforms.Compose([
        transforms.Resize(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets
    train_dataset = PedestrianSequenceDataset(split="train", transform=transform)
    val_dataset = PedestrianSequenceDataset(split="val", transform=transform)

    if len(train_dataset) == 0:
        raise RuntimeError("No training sequences found. Run 'pedsense preprocess resnet' first.")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Model
    model = ResNetLSTM(num_classes=2).to(dev)

    # Loss with class weights
    class_weights = _compute_class_weights(train_dataset).to(dev)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer and scheduler
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for sequences, labels in track(train_loader, description=f"Epoch {epoch+1}/{epochs} [train]"):
            sequences = sequences.to(dev)
            labels = labels.to(dev)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(dev)
                labels = labels.to(dev)
                outputs = model(sequences)
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

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_dir / "best.pt")
            console.print(f"  [green]New best model saved (val_acc={val_acc:.4f})[/green]")

    # Save last model
    torch.save(model.state_dict(), output_dir / "last.pt")

    # Save config
    config = {
        "model_type": "resnet-lstm",
        "num_classes": 2,
        "class_names": ["not-crossing", "crossing"],
        "hidden_size": 256,
        "num_layers": 1,
        "sequence_length": SEQUENCE_LENGTH,
        "crop_size": list(CROP_SIZE),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "best_val_acc": best_val_acc,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    return output_dir

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from rich.console import Console
from rich.progress import track

from pedsense.config import CLASSIFIER_MODELS_DIR, KEYPOINTS_DIR
from pedsense.train.resnet_lstm import KeypointLSTM

console = Console()


class KeypointSequenceDataset(Dataset):
    """Loads (T, 17, 2) keypoint sequences from data/processed/keypoints/."""

    def __init__(self, split: str = "train"):
        self.sequences: list[tuple[Path, int]] = []

        labels_csv = KEYPOINTS_DIR / "labels.csv"
        if not labels_csv.exists():
            raise FileNotFoundError(
                f"Labels not found: {labels_csv}. "
                "Run 'pedsense preprocess keypoints' first."
            )

        with open(labels_csv) as f:
            for row in csv.DictReader(f):
                if row["split"] != split:
                    continue
                seq_path = KEYPOINTS_DIR / row["file"]
                if seq_path.exists():
                    self.sequences.append((seq_path, int(row["label"])))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.sequences[idx]
        seq = np.load(path)                  # (T, 17, 2)
        seq = seq.reshape(seq.shape[0], -1)  # (T, 34)
        return torch.from_numpy(seq).float(), label


def _compute_class_weights(dataset: KeypointSequenceDataset) -> torch.Tensor:
    counts = [0, 0]
    for _, label in dataset.sequences:
        counts[label] += 1
    total = sum(counts)
    weights = [total / (2 * c) if c > 0 else 1.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


def _f1(labels: list[int], preds: list[int]) -> float:
    """Binary F1 for class 1 (crossing)."""
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def _auc(labels: list[int], probs: list[float]) -> float:
    """Binary AUC via Mann-Whitney U statistic."""
    pos = [p for l, p in zip(labels, probs) if l == 1]
    neg = [p for l, p in zip(labels, probs) if l == 0]
    if not pos or not neg:
        return 0.5
    return sum(1.0 for p in pos for n in neg if p > n) / (len(pos) * len(neg))


def train_keypoint_lstm(
    name: str | None = None,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    device: str | None = None,
) -> Path:
    """Train a KeypointLSTM on keypoint sequences.

    Checkpoints on validation F1 — more meaningful than accuracy for the
    imbalanced crossing/not-crossing split in JAAD.

    Returns path to saved model directory.
    """
    CLASSIFIER_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = CLASSIFIER_MODELS_DIR / f"{name or 'keypoint-lstm'}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    train_dataset = KeypointSequenceDataset(split="train")
    val_dataset = KeypointSequenceDataset(split="val")

    if len(train_dataset) == 0:
        raise RuntimeError(
            "No training sequences found. Run 'pedsense preprocess keypoints' first."
        )

    num_workers = min(4, os.cpu_count() or 0)
    pin = device != "cpu"

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )

    sample, _ = train_dataset[0]
    sequence_length = sample.shape[0]  # T
    input_size = sample.shape[-1]  # 34

    model = KeypointLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(dev)

    criterion = nn.CrossEntropyLoss(weight=_compute_class_weights(train_dataset).to(dev))
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0.0

    results_csv = output_dir / "results.csv"
    with open(results_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_acc",
            "val_loss", "val_acc", "val_f1", "val_auc",
        ])

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for seqs, labels in track(train_loader, description=f"Epoch {epoch+1}/{epochs} [train]"):
            seqs, labels = seqs.to(dev), labels.to(dev)
            optimizer.zero_grad()
            outputs = model(seqs)
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
        val_preds: list[int] = []
        val_labels: list[int] = []
        val_probs: list[float] = []

        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs, labels_dev = seqs.to(dev), labels.to(dev)
                outputs = model(seqs)
                val_loss += criterion(outputs, labels_dev).item() * labels.size(0)
                val_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().tolist())
                val_preds.extend(outputs.argmax(dim=1).cpu().tolist())
                val_labels.extend(labels.tolist())

        n_val = len(val_labels)
        train_acc = train_correct / train_total
        val_acc = sum(p == l for p, l in zip(val_preds, val_labels)) / n_val
        val_f1 = _f1(val_labels, val_preds)
        val_auc = _auc(val_labels, val_probs)

        console.print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss/train_total:.4f}  Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss/n_val:.4f}  Acc: {val_acc:.4f}  "
            f"F1: {val_f1:.4f}  AUC: {val_auc:.4f}"
        )

        with open(results_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1,
                f"{train_loss/train_total:.6f}",
                f"{train_acc:.6f}",
                f"{val_loss/n_val:.6f}",
                f"{val_acc:.6f}",
                f"{val_f1:.6f}",
                f"{val_auc:.6f}",
            ])

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), output_dir / "best.pt")
            console.print(f"  [green]New best model (val_f1={val_f1:.4f})[/green]")

    torch.save(model.state_dict(), output_dir / "last.pt")

    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "model_type": "keypoint-lstm",
            "input_size": input_size,
            "sequence_length": sequence_length,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "best_val_f1": best_val_f1,
        }, f, indent=2)

    return output_dir

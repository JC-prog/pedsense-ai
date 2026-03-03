import typer
from rich.console import Console

app = typer.Typer(help="PedSense: Pedestrian Intent Prediction Suite")
console = Console()


@app.command()
def setup():
    """Verify project structure and prepare environment."""
    from pedsense.config import (
        RAW_DIR,
        PROCESSED_DIR,
        FRAMES_DIR,
        YOLO_DIR,
        RESNET_DIR,
        BASE_MODELS_DIR,
        CUSTOM_MODELS_DIR,
    )

    for folder in [RAW_DIR, PROCESSED_DIR, FRAMES_DIR, YOLO_DIR, RESNET_DIR, BASE_MODELS_DIR, CUSTOM_MODELS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    console.print("[bold green]Project structure verified![/bold green]")
    console.print("Place your JAAD raw data in [bold cyan]data/raw/[/bold cyan]")


@app.command()
def attributes():
    """List annotation attributes, class values, and track label types."""
    from pedsense.processing.annotations import ATTRIBUTE_LABELS, PEDESTRIAN_LABELS, TRACK_LABELS

    console.print("[bold]Behavioral attributes[/bold] (use with --attribute):")
    for attr, labels in ATTRIBUTE_LABELS.items():
        console.print(f"  [bold cyan]{attr}[/bold cyan]: {labels}")

    console.print()
    console.print("[bold]Track label types[/bold] (use with --track-labels):")
    for label in TRACK_LABELS:
        tag = " [dim](pedestrian variant)[/dim]" if label in PEDESTRIAN_LABELS else ""
        console.print(f"  [bold cyan]{label}[/bold cyan]{tag}")


@app.command()
def preprocess(
    step: str = typer.Argument(
        "all",
        help="Processing step: frames, yolo, resnet, or all",
    ),
    video_id: str = typer.Option(
        None,
        "--video", "-v",
        help="Process a single video (e.g., video_0001). Default: all videos.",
    ),
    fps: float = typer.Option(
        None,
        "--fps",
        help="Target FPS for frame extraction (e.g. 10, 15). Default: all frames at native FPS.",
    ),
    attribute: str = typer.Option(
        "cross",
        "--attribute", "-a",
        help="Annotation attribute to classify on for pedestrian tracks. Run 'pedsense attributes' to see options.",
    ),
    track_labels: list[str] | None = typer.Option(
        None,
        "--track-labels", "-t",
        help=(
            "Track label types to include. Repeat for multiple: -t pedestrian -t traffic_light. "
            "Pedestrian variants (pedestrian, ped, people) are classified by --attribute; "
            "others become additional YOLO detection classes. "
            "Default: pedestrian only. Run 'pedsense attributes' to see options."
        ),
    ),
):
    """Extract frames and prepare datasets from raw JAAD data."""
    from pedsense.processing import extract_frames, convert_to_yolo, convert_to_resnet
    from pedsense.processing.annotations import ATTRIBUTE_LABELS, TRACK_LABELS

    if step in ("all", "yolo", "resnet") and attribute not in ATTRIBUTE_LABELS:
        console.print(
            f"[bold red]Unknown attribute '{attribute}'. "
            "Run 'pedsense attributes' to see options.[/bold red]"
        )
        raise typer.Exit(1)

    if track_labels:
        unknown = [t for t in track_labels if t not in TRACK_LABELS]
        if unknown:
            console.print(
                f"[bold red]Unknown track labels: {unknown}. "
                "Run 'pedsense attributes' to see options.[/bold red]"
            )
            raise typer.Exit(1)

    if step in ("all", "frames"):
        if fps is not None:
            console.print(
                f"[bold yellow]Frame downsampling enabled (target {fps} FPS). "
                "ResNet+LSTM sequence coverage may be reduced.[/bold yellow]"
            )
        console.print("[bold cyan]Extracting frames...[/bold cyan]")
        extract_frames(video_id=video_id, fps=fps)
        console.print("[bold green]Frame extraction complete.[/bold green]")

    if step in ("all", "yolo"):
        console.print("[bold cyan]Converting to YOLO format...[/bold cyan]")
        data_yaml = convert_to_yolo(attribute=attribute, track_labels=track_labels or None)
        console.print(f"[bold green]YOLO dataset ready: {data_yaml}[/bold green]")

    if step in ("all", "resnet"):
        console.print("[bold cyan]Converting to ResNet+LSTM format...[/bold cyan]")
        labels_csv = convert_to_resnet(attribute=attribute, ped_labels=track_labels or None)
        console.print(f"[bold green]ResNet dataset ready: {labels_csv}[/bold green]")


@app.command()
def train(
    model: str = typer.Option(
        ...,
        "--model", "-m",
        help="Model to train: yolo, yolo-detector, resnet-lstm, hybrid",
    ),
    name: str = typer.Option(
        None,
        "--name", "-n",
        help="Custom name prefix for saved model folder (datetime is appended automatically)",
    ),
    epochs: int = typer.Option(50, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(16, "--batch-size", "-b", help="Batch size"),
    yolo_model: str = typer.Option(
        None,
        "--yolo-model",
        help="Path to existing YOLO model for hybrid training (skips YOLO training stage)",
    ),
    yolo_variant: str = typer.Option(
        "yolo26n",
        "--yolo-variant",
        help="YOLO26 base model variant: yolo26n, yolo26s, yolo26m, yolo26l, yolo26x",
    ),
    imgsz: int = typer.Option(
        640,
        "--imgsz",
        help="Input image size for YOLO training. Common: 320 (fast), 640 (default), 1280 (best accuracy). Applies to: yolo, yolo-detector.",
    ),
    patience: int = typer.Option(
        100,
        "--patience",
        help="Early stopping: stop training if no improvement for N epochs. Set 0 to disable. Applies to: yolo, yolo-detector.",
    ),
    lr: float = typer.Option(
        None,
        "--lr",
        help="Learning rate (default: 1e-4). Applies to: resnet-lstm, hybrid.",
    ),
    yolo_epochs: int = typer.Option(
        50,
        "--yolo-epochs",
        help="YOLO detector training epochs for hybrid stage 1 (default: 50). Applies to: hybrid.",
    ),
    device: str = typer.Option(
        None,
        "--device",
        help="Training device: '0' for first GPU, 'cpu', '0,1' for multi-GPU. Default: auto-detect.",
    ),
):
    """Train a model for pedestrian crossing intent prediction."""
    from pedsense.train import train_yolo, train_yolo_detector, train_resnet_lstm, train_hybrid

    if model == "yolo":
        console.print(f"[bold yellow]Training YOLO26 model ({yolo_variant})...[/bold yellow]")
        output = train_yolo(name=name, epochs=epochs, batch_size=batch_size, model_variant=yolo_variant, imgsz=imgsz, patience=patience, device=device)
    elif model == "yolo-detector":
        console.print(f"[bold yellow]Training YOLO26 pedestrian detector ({yolo_variant})...[/bold yellow]")
        output = train_yolo_detector(name=name, epochs=epochs, batch_size=batch_size, model_variant=yolo_variant, imgsz=imgsz, patience=patience, device=device)
    elif model == "resnet-lstm":
        console.print("[bold yellow]Training ResNet+LSTM model...[/bold yellow]")
        output = train_resnet_lstm(name=name, epochs=epochs, batch_size=batch_size, learning_rate=lr or 1e-4, device=device)
    elif model == "hybrid":
        console.print("[bold yellow]Training Hybrid model (YOLO + ResNet)...[/bold yellow]")
        output = train_hybrid(name=name, yolo_model=yolo_model, epochs=epochs, batch_size=batch_size, learning_rate=lr or 1e-4, yolo_epochs=yolo_epochs, device=device)
    else:
        console.print(f"[bold red]Unknown model: {model}. Use 'yolo', 'yolo-detector', 'resnet-lstm', or 'hybrid'.[/bold red]")
        raise typer.Exit(1)

    console.print(f"[bold green]Model saved to: {output}[/bold green]")


@app.command()
def resume():
    """List trained YOLO models and continue training for additional epochs."""
    import csv
    import yaml
    from pathlib import Path
    from rich.table import Table
    from pedsense.config import CUSTOM_MODELS_DIR
    from pedsense.train.yolo_trainer import train_yolo_resume

    models = []
    for d in sorted(CUSTOM_MODELS_DIR.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "weights" / "last.pt").exists():
            continue
        args_file = d / "args.yaml"
        if not args_file.exists():
            continue

        with open(args_file) as f:
            args = yaml.safe_load(f)

        epochs_trained, best_map50 = 0, "-"
        results_csv = d / "results.csv"
        if results_csv.exists():
            with open(results_csv) as f:
                rows = list(csv.DictReader(f))
            if rows:
                epochs_trained = len(rows)
                val = rows[-1].get("metrics/mAP50(B)", "").strip()
                best_map50 = f"{float(val):.3f}" if val else "-"

        data_path = args.get("data", "")
        model_type = "yolo-detector" if "yolo_detector" in data_path else "yolo"
        variant = Path(args.get("model", "unknown")).stem

        models.append({
            "dir": d,
            "name": d.name,
            "type": model_type,
            "variant": variant,
            "epochs": epochs_trained,
            "map50": best_map50,
        })

    if not models:
        console.print("[bold red]No resumable YOLO models found in models/custom/.[/bold red]")
        raise typer.Exit(1)

    table = Table(title="Resumable YOLO Models")
    table.add_column("#", style="bold cyan", justify="right")
    table.add_column("Name")
    table.add_column("Type")
    table.add_column("Variant")
    table.add_column("Epochs", justify="right")
    table.add_column("mAP50", justify="right")

    for i, m in enumerate(models, 1):
        table.add_row(str(i), m["name"], m["type"], m["variant"], str(m["epochs"]), m["map50"])

    console.print(table)

    idx = typer.prompt("\nSelect model number", type=int)
    if idx < 1 or idx > len(models):
        console.print("[bold red]Invalid selection.[/bold red]")
        raise typer.Exit(1)

    selected = models[idx - 1]
    console.print(f"Selected: [bold cyan]{selected['name']}[/bold cyan] ({selected['epochs']} epochs trained so far)")

    additional = typer.prompt("Additional epochs to train", type=int)
    if additional < 1:
        console.print("[bold red]Must be at least 1 epoch.[/bold red]")
        raise typer.Exit(1)

    total = selected["epochs"] + additional
    console.print(f"[yellow]Training {additional} more epochs → total {total} epochs[/yellow]")

    output = train_yolo_resume(model_dir=selected["dir"], additional_epochs=additional)
    console.print(f"[bold green]Resumed model saved to: {output}[/bold green]")


@app.command()
def demo(
    model_path: str = typer.Option(
        None,
        "--model", "-m",
        help="Model directory name or path. If not given, uses latest in models/custom/.",
    ),
    port: int = typer.Option(7860, "--port", "-p", help="Gradio server port"),
):
    """Launch the Gradio web interface for inference."""
    from pedsense.demo import launch_demo

    console.print("[bold magenta]Launching Gradio demo...[/bold magenta]")
    launch_demo(model_path=model_path, port=port)


if __name__ == "__main__":
    app()

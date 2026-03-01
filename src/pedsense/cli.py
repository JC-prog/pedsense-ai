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
):
    """Train a model for pedestrian crossing intent prediction."""
    from pedsense.train import train_yolo, train_yolo_detector, train_resnet_lstm, train_hybrid

    if model == "yolo":
        console.print(f"[bold yellow]Training YOLO26 model ({yolo_variant})...[/bold yellow]")
        output = train_yolo(name=name, epochs=epochs, batch_size=batch_size, model_variant=yolo_variant)
    elif model == "yolo-detector":
        console.print(f"[bold yellow]Training YOLO26 pedestrian detector ({yolo_variant})...[/bold yellow]")
        output = train_yolo_detector(name=name, epochs=epochs, batch_size=batch_size, model_variant=yolo_variant)
    elif model == "resnet-lstm":
        console.print("[bold yellow]Training ResNet+LSTM model...[/bold yellow]")
        output = train_resnet_lstm(name=name, epochs=epochs, batch_size=batch_size)
    elif model == "hybrid":
        console.print("[bold yellow]Training Hybrid model (YOLO + ResNet)...[/bold yellow]")
        output = train_hybrid(name=name, yolo_model=yolo_model, epochs=epochs, batch_size=batch_size)
    else:
        console.print(f"[bold red]Unknown model: {model}. Use 'yolo', 'yolo-detector', 'resnet-lstm', or 'hybrid'.[/bold red]")
        raise typer.Exit(1)

    console.print(f"[bold green]Model saved to: {output}[/bold green]")


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

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
):
    """Extract frames and prepare datasets from raw JAAD data."""
    from pedsense.processing import extract_frames, convert_to_yolo, convert_to_resnet

    if step in ("all", "frames"):
        console.print("[bold cyan]Extracting frames...[/bold cyan]")
        extract_frames(video_id=video_id)
        console.print("[bold green]Frame extraction complete.[/bold green]")

    if step in ("all", "yolo"):
        console.print("[bold cyan]Converting to YOLO format...[/bold cyan]")
        data_yaml = convert_to_yolo()
        console.print(f"[bold green]YOLO dataset ready: {data_yaml}[/bold green]")

    if step in ("all", "resnet"):
        console.print("[bold cyan]Converting to ResNet+LSTM format...[/bold cyan]")
        labels_csv = convert_to_resnet()
        console.print(f"[bold green]ResNet dataset ready: {labels_csv}[/bold green]")


@app.command()
def train(
    model: str = typer.Option(
        ...,
        "--model", "-m",
        help="Model to train: yolo, resnet-lstm, hybrid",
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
):
    """Train a model for pedestrian crossing intent prediction."""
    from pedsense.train import train_yolo, train_resnet_lstm, train_hybrid

    if model == "yolo":
        console.print("[bold yellow]Training YOLO26 model...[/bold yellow]")
        output = train_yolo(name=name, epochs=epochs, batch_size=batch_size)
    elif model == "resnet-lstm":
        console.print("[bold yellow]Training ResNet+LSTM model...[/bold yellow]")
        output = train_resnet_lstm(name=name, epochs=epochs, batch_size=batch_size)
    elif model == "hybrid":
        console.print("[bold yellow]Training Hybrid model (YOLO + ResNet)...[/bold yellow]")
        output = train_hybrid(name=name, yolo_model=yolo_model, epochs=epochs, batch_size=batch_size)
    else:
        console.print(f"[bold red]Unknown model: {model}. Use 'yolo', 'resnet-lstm', or 'hybrid'.[/bold red]")
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

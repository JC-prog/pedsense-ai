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
        POSE_DIR,
        BASE_MODELS_DIR,
        DETECTOR_MODELS_DIR,
        CLASSIFIER_MODELS_DIR,
    )

    for folder in [RAW_DIR, PROCESSED_DIR, FRAMES_DIR, YOLO_DIR, RESNET_DIR, POSE_DIR, BASE_MODELS_DIR, DETECTOR_MODELS_DIR, CLASSIFIER_MODELS_DIR]:
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
def download(
    repo_id: str = typer.Argument(
        ...,
        help="Hugging Face repo ID (e.g. JCProg/pedsense-yolo).",
    ),
    name: str = typer.Option(
        None,
        "--name", "-n",
        help="Local folder name for the model. Defaults to the repo name (part after '/').",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        is_flag=True,
        help="Overwrite an existing local model folder.",
    ),
):
    """Download a PedSense model from Hugging Face Hub into the correct local directory."""
    import shutil
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
    from pedsense.config import DETECTOR_MODELS_DIR, CLASSIFIER_MODELS_DIR
    from pedsense.demo import _detect_model_type

    local_name = name or repo_id.split("/")[-1]

    console.print(f"[bold cyan]Downloading [/bold cyan][bold]{repo_id}[/bold] from Hugging Face Hub...")
    try:
        cached_dir = snapshot_download(repo_id=repo_id)
    except RepositoryNotFoundError:
        console.print(f"[bold red]Repository not found: '{repo_id}'. Check the repo ID and your access.[/bold red]")
        raise typer.Exit(1)
    except HfHubHTTPError as e:
        console.print(f"[bold red]Network error while contacting Hugging Face Hub:[/bold red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Unexpected download error:[/bold red] {e}")
        raise typer.Exit(1)

    cached_path = Path(cached_dir)
    model_type = _detect_model_type(cached_path)

    DETECTOR_TYPES = {"yolo", "yolo-pose", "hybrid"}
    CLASSIFIER_TYPES = {"keypoint-lstm", "resnet-lstm"}

    if model_type in DETECTOR_TYPES:
        dest_base = DETECTOR_MODELS_DIR
    elif model_type in CLASSIFIER_TYPES:
        dest_base = CLASSIFIER_MODELS_DIR
    else:
        console.print(
            f"[bold yellow]Warning: unrecognised model type '{model_type}'. "
            f"Placing under models/detector/ — move manually if needed.[/bold yellow]"
        )
        dest_base = DETECTOR_MODELS_DIR

    dest = dest_base / local_name

    if dest.exists():
        if not force:
            console.print(
                f"[bold yellow]'{local_name}' already exists at {dest}\n"
                f"Use --force / -f to overwrite.[/bold yellow]"
            )
            raise typer.Exit(1)
        console.print(f"[yellow]--force: removing existing {dest}[/yellow]")
        shutil.rmtree(dest)

    dest_base.mkdir(parents=True, exist_ok=True)
    shutil.copytree(str(cached_path), str(dest))

    console.print(
        f"[bold green]Model installed:[/bold green] "
        f"[bold cyan]{local_name}[/bold cyan] ({model_type}) → {dest}"
    )


@app.command()
def preprocess(
    step: str = typer.Argument(
        "all",
        help="Processing step: frames, yolo, resnet, pose, keypoints, or all",
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
    pose_variant: str = typer.Option(
        "yolo11n-pose",
        "--pose-variant",
        help="YOLO-Pose model for keypoint extraction: yolo11n-pose, yolo11s-pose, yolo11m-pose. Applies to: pose.",
    ),
    conf: float = typer.Option(
        0.25,
        "--conf",
        help="Detection confidence threshold for pose/keypoint extraction (default: 0.25). Applies to: pose, keypoints.",
    ),
    iou_threshold: float = typer.Option(
        0.3,
        "--iou-threshold",
        help="Minimum IoU to match a YOLO-Pose detection to a JAAD pedestrian track (default: 0.3). Applies to: keypoints.",
    ),
    sequence_length: int = typer.Option(
        None,
        "--sequence-length",
        help="Number of frames per keypoint sequence window (default: from config). Applies to: keypoints.",
    ),
    sequence_stride: int = typer.Option(
        None,
        "--sequence-stride",
        help="Step between consecutive keypoint windows in annotated frames (default: from config). Applies to: keypoints.",
    ),
    prediction_horizon: float | None = typer.Option(
        1.0,
        "--prediction-horizon",
        help=(
            "Seconds before crossing_point that observation windows must end by "
            "(e.g. 1.0 = predict at least 1 second before crossing). "
            "Default: 1.0 second. Applies to: keypoints."
        ),
    ),
):
    """Extract frames and prepare datasets from raw JAAD data."""
    from pedsense.processing import extract_frames, convert_to_yolo, convert_to_resnet, extract_pose_labels, build_keypoint_dataset
    from pedsense.processing.annotations import ATTRIBUTE_LABELS, TRACK_LABELS

    if step not in ("all", "frames", "yolo", "resnet", "pose", "keypoints"):
        console.print(f"[bold red]Unknown step '{step}'. Use: frames, yolo, resnet, pose, keypoints, or all.[/bold red]")
        raise typer.Exit(1)

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

    if step == "pose":
        console.print(f"[bold cyan]Extracting pose keypoints ({pose_variant})...[/bold cyan]")
        data_yaml = extract_pose_labels(model_variant=pose_variant, video_id=video_id, conf=conf)
        console.print(f"[bold green]Pose dataset ready: {data_yaml}[/bold green]")

    if step == "keypoints":
        console.print(f"[bold cyan]Building keypoint sequence dataset ({pose_variant})...[/bold cyan]")
        from pedsense.config import SEQUENCE_LENGTH, SEQUENCE_STRIDE
        labels_csv = build_keypoint_dataset(
            model_variant=pose_variant,
            conf=conf,
            iou_threshold=iou_threshold,
            sequence_length=sequence_length if sequence_length is not None else SEQUENCE_LENGTH,
            sequence_stride=sequence_stride if sequence_stride is not None else SEQUENCE_STRIDE,
            prediction_horizon=prediction_horizon,
            video_id=video_id,
        )
        console.print(f"[bold green]Keypoint dataset ready: {labels_csv}[/bold green]")


@app.command()
def train(
    model: str = typer.Option(
        ...,
        "--model", "-m",
        help="Model to train: yolo, yolo-detector, yolo-pose, resnet-lstm, hybrid, keypoint-lstm",
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
        help="YOLO26 base model variant: yolo26n, yolo26s, yolo26m, yolo26l, yolo26x. For yolo-pose, use: yolo11n-pose, yolo11s-pose, yolo11m-pose.",
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
        help="Learning rate (default: 1e-4 for resnet-lstm/hybrid, 1e-3 for keypoint-lstm). Applies to: resnet-lstm, hybrid, keypoint-lstm.",
    ),
    hidden_size: int = typer.Option(
        128,
        "--hidden-size",
        help="LSTM hidden state size (default: 128). Applies to: keypoint-lstm.",
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
    aug_degrees: float = typer.Option(
        0.0,
        "--aug-degrees",
        help="Rotation augmentation range in degrees (e.g. 10 = random ±10°). Default 0 (off). Applies to: yolo, yolo-detector.",
    ),
    aug_scale: float = typer.Option(
        0.5,
        "--aug-scale",
        help="Scale jitter fraction (e.g. 0.5 = 50-150%% size variation). Higher helps detect pedestrians at varied distances. Applies to: yolo, yolo-detector.",
    ),
    aug_mosaic: float = typer.Option(
        1.0,
        "--aug-mosaic",
        help="Mosaic augmentation probability 0-1. 1.0 = always on (default), 0 = disabled. Applies to: yolo, yolo-detector.",
    ),
    aug_mixup: float = typer.Option(
        0.0,
        "--aug-mixup",
        help="Mixup augmentation probability 0-1. Default 0 (off). Try 0.1-0.2 to improve generalisation. Applies to: yolo, yolo-detector.",
    ),
    aug_fliplr: float = typer.Option(
        0.5,
        "--aug-fliplr",
        help="Horizontal flip probability 0-1. Default 0.5. Set to 0 if pedestrian direction matters. Applies to: yolo, yolo-detector.",
    ),
):
    """Train a model for pedestrian crossing intent prediction."""
    from pedsense.train import train_yolo, train_yolo_detector, train_yolo_pose, train_resnet_lstm, train_hybrid, train_keypoint_lstm

    if model == "yolo":
        console.print(f"[bold yellow]Training YOLO26 model ({yolo_variant})...[/bold yellow]")
        output = train_yolo(name=name, epochs=epochs, batch_size=batch_size, model_variant=yolo_variant, imgsz=imgsz, patience=patience, device=device, degrees=aug_degrees, scale=aug_scale, mosaic=aug_mosaic, mixup=aug_mixup, fliplr=aug_fliplr)
    elif model == "yolo-detector":
        console.print(f"[bold yellow]Training YOLO26 pedestrian detector ({yolo_variant})...[/bold yellow]")
        output = train_yolo_detector(name=name, epochs=epochs, batch_size=batch_size, model_variant=yolo_variant, imgsz=imgsz, patience=patience, device=device, degrees=aug_degrees, scale=aug_scale, mosaic=aug_mosaic, mixup=aug_mixup, fliplr=aug_fliplr)
    elif model == "yolo-pose":
        pose_variant = yolo_variant if yolo_variant.endswith("-pose") else "yolo11n-pose"
        console.print(f"[bold yellow]Training YOLO-Pose model ({pose_variant})...[/bold yellow]")
        output = train_yolo_pose(name=name, epochs=epochs, batch_size=batch_size, model_variant=pose_variant, imgsz=imgsz, patience=patience, device=device)
    elif model == "resnet-lstm":
        console.print("[bold yellow]Training ResNet+LSTM model...[/bold yellow]")
        output = train_resnet_lstm(name=name, epochs=epochs, batch_size=batch_size, learning_rate=lr or 1e-4, device=device)
    elif model == "hybrid":
        console.print("[bold yellow]Training Hybrid model (YOLO + ResNet)...[/bold yellow]")
        output = train_hybrid(name=name, yolo_model=yolo_model, epochs=epochs, batch_size=batch_size, learning_rate=lr or 1e-4, yolo_epochs=yolo_epochs, device=device)
    elif model == "keypoint-lstm":
        console.print("[bold yellow]Training KeypointLSTM model...[/bold yellow]")
        output = train_keypoint_lstm(name=name, epochs=epochs, batch_size=batch_size, learning_rate=lr or 1e-3, hidden_size=hidden_size, device=device)
    else:
        console.print(f"[bold red]Unknown model: {model}. Use 'yolo', 'yolo-detector', 'yolo-pose', 'resnet-lstm', 'hybrid', or 'keypoint-lstm'.[/bold red]")
        raise typer.Exit(1)

    console.print(f"[bold green]Model saved to: {output}[/bold green]")


@app.command()
def resume():
    """List trained YOLO models and continue training for additional epochs."""
    import csv
    import yaml
    from pathlib import Path
    from rich.table import Table
    from pedsense.config import DETECTOR_MODELS_DIR
    from pedsense.train.yolo_trainer import train_yolo_resume

    models = []
    for d in sorted(DETECTOR_MODELS_DIR.iterdir()):
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
        help="Model directory name to pre-select in the demo. Detection models are in models/detector/, intent classifiers in models/classifier/.",
    ),
    port: int = typer.Option(7860, "--port", "-p", help="Gradio server port"),
):
    """Launch the Gradio web interface for inference."""
    from pedsense.demo import launch_demo

    console.print("[bold magenta]Launching Gradio demo...[/bold magenta]")
    launch_demo(model_path=model_path, port=port)


if __name__ == "__main__":
    app()

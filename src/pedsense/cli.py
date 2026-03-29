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
        DETECTOR_POSE_MODELS_DIR,
        CLASSIFIER_LSTM_MODELS_DIR,
        CLASSIFIER_STGCN_MODELS_DIR,
    )

    for folder in [
        RAW_DIR, PROCESSED_DIR, FRAMES_DIR, YOLO_DIR, RESNET_DIR, POSE_DIR,
        BASE_MODELS_DIR,
        DETECTOR_MODELS_DIR,
        DETECTOR_POSE_MODELS_DIR,
        CLASSIFIER_LSTM_MODELS_DIR,
        CLASSIFIER_STGCN_MODELS_DIR,
    ]:
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
        help="YOLO-Pose model for keypoint extraction: a variant name (yolo11n-pose, yolo11s-pose, yolo11m-pose) or a path to a trained .pt file (e.g. models/detector/my_pose_model_.../weights/best.pt). Applies to: pose, keypoints.",
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
    keypoints_dir: str = typer.Option(
        None,
        "--keypoints-dir",
        help=(
            "Output directory for keypoint sequences. Default: data/processed/keypoints/. "
            "Set a different path per horizon to avoid overwriting, e.g. "
            "'data/processed/keypoints_3s'. Applies to: keypoints."
        ),
    ),
    save_csv: bool = typer.Option(
        False,
        "--csv/--no-csv",
        help=(
            "Save sequences as CSV rows instead of individual .npy files. "
            "Use 'pedsense convert-sequences' afterwards to convert back to npy "
            "for training. Applies to: keypoints."
        ),
    ),
    dataset_name: str = typer.Option(
        None,
        "--name",
        help="Dataset folder name. Creates data/processed/{name}/. Required when step=dataset.",
    ),
    dataset_mode: str = typer.Option(
        None,
        "--mode",
        help=(
            "Annotation mode for step=dataset: "
            "'pedestrian' (YOLO bounding-box labels), "
            "'keypoint' (per-frame skeleton CSVs, no label), or "
            "'crossing_keypoint' (per-frame skeletons with crossing label)."
        ),
    ),
    with_frames: bool = typer.Option(
        False,
        "--with-frames",
        help="Extract frames from clips before building the dataset. Applies to: dataset.",
    ),
    split: str = typer.Option(
        "70/15/15",
        "--split",
        help=(
            "Train/test/val split ratios as three integers summing to 100 "
            "(e.g. '70/15/15'). Applies to: dataset."
        ),
    ),
    pose_model: str = typer.Option(
        "yolo11n-pose",
        "--pose-model",
        help=(
            "YOLO-Pose variant name (e.g. 'yolo11n-pose', 'yolo11m-pose') or path to a "
            "trained .pt file (e.g. models/detector/my_pose_model_.../weights/best.pt). "
            "Applies to: dataset --mode keypoint, dataset --mode crossing_keypoint."
        ),
    ),
    horizon: int = typer.Option(
        30,
        "--horizon",
        help=(
            "Number of frames before crossing_point to label as crossing (label=1). "
            "Applies to: dataset --mode crossing_keypoint. "
            "Ignored if --horizon-seconds is set."
        ),
    ),
    horizon_seconds: float = typer.Option(
        None,
        "--horizon-seconds",
        help=(
            "Convenience alternative to --horizon. Automatically converts seconds to frames "
            "using the extracted FPS from meta.json. "
            "E.g. --horizon-seconds 1.0 at 10fps gives --horizon 10. "
            "Applies to: dataset --mode crossing_keypoint."
        ),
    ),
):
    """Extract frames and prepare datasets from raw JAAD data."""
    from pedsense.processing import extract_frames, convert_to_yolo, convert_to_resnet, extract_pose_labels, build_keypoint_dataset
    from pedsense.processing.annotations import ATTRIBUTE_LABELS, TRACK_LABELS

    if step not in ("all", "frames", "yolo", "resnet", "pose", "keypoints", "dataset"):
        console.print(f"[bold red]Unknown step '{step}'. Use: frames, yolo, resnet, pose, keypoints, dataset, or all.[/bold red]")
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

    if step == "dataset":
        if not dataset_name:
            console.print("[bold red]--name is required for step=dataset.[/bold red]")
            raise typer.Exit(1)
        if dataset_mode not in ("pedestrian", "keypoint", "crossing_keypoint"):
            console.print("[bold red]--mode must be pedestrian, keypoint, or crossing_keypoint.[/bold red]")
            raise typer.Exit(1)
        if with_frames:
            console.print("[bold cyan]Extracting frames...[/bold cyan]")
            extract_frames(video_id=video_id, fps=fps)
        console.print(f"[bold cyan]Building dataset '{dataset_name}' (mode={dataset_mode})...[/bold cyan]")
        from pedsense.processing.dataset_builder import build_dataset
        out_dir = build_dataset(
            name=dataset_name,
            mode=dataset_mode,
            split_ratio=split,
            video_id=video_id,
            pose_model=pose_model,
            horizon=horizon,
            horizon_seconds=horizon_seconds,
            conf=conf,
            iou_threshold=iou_threshold,
        )
        console.print(f"[bold green]Dataset ready: {out_dir}[/bold green]")

    if step == "keypoints":
        console.print(f"[bold cyan]Building keypoint sequence dataset ({pose_variant})...[/bold cyan]")
        from pathlib import Path as _Path
        from pedsense.config import SEQUENCE_LENGTH, SEQUENCE_STRIDE
        out_dir = _Path(keypoints_dir) if keypoints_dir else None
        labels_csv = build_keypoint_dataset(
            model_variant=pose_variant,
            conf=conf,
            iou_threshold=iou_threshold,
            sequence_length=sequence_length if sequence_length is not None else SEQUENCE_LENGTH,
            sequence_stride=sequence_stride if sequence_stride is not None else SEQUENCE_STRIDE,
            prediction_horizon=prediction_horizon,
            video_id=video_id,
            output_dir=out_dir,
            save_csv=save_csv,
        )
        if save_csv:
            console.print(f"[bold green]Keypoint CSV sequences ready in: {labels_csv.parent}[/bold green]")
            console.print("[dim]Run 'pedsense convert-sequences' to convert to npy for training.[/dim]")
        else:
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
    train_keypoints_dir: str = typer.Option(
        None,
        "--keypoints-dir",
        help=(
            "Keypoint dataset directory to train on. Default: data/processed/keypoints/. "
            "Set to a horizon-specific directory, e.g. data/processed/keypoints_3s. "
            "Applies to: keypoint-lstm."
        ),
    ),
    train_sequence_length: int = typer.Option(
        5,
        "--sequence-length",
        help="Number of consecutive frames per LSTM input window (default: 5). Applies to: keypoint-lstm.",
    ),
    train_sequence_stride: int = typer.Option(
        1,
        "--sequence-stride",
        help="Step between consecutive windows over a track's frames (default: 1). Applies to: keypoint-lstm.",
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
        from pathlib import Path as _Path
        kdir = _Path(train_keypoints_dir) if train_keypoints_dir else None
        output = train_keypoint_lstm(name=name, epochs=epochs, batch_size=batch_size, learning_rate=lr or 1e-3, hidden_size=hidden_size, device=device, keypoints_dir=kdir, sequence_length=train_sequence_length, sequence_stride=train_sequence_stride)
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
    from pedsense.config import DETECTOR_MODELS_DIR, DETECTOR_POSE_MODELS_DIR
    from pedsense.train.yolo_trainer import train_yolo_resume

    _resume_dirs = [d for base in (DETECTOR_MODELS_DIR, DETECTOR_POSE_MODELS_DIR) if base.exists() for d in base.iterdir()]
    models = []
    for d in sorted(_resume_dirs):
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
        console.print("[bold red]No resumable YOLO models found in models/detector/ or models/detector-pose/.[/bold red]")
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


@app.command(name="convert-sequences")
def convert_sequences(
    keypoints_dir: str = typer.Argument(
        ...,
        help="Path to the keypoint dataset directory containing sequences_train.csv / sequences_val.csv and labels.csv (e.g. data/processed/keypoints_3s).",
    ),
):
    """Convert CSV keypoint sequences to npy files for training.

    Use this after 'preprocess keypoints --csv' to convert the compact CSV output
    back into individual .npy sequence files expected by the trainer — without
    modifying any trainer code.
    """
    from pathlib import Path
    from pedsense.processing import convert_sequences_csv_to_npy

    kdir = Path(keypoints_dir)
    if not kdir.exists():
        console.print(f"[bold red]Directory not found: {kdir}[/bold red]")
        raise typer.Exit(1)

    console.print(f"[bold cyan]Converting CSV sequences in {kdir} to npy...[/bold cyan]")
    labels_csv = convert_sequences_csv_to_npy(kdir)
    console.print(f"[bold green]Done. Trainer-ready dataset at: {labels_csv}[/bold green]")


if __name__ == "__main__":
    app()

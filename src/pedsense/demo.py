import json
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import yaml
from torchvision import transforms
from ultralytics import YOLO

from pedsense.config import CUSTOM_MODELS_DIR, CROP_SIZE
from pedsense.train.resnet_lstm import ResNetLSTM, ResNetClassifier

CLASS_NAMES = ["not-crossing", "crossing"]
COLORS = {"not-crossing": (0, 255, 0), "crossing": (0, 0, 255)}  # BGR: green, red

# COCO skeleton connections (index pairs into the 17 keypoints)
_COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # arms
    (5, 11), (6, 12), (11, 12),               # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
]


def _find_yolo_weights(model_dir: Path) -> Path | None:
    """Find the best available YOLO weights file in an Ultralytics output directory."""
    weights_dir = model_dir / "weights"
    if not weights_dir.is_dir():
        return None
    # Prefer best.pt, then last.pt, then any .pt file
    for name in ("best.pt", "last.pt"):
        candidate = weights_dir / name
        if candidate.exists():
            return candidate
    pt_files = list(weights_dir.glob("*.pt"))
    return pt_files[0] if pt_files else None


def _list_available_models() -> list[str]:
    """List all model directories in models/custom/."""
    if not CUSTOM_MODELS_DIR.exists():
        return []
    models = []
    for d in sorted(CUSTOM_MODELS_DIR.iterdir(), reverse=True):
        if d.is_dir() and (d / "config.json").exists():
            models.append(d.name)
        elif d.is_dir() and _find_yolo_weights(d) is not None:
            # Ultralytics YOLO output structure
            models.append(d.name)
    return models


def _get_latest_model() -> str | None:
    """Return the most recently modified model directory name."""
    models = _list_available_models()
    return models[0] if models else None


def _detect_model_type(model_dir: Path) -> str:
    """Detect model type from config.json, args.yaml, or directory contents."""
    # Check args.yaml first (written by Ultralytics after training)
    args_file = model_dir / "args.yaml"
    if args_file.exists():
        with open(args_file) as f:
            args = yaml.safe_load(f)
        model_path = args.get("model", "")
        if "pose" in Path(model_path).stem:
            return "yolo-pose"

    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        return config.get("model_type", "yolo")

    # Fallback: check for YOLO weights directory
    if _find_yolo_weights(model_dir) is not None:
        return "yolo"
    if (model_dir / "best.pt").exists():
        return "resnet-lstm"
    return "yolo"


def _load_yolo_model(model_dir: Path) -> YOLO:
    """Load a trained YOLO model."""
    weights = _find_yolo_weights(model_dir)
    if weights is None:
        raise FileNotFoundError(f"No YOLO weights found in {model_dir / 'weights'}")
    return YOLO(str(weights))


def _load_resnet_lstm_model(model_dir: Path, device: torch.device) -> ResNetLSTM:
    """Load a trained ResNet+LSTM model."""
    model = ResNetLSTM(num_classes=2)
    model.load_state_dict(torch.load(model_dir / "best.pt", map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def _load_hybrid_model(model_dir: Path, device: torch.device) -> tuple[YOLO, ResNetClassifier]:
    """Load hybrid model (YOLO detector + ResNet classifier)."""
    yolo = YOLO(str(model_dir / "yolo_detector.pt"))
    resnet = ResNetClassifier(num_classes=2)
    resnet.load_state_dict(
        torch.load(model_dir / "resnet_classifier.pt", map_location=device, weights_only=True)
    )
    resnet.to(device)
    resnet.eval()
    return yolo, resnet


def _run_yolo_inference(
    video_path: str, model_dir: Path, confidence: float
) -> tuple[str, dict]:
    """Run YOLO inference on a video."""
    model = _load_yolo_model(model_dir)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    stats = {"total_detections": 0, "crossing": 0, "not_crossing": 0}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence, verbose=False)

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
                    color = COLORS.get(cls_name, (255, 255, 255))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{cls_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    stats["total_detections"] += 1
                    if cls_name == "crossing":
                        stats["crossing"] += 1
                    else:
                        stats["not_crossing"] += 1

            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    return out_path, stats


def _run_hybrid_inference(
    video_path: str, model_dir: Path, confidence: float
) -> tuple[str, dict]:
    """Run hybrid inference: YOLO detects, ResNet classifies."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo, resnet = _load_hybrid_model(model_dir, device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    stats = {"total_detections": 0, "crossing": 0, "not_crossing": 0}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo(frame, conf=confidence, verbose=False)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Crop and classify with ResNet
                    crop = frame[y1:y2, x1:x2]
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    input_tensor = transform(crop_rgb).unsqueeze(0).to(device)

                    with torch.no_grad():
                        logits = resnet(input_tensor)
                        cls_id = logits.argmax(dim=1).item()

                    cls_name = CLASS_NAMES[cls_id]
                    color = COLORS.get(cls_name, (255, 255, 255))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    stats["total_detections"] += 1
                    if cls_name == "crossing":
                        stats["crossing"] += 1
                    else:
                        stats["not_crossing"] += 1

            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    return out_path, stats


def _run_yolo_pose_inference(
    video_path: str, model_dir: Path, confidence: float
) -> tuple[str, dict]:
    """Run YOLO-Pose inference on a video, rendering keypoints and skeleton."""
    weights = _find_yolo_weights(model_dir)
    if weights is None:
        raise FileNotFoundError(f"No YOLO weights found in {model_dir / 'weights'}")
    model = YOLO(str(weights))

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = tempfile.mktemp(suffix=".mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    stats = {"total_pedestrians": 0, "model_type": "yolo-pose"}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence, verbose=False)

            for r in results:
                if r.boxes is None:
                    continue

                kpts_data = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None
                kpts_conf = r.keypoints.conf.cpu().numpy() if (r.keypoints is not None and r.keypoints.conf is not None) else None

                for i, box in enumerate(r.boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)  # cyan box
                    cv2.putText(frame, "pedestrian", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    stats["total_pedestrians"] += 1

                    if kpts_data is None or i >= len(kpts_data):
                        continue

                    kp = kpts_data[i]  # shape (17, 2)
                    kp_c = kpts_conf[i] if kpts_conf is not None else np.ones(17)

                    # Draw skeleton connections
                    for a, b in _COCO_SKELETON:
                        if kp_c[a] > 0.3 and kp_c[b] > 0.3:
                            pt_a = (int(kp[a][0]), int(kp[a][1]))
                            pt_b = (int(kp[b][0]), int(kp[b][1]))
                            if pt_a != (0, 0) and pt_b != (0, 0):
                                cv2.line(frame, pt_a, pt_b, (0, 255, 0), 2)  # green skeleton

                    # Draw keypoints
                    for j in range(17):
                        if kp_c[j] > 0.3:
                            px, py = int(kp[j][0]), int(kp[j][1])
                            if px > 0 or py > 0:
                                cv2.circle(frame, (px, py), 4, (0, 215, 255), -1)  # yellow dots

            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    return out_path, stats


def run_inference(
    video_path: str, model_name: str, confidence: float
) -> tuple[str | None, dict]:
    """Run model inference on video and return annotated output."""
    if not video_path or not model_name:
        return None, {"error": "Please select a video and model."}

    model_dir = CUSTOM_MODELS_DIR / model_name
    if not model_dir.exists():
        return None, {"error": f"Model not found: {model_dir}"}

    model_type = _detect_model_type(model_dir)

    if model_type == "yolo":
        return _run_yolo_inference(video_path, model_dir, confidence)
    elif model_type == "yolo-pose":
        return _run_yolo_pose_inference(video_path, model_dir, confidence)
    elif model_type == "hybrid":
        return _run_hybrid_inference(video_path, model_dir, confidence)
    elif model_type == "resnet-lstm":
        # ResNet+LSTM needs a detector; fall back to hybrid-like approach if YOLO available
        return None, {
            "error": "ResNet+LSTM requires a separate pedestrian detector. "
            "Use a YOLO or Hybrid model for the demo, or train a Hybrid model."
        }
    else:
        return None, {"error": f"Unknown model type: {model_type}"}


def launch_demo(model_path: str | None = None, port: int = 7860) -> None:
    """Launch Gradio interface for pedestrian intent prediction."""
    available_models = _list_available_models()
    default_model = model_path if model_path else _get_latest_model()

    with gr.Blocks(title="PedSense-AI Demo") as app:
        gr.Markdown("# PedSense-AI: Pedestrian Crossing Intent Prediction")

        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video")
                model_selector = gr.Dropdown(
                    choices=available_models,
                    value=default_model,
                    label="Select Model",
                )
                confidence_slider = gr.Slider(
                    0.1, 1.0, value=0.5, step=0.05,
                    label="Confidence Threshold",
                )
                run_button = gr.Button("Run Prediction", variant="primary")

            with gr.Column():
                video_output = gr.Video(label="Annotated Output")
                stats_output = gr.JSON(label="Detection Statistics")

        run_button.click(
            fn=run_inference,
            inputs=[video_input, model_selector, confidence_slider],
            outputs=[video_output, stats_output],
        )

    app.launch(server_port=port, share=False)

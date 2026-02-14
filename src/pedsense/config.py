from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Raw data
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLIPS_DIR = RAW_DIR / "clips"
ANNOTATIONS_DIR = RAW_DIR / "annotations"
FRAMES_DIR = RAW_DIR / "frames"

# Processed data
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
YOLO_DIR = PROCESSED_DIR / "yolo"
RESNET_DIR = PROCESSED_DIR / "resnet"

# Models
MODELS_DIR = PROJECT_ROOT / "models"
BASE_MODELS_DIR = MODELS_DIR / "base"
CUSTOM_MODELS_DIR = MODELS_DIR / "custom"

# Training defaults
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
SEQUENCE_LENGTH = 16
SEQUENCE_STRIDE = 8
CROP_SIZE = (224, 224)
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

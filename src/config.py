"""Central configuration for the DR screening project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Data layout (APTOS 2019 format: CSV with `id_code`, `diagnosis` + folder of .png images)
DATA_DIR = ROOT / "data"
RAW_IMAGE_DIR = DATA_DIR / "train_images"
LABELS_CSV = DATA_DIR / "train.csv"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "dr_model.keras"

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42

# 5-grade scale used by APTOS / EyePACS
GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
# Binary screening: grade 0 vs grade >= 1
BINARY_NAMES = ["No DR", "DR"]

# Training schedule: phase 1 trains the new head with a frozen backbone,
# phase 2 unfreezes the top of the backbone and fine-tunes with a small LR.
HEAD_EPOCHS = 5
FINE_TUNE_EPOCHS = 15
HEAD_LR = 1e-3
FINE_TUNE_LR = 1e-5
UNFREEZE_LAYERS = 60


def class_names(mode: str) -> list[str]:
    return BINARY_NAMES if mode == "binary" else GRADE_NAMES

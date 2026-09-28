"""Shared paths and configuration for the project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
DATASET_PATH = DATA_DIR / "KDDTrain+.txt"
TARGET_COLUMN = "label"
BINARY_TARGET_COLUMN = "behavior"
RANDOM_STATE = 42

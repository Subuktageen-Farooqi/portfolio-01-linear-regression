from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
from matplotlib import pyplot as plt

DEFAULT_LOW_RISK_THRESHOLD = 33.0
DEFAULT_HIGH_RISK_THRESHOLD = 66.0

class StandardScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, array: np.ndarray) -> "StandardScaler":
        self.mean_ = array.mean(axis=0)
        scale = array.std(axis=0)
        scale[scale == 0.0] = 1.0
        self.scale_ = scale
        return self

    def transform(self, array: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        return (array - self.mean_) / self.scale_



def repo_root() -> Path:
    """Resolve repository root from the src package location."""
    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    return repo_root() / "data"


def outputs_dir() -> Path:
    out = repo_root() / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def dataset_path(filename: str = "linear_data.csv") -> Path:
    return data_dir() / filename


def load_csv_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load x,y columns from a CSV with header x,y."""
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    x = data[:, 0]
    y = data[:, 1]
    return x, y


def risk_band_from_score(score: float, low_threshold: float, high_threshold: float) -> str:
    """Map a continuous score to Low/Medium/High risk bands.

    Threshold semantics:
    - Low: score < low_threshold
    - Medium: low_threshold <= score <= high_threshold
    - High: score > high_threshold
    """
    if low_threshold >= high_threshold:
        raise ValueError("low_threshold must be lower than high_threshold")

    if score < low_threshold:
        return "Low"
    if score <= high_threshold:
        return "Medium"
    return "High"

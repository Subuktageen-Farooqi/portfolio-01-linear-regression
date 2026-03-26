from __future__ import annotations

from pathlib import Path

import numpy as np


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

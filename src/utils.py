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


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_flat = np.asarray(y_true).reshape(-1)
    y_pred_flat = np.asarray(y_pred).reshape(-1)
    return {
        "mae": float(mean_absolute_error(y_true_flat, y_pred_flat)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))),
        "r2": float(r2_score(y_true_flat, y_pred_flat)),
    }


def save_metrics_json(path: Path, metrics: dict[str, object]) -> None:
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_test_predictions_csv(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    payload = pd.DataFrame(
        {
            "actual": np.asarray(y_true).reshape(-1),
            "predicted": np.asarray(y_pred).reshape(-1),
            "residual": np.asarray(y_true).reshape(-1) - np.asarray(y_pred).reshape(-1),
        }
    )
    payload.to_csv(path, index=False)


def plot_loss_curve(path: Path, history: list[dict[str, float]]) -> None:
    sns.set_theme(style="whitegrid")
    epochs = [int(row["epoch"]) for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="Train loss", linewidth=2)
    ax.plot(epochs, val_loss, label="Val loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pred_vs_actual(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    sns.set_theme(style="whitegrid")
    y_true_flat = np.asarray(y_true).reshape(-1)
    y_pred_flat = np.asarray(y_pred).reshape(-1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true_flat, y_pred_flat, alpha=0.75, edgecolors="none")
    lower = min(y_true_flat.min(), y_pred_flat.min())
    upper = max(y_true_flat.max(), y_pred_flat.max())
    ax.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual (Test)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residuals(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    sns.set_theme(style="whitegrid")
    residuals = np.asarray(y_true).reshape(-1) - np.asarray(y_pred).reshape(-1)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, bins=30, kde=True, ax=ax)
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution (Test)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

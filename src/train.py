from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from utils import (
    compute_regression_metrics,
    plot_loss_curve,
    plot_pred_vs_actual,
    plot_residuals,
    save_metrics_json,
    save_test_predictions_csv,
)

PREDICTOR_COLUMNS = [
    "temperature_c",
    "humidity_percent",
    "wind_speed_kmh",
    "air_quality_index",
    "vegetation_index",
    "distance_to_station_km",
    "response_time_min",
]
TARGET_COLUMN = "fire_risk_score"
EXCLUDED_COLUMNS = {"random_noise", "fire_risk_level"}


class SmallMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


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
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a fire-risk MLP regressor.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to source CSV. Defaults to first CSV in data/.",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument(
        "--save-test-predictions",
        action="store_true",
        help="Write outputs/test_predictions.csv for manual inspection.",
    )
    return parser.parse_args()


def resolve_data_path(data_path: Path | None) -> Path:
    if data_path is not None:
        return data_path

    csv_files = sorted((repo_root() / "data").glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV files found under data/.")
    return csv_files[0]


def validate_columns(dataframe: pd.DataFrame) -> None:
    missing = [column for column in [*PREDICTOR_COLUMNS, TARGET_COLUMN] if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def split_indices(
    n_rows: int,
    seed: int,
    val_size: float,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if val_size <= 0.0 or test_size <= 0.0 or (val_size + test_size) >= 1.0:
        raise ValueError("val_size and test_size must be > 0 and sum to less than 1.")

    rng = np.random.default_rng(seed)
    indices = np.arange(n_rows)
    rng.shuffle(indices)

    n_test = int(round(n_rows * test_size))
    n_val = int(round(n_rows * val_size))
    n_train = n_rows - n_val - n_test

    if min(n_train, n_val, n_test) <= 0:
        raise ValueError("Split sizes produce an empty train/val/test split.")

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


def select_xy(dataframe: pd.DataFrame, indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    subset = dataframe.iloc[indices]
    x = subset[PREDICTOR_COLUMNS].to_numpy(dtype=np.float32)
    y = subset[TARGET_COLUMN].to_numpy(dtype=np.float32).reshape(-1, 1)
    return x, y


def rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((predictions - targets) ** 2)).item()


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    rmses: list[float] = []
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features)
            losses.append(criterion(outputs, targets).item())
            rmses.append(rmse(outputs, targets))

    return float(np.mean(losses)), float(np.mean(rmses))


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for features, batch_targets in loader:
            outputs = model(features.to(device))
            predictions.append(outputs.cpu().numpy().reshape(-1))
            targets.append(batch_targets.cpu().numpy().reshape(-1))
    return np.concatenate(predictions), np.concatenate(targets)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_path = resolve_data_path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    dataframe = pd.read_csv(data_path)
    validate_columns(dataframe)

    accidental_features = EXCLUDED_COLUMNS.intersection(PREDICTOR_COLUMNS)
    if accidental_features:
        raise ValueError(f"Excluded columns unexpectedly present in predictors: {accidental_features}")

    train_idx, val_idx, test_idx = split_indices(
        n_rows=len(dataframe),
        seed=args.seed,
        val_size=args.val_size,
        test_size=args.test_size,
    )

    x_train, y_train = select_xy(dataframe, train_idx)
    x_val, y_val = select_xy(dataframe, val_idx)
    x_test, y_test = select_xy(dataframe, test_idx)

    scaler = StandardScaler().fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    out_dir = repo_root() / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out_dir / "scaler.joblib")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train_scaled), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val_scaled), torch.from_numpy(y_val)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test_scaled), torch.from_numpy(y_test)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = SmallMLP(input_dim=len(PREDICTOR_COLUMNS), hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    history: list[dict[str, float]] = []
    best_val_rmse = float("inf")

    best_model_path = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = float(np.mean(batch_losses))
        val_loss, val_rmse = evaluate(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse": val_rmse,
            }
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_features": PREDICTOR_COLUMNS,
                    "target": TARGET_COLUMN,
                    "epoch": epoch,
                    "val_rmse": val_rmse,
                },
                best_model_path,
            )

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_predictions, test_targets = predict(model, test_loader, device)
    test_metrics = compute_regression_metrics(test_targets, test_predictions)

    metrics = {
        "data_path": str(data_path),
        "seed": args.seed,
        "n_samples": len(dataframe),
        "splits": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "best_val_rmse": checkpoint["val_rmse"],
        "test_metrics": test_metrics,
    }

    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_metrics_json(out_dir / "metrics.json", metrics)
    plot_loss_curve(out_dir / "loss_curve.png", history)
    plot_pred_vs_actual(out_dir / "pred_vs_actual.png", test_targets, test_predictions)
    plot_residuals(out_dir / "residuals.png", test_targets, test_predictions)
    if args.save_test_predictions:
        save_test_predictions_csv(out_dir / "test_predictions.csv", test_targets, test_predictions)

    print(f"Scaler saved to {out_dir / 'scaler.joblib'}")
    print(f"Best checkpoint saved to {best_model_path}")
    print(f"Metrics saved to {out_dir / 'metrics.json'}")
    print(f"Loss curve saved to {out_dir / 'loss_curve.png'}")
    print(f"Predicted-vs-actual plot saved to {out_dir / 'pred_vs_actual.png'}")
    print(f"Residual plot saved to {out_dir / 'residuals.png'}")
    if args.save_test_predictions:
        print(f"Test predictions saved to {out_dir / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from train import PREDICTOR_COLUMNS, SmallMLP
from utils import (
    DEFAULT_HIGH_RISK_THRESHOLD,
    DEFAULT_LOW_RISK_THRESHOLD,
    outputs_dir,
    risk_band_from_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch predictions with the trained fire-risk model.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Path to CSV containing feature columns for inference.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=outputs_dir() / "predictions.csv",
        help="Path to output CSV. Defaults to outputs/predictions.csv.",
    )
    parser.add_argument(
        "--include-risk-band",
        action="store_true",
        help=(
            "Include categorical risk_band column. "
            "Bands are Low (< low-risk-threshold), Medium ([low-risk-threshold, high-risk-threshold]), "
            "High (> high-risk-threshold)."
        ),
    )
    parser.add_argument(
        "--low-risk-threshold",
        type=float,
        default=DEFAULT_LOW_RISK_THRESHOLD,
        help="Lower threshold for risk banding (default: 33.0).",
    )
    parser.add_argument(
        "--high-risk-threshold",
        type=float,
        default=DEFAULT_HIGH_RISK_THRESHOLD,
        help="Upper threshold for risk banding (default: 66.0).",
    )
    return parser.parse_args()


def validate_feature_columns(dataframe: pd.DataFrame) -> None:
    missing_columns = [column for column in PREDICTOR_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(
            "Input CSV is missing required feature columns: "
            f"{missing_columns}. Expected columns: {PREDICTOR_COLUMNS}"
        )


def main() -> None:
    args = parse_args()

    if args.low_risk_threshold >= args.high_risk_threshold:
        raise ValueError("--low-risk-threshold must be lower than --high-risk-threshold.")

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found at {args.input_csv}")

    out_dir = outputs_dir()
    scaler_path = out_dir / "scaler.joblib"
    checkpoint_path = out_dir / "best_model.pt"

    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler artifact not found at {scaler_path}. Run `python src/train.py` first.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}. Run `python src/train.py` first.")

    dataframe = pd.read_csv(args.input_csv)
    validate_feature_columns(dataframe)

    scaler = joblib.load(scaler_path)
    features = dataframe[PREDICTOR_COLUMNS].to_numpy(dtype=np.float32)
    features_scaled = scaler.transform(features).astype(np.float32)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = SmallMLP(input_dim=len(PREDICTOR_COLUMNS))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        predictions = model(torch.from_numpy(features_scaled)).squeeze(-1).numpy()

    output_df = dataframe.copy()
    output_df["predicted_fire_risk_score"] = predictions

    if args.include_risk_band:
        output_df["risk_band"] = output_df["predicted_fire_risk_score"].map(
            lambda score: risk_band_from_score(
                score=score,
                low_threshold=args.low_risk_threshold,
                high_threshold=args.high_risk_threshold,
            )
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_csv, index=False)
    print(f"Wrote predictions to {args.output_csv}")


if __name__ == "__main__":
    main()

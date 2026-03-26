from __future__ import annotations

import argparse
import json

import numpy as np

from model import LinearRegressionGD
from utils import outputs_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run predictions with trained model.")
    parser.add_argument("--x", type=float, help="Single x value for prediction.")
    parser.add_argument(
        "--x-values",
        type=float,
        nargs="+",
        help="One or more x values for batch prediction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = outputs_dir() / "model.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Run `python src/train.py` first."
        )

    payload = json.loads(model_path.read_text(encoding="utf-8"))
    model = LinearRegressionGD.from_dict(payload)

    if args.x is not None:
        result = model.predict(np.array([args.x]))[0]
        print(f"x={args.x:.4f} -> y_pred={result:.4f}")
        return

    if args.x_values:
        x_arr = np.array(args.x_values)
        preds = model.predict(x_arr)
        for x_val, y_hat in zip(x_arr, preds):
            print(f"x={x_val:.4f} -> y_pred={y_hat:.4f}")
        return

    raise ValueError("Provide either --x or --x-values.")


if __name__ == "__main__":
    main()

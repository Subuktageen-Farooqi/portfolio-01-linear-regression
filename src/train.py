from __future__ import annotations

import json

from model import LinearRegressionGD
from utils import dataset_path, load_csv_xy, outputs_dir


def main() -> None:
    data_path = dataset_path()
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. Run `python data/generate_data.py` first."
        )

    x, y = load_csv_xy(data_path)

    model = LinearRegressionGD(learning_rate=0.0009, epochs=6000)
    model.fit(x, y)
    mse = model.mse(x, y)

    out_dir = outputs_dir()
    model_path = out_dir / "model.json"
    metrics_path = out_dir / "metrics.json"

    model_path.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")
    metrics_path.write_text(
        json.dumps({"mse": mse, "n_samples": int(len(x))}, indent=2),
        encoding="utf-8",
    )

    print(f"Model saved to {model_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()

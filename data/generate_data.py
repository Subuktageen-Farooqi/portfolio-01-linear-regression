from __future__ import annotations

from pathlib import Path

import numpy as np


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> None:
    rng = np.random.default_rng(42)
    x = np.linspace(0, 50, 120)
    noise = rng.normal(0, 8.0, size=x.shape[0])
    y = 3.2 * x + 14.0 + noise

    out_path = repo_root() / "data" / "linear_data.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.column_stack((x, y))
    np.savetxt(out_path, arr, delimiter=",", header="x,y", comments="", fmt="%.6f")

    print(f"Wrote dataset to {out_path}")


if __name__ == "__main__":
    main()
